#include "pch.h"

#include "DebugTextRenderer.h"
#include "Graphics/GraphicsHelper.h"
#include "Core/Utility.h"

#include <imgui.h>
#include <iostream>

// -----------------------------------------------------------------------------
// Initialize
// -----------------------------------------------------------------------------
bool DebugTextRenderer::Initialize(ID3D12Device* device, ID3D12CommandQueue* cmdQueue, ID3D12RootSignature* mainRootSignature)
{
    // Shared render-data buffer: counters + text instances + line instances.
    // Created in UAV state; Render() leaves it in UAV state each frame so
    // producers can always append via ResourceDescriptorHeap.
    if (!CreateBuffer(m_RenderData, sizeof(Data),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      true, true, "SB_DebugRenderData"))
    {
        std::cerr << "[DebugText] Failed to create render-data buffer" << std::endl;
        return false;
    }

    // Indirect draw args: [0] = glyph quads, [1] = lines
    if (!CreateStructuredBuffer(m_IndirectArgs, 16, 2,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_DebugTextArgs"))
    {
        std::cerr << "[DebugText] Failed to create indirect args buffer" << std::endl;
        return false;
    }

    // Plain DRAW command signature (no root-arg changes)
    {
        D3D12_INDIRECT_ARGUMENT_DESC drawArg = {};
        drawArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;

        D3D12_COMMAND_SIGNATURE_DESC sigDesc = {};
        sigDesc.ByteStride       = sizeof(D3D12_DRAW_ARGUMENTS);
        sigDesc.NumArgumentDescs = 1;
        sigDesc.pArgumentDescs   = &drawArg;

        CHECK_HR(device->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&m_DrawCommandSignature)),
                 "[DebugText] CreateCommandSignature failed");
    }

    if (!CreateFontAtlas(device, cmdQueue))
        return false;

    return CreatePipelines(device, mainRootSignature);
}

// -----------------------------------------------------------------------------
// CreateFontAtlas — rasterize the ImGui default font into an RGBA8 texture and
// extract per-glyph metrics.
// -----------------------------------------------------------------------------
bool DebugTextRenderer::CreateFontAtlas(ID3D12Device* device, ID3D12CommandQueue* cmdQueue)
{
    ImFontAtlas atlas;
    ImFont* font = atlas.AddFontDefault();

    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    atlas.GetTexDataAsRGBA32(&pixels, &width, &height);

    if (!CreateTexture(m_FontAtlas, static_cast<UINT>(width), static_cast<UINT>(height),
                       DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COMMON, nullptr, 1, 1, "Tex_DebugFontAtlas"))
    {
        std::cerr << "[DebugText] Failed to create font atlas texture" << std::endl;
        return false;
    }

    // Glyph metrics for ASCII
    DebugGlyph glyphs[128] = {};
    for (const ImFontGlyph& g : font->Glyphs)
    {
        if (g.Codepoint >= 128)
            continue;
        DebugGlyph& d   = glyphs[g.Codepoint];
        d.MinUV         = { g.U0, g.V0 };
        d.MaxUV         = { g.U1, g.V1 };
        d.Dimensions    = { g.X1 - g.X0, g.Y1 - g.Y0 };
        d.Offset        = { g.X0, g.Y0 };
        d.AdvanceX      = g.AdvanceX;
    }
    m_FontSize = font->FontSize;

    if (!CreateStructuredBuffer(m_GlyphData, sizeof(DebugGlyph), 128,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                "SB_DebugGlyphData"))
    {
        std::cerr << "[DebugText] Failed to create glyph data buffer" << std::endl;
        return false;
    }

    // One-shot upload of texture + glyph buffer
    const UINT64 textureMemSize = static_cast<UINT64>(width) * height * 4;
    const UINT64 glyphMemSize   = sizeof(glyphs);

    Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer;
    {
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width            = textureMemSize + glyphMemSize;
        desc.Height           = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels        = 1;
        desc.Format           = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        CHECK_HR(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                 IID_PPV_ARGS(&uploadBuffer)),
                 "[DebugText] CreateCommittedResource (upload) failed");

        uint8_t* mapped = nullptr;
        uploadBuffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
        memcpy(mapped, pixels, textureMemSize);
        memcpy(mapped + textureMemSize, glyphs, glyphMemSize);
        uploadBuffer->Unmap(0, nullptr);
    }

    // Record + submit a small copy command list and wait for it
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
    CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
             "[DebugText] CreateCommandAllocator failed");
    CHECK_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                       IID_PPV_ARGS(&list)),
             "[DebugText] CreateCommandList failed");

    // Texture: COMMON → COPY_DEST → copy → PIXEL_SHADER_RESOURCE
    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_FontAtlas.resource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1, &barrier);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
        UINT numRows = 0;
        UINT64 rowSize = 0, totalSize = 0;
        D3D12_RESOURCE_DESC texDesc = m_FontAtlas.resource->GetDesc();
        device->GetCopyableFootprints(&texDesc, 0, 1, 0, &layout, &numRows, &rowSize, &totalSize);

        // Rows in the upload buffer are tightly packed (width*4); copy row-by-row
        // only if the runtime wants a different pitch.
        if (layout.Footprint.RowPitch == static_cast<UINT>(width) * 4)
        {
            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource        = m_FontAtlas.resource.Get();
            dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.SubresourceIndex = 0;

            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource       = uploadBuffer.Get();
            src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = layout;

            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        else
        {
            std::cerr << "[DebugText] Unexpected font atlas row pitch" << std::endl;
            return false;
        }

        barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_FontAtlas.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &barrier);
    }

    // Glyph buffer: COMMON → COPY_DEST → copy → GENERIC_READ
    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_GlyphData.resource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1, &barrier);

        list->CopyBufferRegion(m_GlyphData.resource.Get(), 0, uploadBuffer.Get(), textureMemSize, glyphMemSize);

        barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_GlyphData.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
        list->ResourceBarrier(1, &barrier);
    }

    CHECK_HR(list->Close(), "[DebugText] Close command list failed");
    ID3D12CommandList* lists[] = { list.Get() };
    cmdQueue->ExecuteCommandLists(1, lists);

    // Fence-wait for the copy to finish before the upload buffer dies
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "[DebugText] CreateFence failed");
    HANDLE evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    cmdQueue->Signal(fence.Get(), 1);
    if (fence->GetCompletedValue() < 1)
    {
        fence->SetEventOnCompletion(1, evt);
        WaitForSingleObject(evt, INFINITE);
    }
    CloseHandle(evt);

    std::cout << "[DebugText] Font atlas created (" << width << "x" << height << ", " << (int)m_FontSize << "px)" << std::endl;
    return true;
}

// -----------------------------------------------------------------------------
// CreatePipelines
// -----------------------------------------------------------------------------
bool DebugTextRenderer::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* mainRootSignature)
{
    // Args builder (CS)
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/DebugTextRender.hlsl", "DebugTextBuildArgsCS", "cs_6_6");
        if (!cs.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = mainRootSignature;
            desc.CS = { cs.data(), cs.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_BuildArgsPSO)),
                     "[DebugText] CreateComputePipelineState (args builder) failed");
        }
    }

    auto vsGlyph = GraphicsHelper::CompileShader("Shaders/DebugTextRender.hlsl", "DebugGlyphVS", "vs_6_6");
    auto psGlyph = GraphicsHelper::CompileShader("Shaders/DebugTextRender.hlsl", "DebugGlyphPS", "ps_6_6");
    auto vsLine  = GraphicsHelper::CompileShader("Shaders/DebugTextRender.hlsl", "DebugLineVS",  "vs_6_6");
    auto psLine  = GraphicsHelper::CompileShader("Shaders/DebugTextRender.hlsl", "DebugLinePS",  "ps_6_6");

    auto baseDesc = [&](D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, D3D12_PRIMITIVE_TOPOLOGY_TYPE topology)
    {
        memset(&desc, 0, sizeof(desc));
        desc.pRootSignature = mainRootSignature;
        desc.InputLayout    = { nullptr, 0 };
        // Alpha blend over the backbuffer
        desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        desc.BlendState.RenderTarget[0].BlendEnable           = TRUE;
        desc.BlendState.RenderTarget[0].SrcBlend              = D3D12_BLEND_SRC_ALPHA;
        desc.BlendState.RenderTarget[0].DestBlend             = D3D12_BLEND_INV_SRC_ALPHA;
        desc.BlendState.RenderTarget[0].BlendOp               = D3D12_BLEND_OP_ADD;
        desc.BlendState.RenderTarget[0].SrcBlendAlpha         = D3D12_BLEND_ONE;
        desc.BlendState.RenderTarget[0].DestBlendAlpha        = D3D12_BLEND_INV_SRC_ALPHA;
        desc.BlendState.RenderTarget[0].BlendOpAlpha          = D3D12_BLEND_OP_ADD;
        desc.RasterizerState            = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        desc.RasterizerState.CullMode   = D3D12_CULL_MODE_NONE;
        desc.DepthStencilState          = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        desc.DepthStencilState.DepthEnable = FALSE;
        desc.PrimitiveTopologyType      = topology;
        desc.NumRenderTargets           = 1;
        desc.RTVFormats[0]              = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count           = 1;
        desc.SampleMask                 = D3D12_DEFAULT_SAMPLE_MASK;
    };

    if (!vsGlyph.empty() && !psGlyph.empty())
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc;
        baseDesc(desc, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        desc.VS = { vsGlyph.data(), vsGlyph.size() };
        desc.PS = { psGlyph.data(), psGlyph.size() };
        CHECK_HR(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_GlyphPSO)),
                 "[DebugText] CreateGraphicsPipelineState (glyph) failed");
    }

    if (!vsLine.empty() && !psLine.empty())
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc;
        baseDesc(desc, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
        desc.VS = { vsLine.data(), vsLine.size() };
        desc.PS = { psLine.data(), psLine.size() };
        CHECK_HR(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_LinePSO)),
                 "[DebugText] CreateGraphicsPipelineState (line) failed");
    }

    return true;
}

// -----------------------------------------------------------------------------
// Render — build args from the producers' counters (resetting them), then
// rasterize all recorded text/lines onto the render target.
// -----------------------------------------------------------------------------
void DebugTextRenderer::Render(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                               D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                               D3D12_CPU_DESCRIPTOR_HANDLE rtv, uint32_t width, uint32_t height)
{
    if (!m_BuildArgsPSO || !m_GlyphPSO || !m_LinePSO)
        return;

    // Producers → args builder ordering
    D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_RenderData.resource.Get());
    cmdList->ResourceBarrier(1, &uavBarrier);

    DebugTextRenderParams params = {};
    params.DataSRVIdx      = static_cast<uint32_t>(m_RenderData.srvIndex);
    params.DataUAVIdx      = static_cast<uint32_t>(m_RenderData.uavIndex);
    params.ArgsUAVIdx      = static_cast<uint32_t>(m_IndirectArgs.uavIndex);
    params.GlyphSRVIdx     = static_cast<uint32_t>(m_GlyphData.srvIndex);
    params.FontAtlasSRVIdx = static_cast<uint32_t>(m_FontAtlas.srvIndex);
    params.TargetWidth     = static_cast<float>(width);
    params.TargetHeight    = static_cast<float>(height);

    // 1. Args builder (reads + resets counters)
    GraphicsHelper::TransitionResource(cmdList, m_IndirectArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetPipelineState(m_BuildArgsPSO.Get());
    cmdList->SetComputeRoot32BitConstants(12, sizeof(DebugTextRenderParams) / 4, &params, 0); // b1
    cmdList->Dispatch(1, 1, 1);

    // 2. Raster onto the backbuffer
    GraphicsHelper::TransitionResource(cmdList, m_IndirectArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    GraphicsHelper::TransitionResource(cmdList, m_RenderData, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_FontAtlas, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
    D3D12_RECT scissor      = CD3DX12_RECT(0, 0, static_cast<LONG>(width), static_cast<LONG>(height));
    cmdList->RSSetViewports(1, &viewport);
    cmdList->RSSetScissorRects(1, &scissor);

    cmdList->SetGraphicsRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetGraphicsRootConstantBufferView(0, frameCBAddress);
    cmdList->SetGraphicsRoot32BitConstants(12, sizeof(DebugTextRenderParams) / 4, &params, 0); // b1

    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmdList->SetPipelineState(m_GlyphPSO.Get());
    cmdList->ExecuteIndirect(m_DrawCommandSignature.Get(), 1, m_IndirectArgs.resource.Get(), 0, nullptr, 0);

    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
    cmdList->SetPipelineState(m_LinePSO.Get());
    cmdList->ExecuteIndirect(m_DrawCommandSignature.Get(), 1, m_IndirectArgs.resource.Get(), 16, nullptr, 0);

    // Leave the render-data buffer in UAV state for next frame's producers
    GraphicsHelper::TransitionResource(cmdList, m_RenderData, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
