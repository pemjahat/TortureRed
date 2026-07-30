#include "pch.h"

#include "Transparency.h"
#include "Core/Utility.h"
#include "Core/Model.h"
#include "Renderer.h"
#include "Graphics/GraphicsHelper.h"

void Transparency::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    auto GetDefaultPsoDesc = [&]() {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = rootSignature;
        desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        desc.RasterizerState.FrontCounterClockwise = TRUE;
        desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        desc.SampleMask = UINT_MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.SampleDesc.Count = 1;
        return desc;
    };

    auto GetTransparencyBlendDesc = [&]() {
        D3D12_RENDER_TARGET_BLEND_DESC blendDesc = {};
        blendDesc.BlendEnable = TRUE;
        blendDesc.LogicOpEnable = FALSE;
        blendDesc.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        blendDesc.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        blendDesc.BlendOp = D3D12_BLEND_OP_ADD;
        blendDesc.SrcBlendAlpha = D3D12_BLEND_ONE;
        blendDesc.DestBlendAlpha = D3D12_BLEND_ZERO;
        blendDesc.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        blendDesc.LogicOp = D3D12_LOGIC_OP_NOOP;
        blendDesc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        return blendDesc;
    };

    // 1. Transparent PSO (Forward)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Forward.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Forward.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        
        // Enable Alpha Blending
        psoDesc.BlendState.RenderTarget[0] = GetTransparencyBlendDesc();

        // Double sided 
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        
        // Read-only depth
        psoDesc.DepthStencilState.DepthEnable = TRUE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;

        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM; // Backbuffer format
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

        CHECK_HR(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_TransparentPSO)), "CreateGraphicsPipelineState for Transparent PSO failed");
    }

    // 2. Transparent HDR PSO (Forward — renders to R16G16B16A16_FLOAT for TAA input, no tonemapping)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Forward.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Forward.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };

        psoDesc.BlendState.RenderTarget[0] = GetTransparencyBlendDesc();

        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;

        psoDesc.DepthStencilState.DepthEnable = TRUE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;

        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT; // HDR intermediate texture format
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

        CHECK_HR(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_TransparentHdrPSO)), "CreateGraphicsPipelineState for Transparent HDR PSO failed");
    }
}

void Transparency::Execute(ID3D12GraphicsCommandList* cmdList, Model* model, Renderer* renderer,
                            const DirectX::BoundingFrustum& frustum, bool rasterTaaActive,
                            uint32_t outputWidth, uint32_t outputHeight)
{
    MICROPROFILE_SCOPEI("Render", "Transparency", MP_ORANGE);
    MICROPROFILE_SCOPEGPUI("Transparency", MP_ORANGE);
    GPU_MARKER(cmdList, L"Transparency Pass");

    GBuffer& gbuffer = renderer->GetGBuffer();

    // Ensure depth is in read state for forward pass
    GraphicsHelper::TransitionResource(cmdList, gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_READ);

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = gbuffer.depth.dsvHandle;

    if (rasterTaaActive)
    {
        // Render into the HDR intermediate texture (same target as the geometry pass).
        // Viewport stays at internal resolution — already set at the top of Render().
        GraphicsHelper::TransitionResource(cmdList, renderer->GetRasterHdrOutputTex(), D3D12_RESOURCE_STATE_RENDER_TARGET);
        D3D12_CPU_DESCRIPTOR_HANDLE hdrRtvHandle = renderer->GetRasterHdrOutputTex().rtvHandle;
        cmdList->OMSetRenderTargets(1, &hdrRtvHandle, FALSE, &dsvHandle);
    }
    else
    {
        // Non-TAA: render directly to the back buffer at output resolution.
        // Internal resolution == output resolution in this path (enforced at
        // initialization and on AA mode toggle), so the G-buffer depth covers
        // the full viewport and can be bound as DSV safely.
        D3D12_VIEWPORT outputViewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(outputWidth), static_cast<float>(outputHeight));
        D3D12_RECT outputScissor = CD3DX12_RECT(0, 0, outputWidth, outputHeight);
        cmdList->RSSetViewports(1, &outputViewport);
        cmdList->RSSetScissorRects(1, &outputScissor);

        renderer->TransitionBackBuffer(D3D12_RESOURCE_STATE_RENDER_TARGET);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = renderer->GetCurrentBackBufferRTV();
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
    }

    // TAA path: use HDR PSO (R16G16B16A16_FLOAT, no tonemapping) to match RasterHdrOutputTex.
    // Non-TAA path: use LDR PSO (R8G8B8A8_UNORM, with tonemapping) to match back buffer.
    ID3D12PipelineState* pso = rasterTaaActive ? m_TransparentHdrPSO.Get() : m_TransparentPSO.Get();

    if (pso)
    {
        cmdList->SetPipelineState(pso);
        model->Render(cmdList, renderer, frustum, AlphaMode::Blend);
    }
}
