#include "pch.h"

#include "DeferredLighting.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Renderer.h"
#include "Graphics/GraphicsHelper.h"

void DeferredLighting::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
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

    // 1. Lighting PSO (LDR — renders to R8G8B8A8_UNORM back buffer with tonemapping)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_LightingPSO));
    }

    // 2. Lighting HDR PSO (renders to R16G16B16A16_FLOAT for TAA input — no tonemapping in shader)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_LightingHdrPSO));
    }
}

void DeferredLighting::CreateLightsBuffer()
{
    // Create Structured Buffer directly on Upload Heap for easy per-frame updates without command list
    CHECK_BOOL(CreateStructuredBuffer(m_LightsBuffer, sizeof(LightConstants), m_MaxLights, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, "SB_LightsBuffer"), "CreateLightsBuffer failed");
}

void DeferredLighting::UpdateLightsBuffer(const std::vector<LightConstants>& lights)
{
    if (lights.empty()) return;
    UINT numLights = std::min((UINT)lights.size(), m_MaxLights);

    // Calculate selection PDFs before copying to GPU
    std::vector<LightConstants> lightsWithPDF = lights;
    float totalWeight = 0.0f;
    std::vector<float> weights(numLights);

    // Light 0 is treated as the main directional light and is not part of the local-light LUT domain.
    lightsWithPDF[0].selectionPDF = 0.0f;

    uint32_t localLightCount = 0;
    for (UINT i = 1; i < numLights; ++i) {
        float w = 0.0f;
        if (lights[i].position.w > 0.5f) {
            float luminance = 0.2126f * lights[i].color.x + 0.7152f * lights[i].color.y + 0.0722f * lights[i].color.z;
            w = lights[i].intensity * luminance;
            localLightCount++;
        }
        weights[i] = w;
        totalWeight += w;
    }

    for (UINT i = 1; i < numLights; ++i) {
        if (lights[i].position.w <= 0.5f) {
            lightsWithPDF[i].selectionPDF = 0.0f;
        } else if (totalWeight > 0.0f) {
            lightsWithPDF[i].selectionPDF = weights[i] / totalWeight;
        } else if (localLightCount > 0) {
            lightsWithPDF[i].selectionPDF = 1.0f / float(localLightCount);
        } else {
            lightsWithPDF[i].selectionPDF = 0.0f;
        }
    }

    memcpy(m_LightsBuffer.cpuPtr, lightsWithPDF.data(), numLights * sizeof(LightConstants));

    // Also update the LUT buffer for importance sampling
    UpdateLightLUTBuffer(lightsWithPDF);
}

void DeferredLighting::CreateLightLUTBuffer()
{
    // LUT buffer: 256 uint entries for O(1) light sampling
    // Each entry maps a CDF bucket to a light index
    CHECK_BOOL(CreateBuffer(m_LightLUTBuffer, sizeof(uint32_t) * LIGHT_LUT_RESOLUTION, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, true, false, "SB_LightLUTBuffer"), "CreateLightLUTBuffer failed");
}

void DeferredLighting::UpdateLightLUTBuffer(const std::vector<LightConstants>& lights)
{
    if (lights.empty() || m_LightLUTBuffer.cpuPtr == nullptr) return;

    UINT numLights = std::min((UINT)lights.size(), m_MaxLights);
    std::vector<uint32_t> lut(LIGHT_LUT_RESOLUTION);
    std::vector<float> weights(numLights);
    std::vector<uint32_t> localLightIndices;
    localLightIndices.reserve(numLights > 0 ? numLights - 1 : 0);

    // Build a local-light-only distribution. Light 0 remains deterministic and bypasses this LUT.
    float totalWeight = 0.0f;
    for (UINT i = 1; i < numLights; ++i) {
        float w = 0.0f;
        if (lights[i].position.w > 0.5f) {
            float luminance = 0.2126f * lights[i].color.x + 0.7152f * lights[i].color.y + 0.0722f * lights[i].color.z;
            w = lights[i].intensity * luminance;
            localLightIndices.push_back(i);
        }
        weights[i] = w;
        totalWeight += w;
    }

    if (localLightIndices.empty()) {
        std::fill(lut.begin(), lut.end(), 0u);
        memcpy(m_LightLUTBuffer.cpuPtr, lut.data(), LIGHT_LUT_RESOLUTION * sizeof(uint32_t));
        return;
    }

    if (totalWeight <= 0.0f) {
        for (UINT i = 0; i < LIGHT_LUT_RESOLUTION; ++i) {
            lut[i] = localLightIndices[i % localLightIndices.size()];
        }
        memcpy(m_LightLUTBuffer.cpuPtr, lut.data(), LIGHT_LUT_RESOLUTION * sizeof(uint32_t));
        return;
    }

    // Build CDF
    std::vector<float> cdf(numLights);
    float cumulative = 0.0f;
    for (UINT i = 1; i < numLights; ++i) {
        cumulative += weights[i] / totalWeight;
        cdf[i] = cumulative;
    }
    cdf[numLights - 1] = 1.0f;

    // Build LUT: for each LUT entry, find which light index to sample
    for (UINT i = 0; i < LIGHT_LUT_RESOLUTION; ++i) {
        float u = (i + 0.5f) / float(LIGHT_LUT_RESOLUTION); // Center of bin

        // Binary search in CDF to find light index
        uint32_t lightIdx = localLightIndices.back();
        for (UINT j = 1; j < numLights; ++j) {
            if (weights[j] > 0.0f && u <= cdf[j]) {
                lightIdx = j;
                break;
            }
        }
        lut[i] = lightIdx;
    }

    // Copy to GPU
    memcpy(m_LightLUTBuffer.cpuPtr, lut.data(), LIGHT_LUT_RESOLUTION * sizeof(uint32_t));
}

void DeferredLighting::Execute(ID3D12GraphicsCommandList* cmdList, Renderer* renderer, Model* model,
                                const FrameConstants& frame, bool rasterTaaActive, bool debugActive,
                                bool debugShadowMap, uint32_t outputWidth, uint32_t outputHeight)
{
    MICROPROFILE_SCOPEI("Render", "Lighting", MP_CYAN);
    MICROPROFILE_SCOPEGPUI("Lighting", MP_CYAN);
    GPU_MARKER(cmdList, debugActive ? L"Full-Screen Debug Pass" : L"Lighting Pass");
    BindlessIndices indices = {};

    GBuffer& gbuffer = renderer->GetGBuffer();

    // Transition G-Buffer targets to SRV state
    GraphicsHelper::TransitionResource(cmdList, gbuffer.albedo, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, gbuffer.normal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, gbuffer.material, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, gbuffer.depth, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // FinalDiffuse/FinalSpecular contain NRD-normalized radiance from ReSTIR passes.
    if ((frame.enableRestirDI || frame.enableRasterIndirectGI) && !debugActive)
    {
        GraphicsHelper::TransitionResource(cmdList, renderer->GetFinalDiffuseTex(),  D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, renderer->GetFinalSpecularTex(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        indices.InputIdx0 = renderer->GetFinalDiffuseTex().srvIndex;
        indices.InputIdx1 = renderer->GetFinalSpecularTex().srvIndex;
    }

    if (debugActive)
    {
        GraphicsHelper::TransitionResource(cmdList, renderer->GetFullScreenDebugTex(),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        indices.InputIdx0 = renderer->GetFullScreenDebugTex().srvIndex;
    }

    cmdList->SetGraphicsRootShaderResourceView(1, model->GetMaterialBufferAddress());
    cmdList->SetGraphicsRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    cmdList->SetGraphicsRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    cmdList->SetGraphicsRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());

    cmdList->SetGraphicsRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices

    if (rasterTaaActive)
    {
        // Render to internal-res HDR texture for TAA input
        GraphicsHelper::TransitionResource(cmdList, renderer->GetRasterHdrOutputTex(), D3D12_RESOURCE_STATE_RENDER_TARGET);

        D3D12_CPU_DESCRIPTOR_HANDLE hdrRtvHandle = renderer->GetRasterHdrOutputTex().rtvHandle;
        cmdList->OMSetRenderTargets(1, &hdrRtvHandle, FALSE, nullptr);

        const float clearColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        cmdList->ClearRenderTargetView(hdrRtvHandle, clearColor, 0, nullptr);

        cmdList->SetPipelineState(debugActive
            ? renderer->GetFullScreenDebugHdrPSO()
            : m_LightingHdrPSO.Get());
    }
    else
    {
        // Switch to output resolution viewport for direct-to-backbuffer rendering
        D3D12_VIEWPORT outputViewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(outputWidth), static_cast<float>(outputHeight));
        D3D12_RECT outputScissor = CD3DX12_RECT(0, 0, outputWidth, outputHeight);
        cmdList->RSSetViewports(1, &outputViewport);
        cmdList->RSSetScissorRects(1, &outputScissor);

        renderer->TransitionBackBuffer(D3D12_RESOURCE_STATE_RENDER_TARGET);

        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = renderer->GetCurrentBackBufferRTV();
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

        const float clearColor[] = { renderer->m_BackgroundColor[0], renderer->m_BackgroundColor[1], renderer->m_BackgroundColor[2], 1.0f };
        cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

        cmdList->SetPipelineState(
            debugShadowMap ? renderer->GetDebugPSO() :
            debugActive    ? renderer->GetFullScreenDebugPSO() :
                              m_LightingPSO.Get());
    }

    cmdList->DrawInstanced(3, 1, 0, 0); // Fullscreen triangle
}
