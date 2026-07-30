#include "pch.h"

#include "GBuffer.h"
#include "Core/Model.h"
#include "Renderer.h"
#include "Graphics/GraphicsHelper.h"

void GBufferPass::CreateResources(uint32_t w, uint32_t h)
{
    float blackClear[] = { 0, 0, 0, 0 };
    CreateTexture(m_GBuffer.albedo, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear, 1, 1, "GBuffer_Albedo");
    CreateTexture(m_GBuffer.normal, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear, 1, 1, "GBuffer_Normal");
    CreateTexture(m_GBuffer.material, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear, 1, 1, "GBuffer_Material");
    CreateTexture(m_GBuffer.depth, w, h, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, 1, 1, "GBuffer_Depth");
}

void GBufferPass::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
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

    // 1. Depth Pre-Pass PSO
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.NumRenderTargets = 0;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_DepthPrePassPSO));
    }

    // 2. G-Buffer PSO
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Gbuffer.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Gbuffer.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.NumRenderTargets = 3;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        psoDesc.RTVFormats[2] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        
        // No depth write, using pre-pass
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_GBufferPSO));

        // Create a version of G-Buffer PSO that writes to depth (for when pre-pass is disabled)
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_GBufferWritePSO));
    }
}

void GBufferPass::Execute(ID3D12GraphicsCommandList* cmdList, Model* model, Renderer* renderer,
                           const DirectX::BoundingFrustum& frustum, bool enableDepthPrePass)
{
    // 1. Depth Pre-Pass
    if (enableDepthPrePass)
    {
        MICROPROFILE_SCOPEI("Render", "DepthPrePass", MP_GREY);
        MICROPROFILE_SCOPEGPUI("DepthPrePass", MP_GREY);
        GPU_MARKER(cmdList, L"Depth Pre-Pass");
        GraphicsHelper::TransitionResource(cmdList, m_GBuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        cmdList->SetPipelineState(m_DepthPrePassPSO.Get());

        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_GBuffer.depth.dsvHandle;
        cmdList->OMSetRenderTargets(0, nullptr, FALSE, &dsvHandle);
        cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

        model->Render(cmdList, renderer, frustum, AlphaMode::Opaque);
    }

    // 2. G-Buffer Pass
    {
        MICROPROFILE_SCOPEI("Render", "GBuffer", MP_BLUE);
        MICROPROFILE_SCOPEGPUI("GBuffer", MP_BLUE);
        GPU_MARKER(cmdList, L"GBuffer Pass");
        // Transition G-Buffer targets to RTV state
        GraphicsHelper::TransitionResource(cmdList, m_GBuffer.albedo, D3D12_RESOURCE_STATE_RENDER_TARGET);
        GraphicsHelper::TransitionResource(cmdList, m_GBuffer.normal, D3D12_RESOURCE_STATE_RENDER_TARGET);
        GraphicsHelper::TransitionResource(cmdList, m_GBuffer.material, D3D12_RESOURCE_STATE_RENDER_TARGET);

        float clearColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        cmdList->ClearRenderTargetView(m_GBuffer.albedo.rtvHandle, clearColor, 0, nullptr);
        cmdList->ClearRenderTargetView(m_GBuffer.normal.rtvHandle, clearColor, 0, nullptr);
        cmdList->ClearRenderTargetView(m_GBuffer.material.rtvHandle, clearColor, 0, nullptr);

        D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = { m_GBuffer.albedo.rtvHandle, m_GBuffer.normal.rtvHandle, m_GBuffer.material.rtvHandle };
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_GBuffer.depth.dsvHandle;

        // If pre-pass was skipped, we MUST clear the depth buffer here
        if (!enableDepthPrePass)
        {
            GraphicsHelper::TransitionResource(cmdList, m_GBuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        }

        cmdList->OMSetRenderTargets(_countof(rtvs), rtvs, FALSE, &dsvHandle);

        cmdList->SetPipelineState(enableDepthPrePass ? m_GBufferPSO.Get() : m_GBufferWritePSO.Get());

        model->Render(cmdList, renderer, frustum, AlphaMode::Opaque);
    }
}
