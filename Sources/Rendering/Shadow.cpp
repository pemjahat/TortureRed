#include "pch.h"

#include "Shadow.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

bool Shadow::CreateResources()
{
    if (!CreateTexture(m_ShadowMap, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, DXGI_FORMAT_R32_TYPELESS,
                        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                        nullptr, 1, 1, "Tex_ShadowMap"))
    {
        std::cerr << "Failed to create shadow map texture" << std::endl;
        return false;
    }

    return true;
}

void Shadow::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = rootSignature;
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.RasterizerState.FrontCounterClockwise = TRUE;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.SampleDesc.Count = 1;

    std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");
    desc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
    desc.RasterizerState.DepthBias = 1000;
    desc.RasterizerState.SlopeScaledDepthBias = 1.5f;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    desc.NumRenderTargets = 0;

    CHECK_HR(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_ShadowPSO)), "CreateGraphicsPipelineState for Shadow PSO failed");
}
