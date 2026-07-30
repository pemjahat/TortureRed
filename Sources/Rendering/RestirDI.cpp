#include "pch.h"

#include "RestirDI.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

void RestirDI::CreateResources(uint32_t internalWidth, uint32_t internalHeight)
{
    const UINT pixelCount = internalWidth * internalHeight;
    for (int i = 0; i < 2; ++i)
        CreateStructuredBuffer(m_DIReservoirBuffer[i], sizeof(DIRreservoir), pixelCount,
                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, i == 0 ? "SB_DIReservoir0" : "SB_DIReservoir1");
    CreateStructuredBuffer(m_DIReservoirIntermediate, sizeof(DIRreservoir), pixelCount,
                           D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_DIReservoirIntermediate");
    // Split DI intermediates for SSO bridge path
    CreateTexture(m_DIDiffuseIntermediate, internalWidth, internalHeight,
                  DXGI_FORMAT_R16G16B16A16_FLOAT,
                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_DIDiffuseIntermediate");
    CreateTexture(m_DISpecularIntermediate, internalWidth, internalHeight,
                  DXGI_FORMAT_R16G16B16A16_FLOAT,
                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_DISpecularIntermediate");
}

void RestirDI::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = rootSignature;

    auto CompileAndCreate = [&](const char* file, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso)
    {
        auto cs = GraphicsHelper::CompileShader(file, "main", "cs_6_6");
        if (!cs.empty())
        {
            desc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
        }
    };

    CompileAndCreate("Shaders/RestirDI_Temporal.hlsl",        m_RestirDITemporalPSO);
    CompileAndCreate("Shaders/RestirDI_Spatial.hlsl",         m_RestirDISpatialPSO);
    CompileAndCreate("Shaders/RestirDI_SplitShade.hlsl",      m_RestirDISplitShadePSO);
}

void RestirDI::Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                        Model* model, const FrameConstants& frame,
                        D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, D3D12_GPU_VIRTUAL_ADDRESS tlasGPUAddress,
                        D3D12_GPU_VIRTUAL_ADDRESS lightsBufferAddress, D3D12_GPU_VIRTUAL_ADDRESS lightLUTBufferAddress,
                        GPUTexture& fullScreenDebugTex, GPUTexture& finalDiffuseTex, GPUTexture& finalSpecularTex,
                        ID3D12PipelineState* nrdStoreShadingOutputPSO,
                        uint32_t internalWidth, uint32_t internalHeight)
{
    if (!frame.enableRestirDI) return;

    const int curr = m_CurrentDIReservoirIndex;
    const int prev = 1 - curr;

    // Ensure all DI resources are in UAV state
    GraphicsHelper::TransitionResource(cmdList, m_DIReservoirBuffer[0],    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DIReservoirBuffer[1],    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DIReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DIDiffuseIntermediate,   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DISpecularIntermediate,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    const bool diDebugActive = frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF;
    if (diDebugActive)
    {
        GraphicsHelper::TransitionResource(cmdList, fullScreenDebugTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);

    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    cmdList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    cmdList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));
    cmdList->SetComputeRootShaderResourceView(4, tlasGPUAddress);
    cmdList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    cmdList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    cmdList->SetComputeRootShaderResourceView(10, lightsBufferAddress);
    cmdList->SetComputeRootShaderResourceView(11, lightLUTBufferAddress);

    const UINT W = internalWidth, H = internalHeight;
    const UINT gx = (W + 7) / 8, gy = (H + 7) / 8;

    BindlessIndices indices = {};

    // --- Pass 1: Combined Initial Sampling + Temporal Resampling ---
    // InputIdx0 = DIRreservoirBuffer[prev] (previous-frame temporal output)
    // OutputIdx0 = DIRreservoirBuffer[curr]
    indices = {};
    indices.InputIdx0  = m_DIReservoirBuffer[prev].srvIndex;
    indices.OutputIdx0 = m_DIReservoirBuffer[curr].uavIndex;
    indices.OutputIdx1 = diDebugActive ? fullScreenDebugTex.uavIndex : UINT(-1);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    cmdList->SetPipelineState(m_RestirDITemporalPSO.Get());
    {
        MICROPROFILE_SCOPEGPUI("DI_Temporal", MP_RED);
        cmdList->Dispatch(gx, gy, 1);
    }

    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(m_DIReservoirBuffer[curr].resource.Get());
        cmdList->ResourceBarrier(1, &b);
    }

    // --- Pass 2: Spatial Resampling ---
    // InputIdx0 = DIRreservoirBuffer[curr], OutputIdx0 = DIRreservoirIntermediate
    indices = {};
    indices.InputIdx0  = m_DIReservoirBuffer[curr].srvIndex;
    indices.OutputIdx0 = m_DIReservoirIntermediate.uavIndex;
    indices.OutputIdx1 = diDebugActive ? fullScreenDebugTex.uavIndex : UINT(-1);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    cmdList->SetPipelineState(m_RestirDISpatialPSO.Get());
    {
        MICROPROFILE_SCOPEGPUI("DI_Spatial", MP_RED);
        cmdList->Dispatch(gx, gy, 1);
    }

    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(m_DIReservoirIntermediate.resource.Get());
        cmdList->ResourceBarrier(1, &b);
    }

    // --- Pass 3: Split Shade — per-lobe NRD-normalized output ---
    // InputIdx0 = DIRreservoirIntermediate, OutputIdx0 = DIDiffuseIntermediate, OutputIdx1 = DISpecularIntermediate
    if (m_RestirDISplitShadePSO)
    {
        indices = {};
        indices.InputIdx0  = m_DIReservoirIntermediate.srvIndex;
        indices.OutputIdx0 = m_DIDiffuseIntermediate.uavIndex;
        indices.OutputIdx1 = m_DISpecularIntermediate.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        cmdList->SetPipelineState(m_RestirDISplitShadePSO.Get());
        {
            MICROPROFILE_SCOPEGPUI("DI_SplitShade", MP_RED);
            cmdList->Dispatch(gx, gy, 1);
        }

        D3D12_RESOURCE_BARRIER splitBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_DIDiffuseIntermediate.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_DISpecularIntermediate.resource.Get()),
        };
        cmdList->ResourceBarrier(_countof(splitBarriers), splitBarriers);
    }

    // --- Pass 3b: StoreShadingOutput Call 1 (DI base) ---
    // Writes DI intermediates into finalDiffuseTex / finalSpecularTex (overwrite).
    // Always dispatched when DI is active (regardless of NRD state).
    if (nrdStoreShadingOutputPSO && m_RestirDISplitShadePSO)
    {
        GraphicsHelper::TransitionResource(cmdList, m_DIDiffuseIntermediate,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_DISpecularIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, finalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, finalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        indices = {};
        indices.InputIdx0  = m_DIDiffuseIntermediate.srvIndex;
        indices.InputIdx1  = m_DISpecularIntermediate.srvIndex;
        indices.OutputIdx0 = finalDiffuseTex.uavIndex;
        indices.OutputIdx1 = finalSpecularTex.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        const UINT isFirstPass = 1u;
        cmdList->SetComputeRoot32BitConstants(13, 1, &isFirstPass, 0);
        cmdList->SetPipelineState(nrdStoreShadingOutputPSO);
        cmdList->Dispatch(gx, gy, 1);

        D3D12_RESOURCE_BARRIER ssoBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(finalDiffuseTex.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(finalSpecularTex.resource.Get()),
        };
        cmdList->ResourceBarrier(_countof(ssoBarriers), ssoBarriers);
    }

    m_CurrentDIReservoirIndex = prev; // Swap for next frame
}
