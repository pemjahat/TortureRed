#include "pch.h"

#include "TAA.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

void TAA::CreateResources(uint32_t outputW, uint32_t outputH, uint32_t internalW, uint32_t internalH)
{
    // Output-resolution textures (shared by both modes)
    for (int i = 0; i < 2; ++i)
    {
        CreateTexture(m_TaaHistoryTex[i], outputW, outputH,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, i == 0 ? "Tex_TaaHistory0" : "Tex_TaaHistory1");
    }

    CreateTexture(m_TaaReprojectedHistoryTex, outputW, outputH,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_TaaReprojectedHistory");

    CreateTexture(m_TaaClosestVelocityTex, outputW, outputH,
        DXGI_FORMAT_R16G16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_TaaClosestVelocity");

    CreateTexture(m_TaaOutputTex, outputW, outputH,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_TaaOutput");

    m_TaaEnabled = true;
    m_TaaHistoryIndex = 0;

    std::cout << "TAA resources created: output=" << outputW << "x" << outputH
              << " internal=" << internalW << "x" << internalH << std::endl;
}

void TAA::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
    computeDesc.pRootSignature = rootSignature;

    // Naive TSR PSOs
    auto reprojectCS = GraphicsHelper::CompileShader("Shaders/NaiveTsr_Reproject.hlsl", "main", "cs_6_6");
    if (!reprojectCS.empty())
    {
        computeDesc.CS = { reprojectCS.data(), reprojectCS.size() };
        CHECK_HR(device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NaiveTsrReprojectPSO)),
            "Failed to create NaiveTsr Reproject PSO");
    }

    auto resolveCS = GraphicsHelper::CompileShader("Shaders/NaiveTsr_Resolve.hlsl", "main", "cs_6_6");
    if (!resolveCS.empty())
    {
        computeDesc.CS = { resolveCS.data(), resolveCS.size() };
        CHECK_HR(device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NaiveTsrResolvePSO)),
            "Failed to create NaiveTsr Resolve PSO");
    }

    // Motion vector generation PSO
    auto motionVecCS = GraphicsHelper::CompileShader("Shaders/MotionVectors.hlsl", "main", "cs_6_6");
    if (!motionVecCS.empty())
    {
        computeDesc.CS = { motionVecCS.data(), motionVecCS.size() };
        CHECK_HR(device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_MotionVectorsPSO)),
            "Failed to create Motion Vectors PSO");
    }

    std::cout << "TAA pipelines created" << std::endl;
}

void TAA::GenerateMotionVectors(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                                 D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, GBuffer& gbuffer,
                                 GPUTexture& motionVectorsTex, uint32_t internalWidth, uint32_t internalHeight)
{
    if (!m_MotionVectorsPSO) return;

    // Transition depth to SRV, motion vectors to UAV
    GraphicsHelper::TransitionResource(cmdList, gbuffer.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, motionVectorsTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    BindlessIndices indices = {};
    indices.OutputIdx0 = motionVectorsTex.uavIndex;
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);

    cmdList->SetPipelineState(m_MotionVectorsPSO.Get());
    cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(motionVectorsTex.resource.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void TAA::Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                   D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, const FrameConstants& frame,
                   const GPUTexture& inputColor, GBuffer& gbuffer, GPUTexture& motionVectorsTex)
{
    if (!m_NaiveTsrReprojectPSO || !m_NaiveTsrResolvePSO) return;

    const uint32_t outputW = frame.outputWidth;
    const uint32_t outputH = frame.outputHeight;

    int currentHistory = m_TaaHistoryIndex;
    int previousHistory = 1 - currentHistory;

    // ---- Pass 1: Reproject History ----
    {
        // Transition inputs to SRV
        GraphicsHelper::TransitionResource(cmdList, m_TaaHistoryTex[previousHistory], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, motionVectorsTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, gbuffer.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Transition outputs to UAV
        GraphicsHelper::TransitionResource(cmdList, m_TaaReprojectedHistoryTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_TaaClosestVelocityTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        cmdList->SetComputeRootSignature(rootSignature);
        cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
        cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
        cmdList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

        BindlessIndices indices = {};
        indices.InputIdx0 = m_TaaHistoryTex[previousHistory].srvIndex;
        indices.InputIdx1 = motionVectorsTex.srvIndex;
        indices.InputIdx2 = gbuffer.depth.srvIndex;
        indices.OutputIdx0 = m_TaaReprojectedHistoryTex.uavIndex;
        indices.OutputIdx1 = m_TaaClosestVelocityTex.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);

        cmdList->SetPipelineState(m_NaiveTsrReprojectPSO.Get());
        {
            MICROPROFILE_SCOPEGPUI("TAA_Reproject", MP_YELLOW);
            cmdList->Dispatch((outputW + 7) / 8, (outputH + 7) / 8, 1);
        }

        // UAV barrier
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaReprojectedHistoryTex.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaClosestVelocityTex.resource.Get()),
        };
        cmdList->ResourceBarrier(2, barriers);
    }

    // ---- Pass 2: TAA Resolve ----
    {
        // Transition inputs to SRV
        GraphicsHelper::TransitionResource(cmdList, const_cast<GPUTexture&>(inputColor), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_TaaReprojectedHistoryTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_TaaClosestVelocityTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Transition outputs to UAV
        GraphicsHelper::TransitionResource(cmdList, m_TaaHistoryTex[currentHistory], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_TaaOutputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BindlessIndices indices = {};
        indices.InputIdx0 = inputColor.srvIndex;
        indices.InputIdx1 = m_TaaReprojectedHistoryTex.srvIndex;
        indices.InputIdx2 = m_TaaClosestVelocityTex.srvIndex;
        indices.OutputIdx0 = m_TaaHistoryTex[currentHistory].uavIndex;
        indices.OutputIdx1 = m_TaaOutputTex.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);

        cmdList->SetPipelineState(m_NaiveTsrResolvePSO.Get());
        {
            MICROPROFILE_SCOPEGPUI("TAA_Resolve", MP_YELLOW);
            cmdList->Dispatch((outputW + 7) / 8, (outputH + 7) / 8, 1);
        }

        // UAV barrier
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaHistoryTex[currentHistory].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaOutputTex.resource.Get()),
        };
        cmdList->ResourceBarrier(2, barriers);
    }

    // Swap history index for next frame
    m_TaaHistoryIndex = previousHistory;
}
