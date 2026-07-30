#include "pch.h"

#include "RestirGI.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

void RestirGI::CreateResources(uint32_t internalWidth, uint32_t internalHeight)
{
    // ------- SHaRC buffers (~160 MB total) -------
    CreateStructuredBuffer(m_SharcHashEntriesBuf,  8,  SHARC_HASH_ENTRIES_NUM, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_SharcHashEntries");
    CreateStructuredBuffer(m_SharcAccumulationBuf, 16, SHARC_HASH_ENTRIES_NUM, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_SharcAccumulation");
    CreateStructuredBuffer(m_SharcResolvedBuf,     16, SHARC_HASH_ENTRIES_NUM, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_SharcResolved");
    m_SharcIndices.HashEntriesBufIdx  = (UINT)m_SharcHashEntriesBuf.uavIndex;
    m_SharcIndices.AccumulationBufIdx = (UINT)m_SharcAccumulationBuf.uavIndex;
    m_SharcIndices.ResolvedBufIdx     = (UINT)m_SharcResolvedBuf.uavIndex;

    // ------- Split Diffuse / Specular ReSTIR buffers -------
    for (int i = 0; i < 2; ++i) {
        CreateStructuredBuffer(m_DiffuseReservoirBuffer[i], sizeof(Reservoir), internalWidth * internalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, i == 0 ? "SB_DiffuseReservoir0" : "SB_DiffuseReservoir1");
        CreateStructuredBuffer(m_SpecularReservoirBuffer[i], sizeof(Reservoir), internalWidth * internalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, i == 0 ? "SB_SpecularReservoir0" : "SB_SpecularReservoir1");
    }
    CreateStructuredBuffer(m_DiffuseReservoirIntermediate, sizeof(Reservoir), internalWidth * internalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_DiffuseReservoirIntermediate");
    CreateStructuredBuffer(m_SpecularReservoirIntermediate, sizeof(Reservoir), internalWidth * internalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_SpecularReservoirIntermediate");
    CreateStructuredBuffer(m_DiffuseCandidateBuffer, sizeof(DiffuseCandidate), internalWidth * internalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_DiffuseCandidateBuffer");

    // GI resolved intermediates (raw float4: NRD-normalized radiance + hitT)
    CreateTexture(m_GIDiffuseIntermediate, internalWidth, internalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_GIDiffuseIntermediate");
    CreateTexture(m_GISpecularIntermediate, internalWidth, internalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_GISpecularIntermediate");
}

void RestirGI::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
    computeDesc.pRootSignature = rootSignature;

    // ------- SHaRC PSOs -------
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/SHaRC_Update.hlsl", "main", "cs_6_6",
            {{L"SHARC_UPDATE", L"1"}, {L"SHARC_PROPAGATION_DEPTH", L"4"}, {L"SHARC_UPDATE_DOWNSCALE", L"5"}});
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SharcUpdatePSO));
        }
    }
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/SHaRC_Resolve.hlsl", "main", "cs_6_6", {});
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SharcResolvePSO));
        }
    }
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/SHaRC_Debug.hlsl", "main", "cs_6_6", {});
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SharcDebugPSO));
        }
    }

    auto giResolveCS = GraphicsHelper::CompileShader("Shaders/RestirGI_ResolveIntermediates.hlsl", "main", "cs_6_6");
    if (!giResolveCS.empty()) {
        computeDesc.CS = { giResolveCS.data(), giResolveCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_GIResolveIntermediatesPSO));
    }

    // ------- Split Diffuse / Specular PSO creation -------
    auto diffuseTemporalCS  = GraphicsHelper::CompileShader("Shaders/RestirGI_Diffuse_Temporal.hlsl",  "main", "cs_6_6");
    auto specularTemporalCS = GraphicsHelper::CompileShader("Shaders/RestirGI_Specular_Temporal.hlsl", "main", "cs_6_6");
    auto diffuseSpatialCS   = GraphicsHelper::CompileShader("Shaders/RestirGI_Diffuse_Spatial.hlsl",   "main", "cs_6_6");
    auto specularSpatialCS  = GraphicsHelper::CompileShader("Shaders/RestirGI_Specular_Spatial.hlsl",  "main", "cs_6_6");

    if (!diffuseTemporalCS.empty()) {
        computeDesc.CS = { diffuseTemporalCS.data(), diffuseTemporalCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_DiffuseTemporalPSO));
    }
    if (!specularTemporalCS.empty()) {
        computeDesc.CS = { specularTemporalCS.data(), specularTemporalCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SpecularTemporalPSO));
    }
    if (!diffuseSpatialCS.empty()) {
        computeDesc.CS = { diffuseSpatialCS.data(), diffuseSpatialCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_DiffuseSpatialPSO));
    }
    if (!specularSpatialCS.empty()) {
        computeDesc.CS = { specularSpatialCS.data(), specularSpatialCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SpecularSpatialPSO));
    }
}

bool RestirGI::Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                        Model* model, const FrameConstants& frame,
                        D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, D3D12_GPU_VIRTUAL_ADDRESS tlasGPUAddress,
                        D3D12_GPU_VIRTUAL_ADDRESS lightsBufferAddress, D3D12_GPU_VIRTUAL_ADDRESS lightLUTBufferAddress,
                        const GBuffer& gbuffer, GPUTexture& fullScreenDebugTex,
                        GPUTexture& finalDiffuseTex, GPUTexture& finalSpecularTex,
                        ID3D12PipelineState* nrdStoreShadingOutputPSO,
                        int currentReservoir, int previousReservoir,
                        uint32_t internalWidth, uint32_t internalHeight)
{
    const bool useCustomRestirHeatmap = frame.restirReservoirDebugMode >= RESTIR_RESERVOIR_DEBUG_SOURCE_PDF;

    // Transition G-Buffer targets to SRV state for compute
    GraphicsHelper::TransitionResource(cmdList, const_cast<GBuffer&>(gbuffer).albedo, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, const_cast<GBuffer&>(gbuffer).normal, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, const_cast<GBuffer&>(gbuffer).material, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, const_cast<GBuffer&>(gbuffer).depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // Split diffuse/specular buffers
    GraphicsHelper::TransitionResource(cmdList, m_DiffuseReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DiffuseReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_SpecularReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_SpecularReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DiffuseReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_SpecularReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DiffuseCandidateBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (useCustomRestirHeatmap)
    {
        GraphicsHelper::TransitionResource(cmdList, fullScreenDebugTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);

    // Bind common resources
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    cmdList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    cmdList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0)); // Bindless
    cmdList->SetComputeRootShaderResourceView(4, tlasGPUAddress);
    cmdList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    cmdList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    cmdList->SetComputeRootShaderResourceView(10, lightsBufferAddress); // Lights Buffer
    cmdList->SetComputeRootShaderResourceView(11, lightLUTBufferAddress); // Light LUT Buffer

    BindlessIndices indices = {};

    // -----------------------------------------------------------------------
    // SHaRC (Spatial Hash Radiance Cache) Pipeline
    // -----------------------------------------------------------------------

    // Bind SHaRC indices; slot 13 (b2) is read by SHaRC_Update, SHaRC_Resolve,
    // and RestirGI_Raster_Temporal (query pass) — set once, persists for all three.
    cmdList->SetComputeRoot32BitConstants(13, sizeof(SharcBindlessIndices) / 4, &m_SharcIndices, 0);

    // --- Pass 1: SHaRC Update — trace secondary rays, deposit samples into hash table ---
    // Downscale by 5 (matching RTXGI default): each thread updates one rotating
    // full-resolution pixel inside a 5x5 tile, giving 25x fewer deposits per frame
    // while maintaining whole-screen coverage over time.
    static constexpr UINT SHARC_UPDATE_DOWNSCALE = 5;
    const UINT sharcUpdateW = (internalWidth  + SHARC_UPDATE_DOWNSCALE - 1) / SHARC_UPDATE_DOWNSCALE;
    const UINT sharcUpdateH = (internalHeight + SHARC_UPDATE_DOWNSCALE - 1) / SHARC_UPDATE_DOWNSCALE;
    cmdList->SetPipelineState(m_SharcUpdatePSO.Get());
    {
        MICROPROFILE_SCOPEGPUI("SHaRC_Update", MP_CYAN);
        cmdList->Dispatch((sharcUpdateW + 7) / 8, (sharcUpdateH + 7) / 8, 1);
    }

    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcHashEntriesBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcAccumulationBuf.resource.Get()),
        };
        cmdList->ResourceBarrier(2, barriers);
    }

    // --- Pass 2: SHaRC Resolve — EMA blend accumulation→resolved, clears accumulation ---
    cmdList->SetPipelineState(m_SharcResolvePSO.Get());
    {
        MICROPROFILE_SCOPEGPUI("SHaRC_Resolve", MP_CYAN);
        cmdList->Dispatch((SHARC_HASH_ENTRIES_NUM + 255) / 256, 1, 1);
    }

    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcHashEntriesBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcResolvedBuf.resource.Get()),
        };
        cmdList->ResourceBarrier(2, barriers);
    }

    // --- SHaRC Debug Visualization ---
    // When sharcDebug != 0, skip all ReSTIR + NRD and render the SHaRC debug
    // overlay directly. The debug color is written to fullScreenDebugTex
    // and displayed by FullScreenDebug.hlsl (replacing Lighting.hlsl).
    if (frame.sharcDebug != 0 && m_SharcDebugPSO)
    {
        GraphicsHelper::TransitionResource(cmdList, fullScreenDebugTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BindlessIndices debugIndices = {};
        debugIndices.OutputIdx0 = fullScreenDebugTex.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &debugIndices, 0);
        cmdList->SetPipelineState(m_SharcDebugPSO.Get());
        {
            MICROPROFILE_SCOPEGPUI("SHaRC_Debug", MP_CYAN);
            cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
        }

        D3D12_RESOURCE_BARRIER debugBarrier = CD3DX12_RESOURCE_BARRIER::UAV(fullScreenDebugTex.resource.Get());
        cmdList->ResourceBarrier(1, &debugBarrier);

        GraphicsHelper::TransitionResource(cmdList, fullScreenDebugTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        return true; // SHaRC debug overlay rendered instead of normal GI passes
    }

    // -----------------------------------------------------------------------
    // Split Diffuse / Specular ReSTIR passes
    // (1) RTDGI Temporal → (2) RTR Temporal → (3) Diffuse Spatial →
    // (4) Specular Spatial → (5) Split Resolve
    // -----------------------------------------------------------------------

    // --- Pass 1: Diffuse Temporal (RTDGI) ---
    // InputIdx0 = prev diffuse reservoirs, OutputIdx0 = curr diffuse reservoirs,
    // OutputIdx1 = diffuse candidate buffer, OutputIdx2 = debug heatmap
    cmdList->SetPipelineState(m_DiffuseTemporalPSO.Get());
    indices.InputIdx0  = m_DiffuseReservoirBuffer[previousReservoir].srvIndex;
    indices.OutputIdx0 = m_DiffuseReservoirBuffer[currentReservoir].uavIndex;
    indices.OutputIdx1 = m_DiffuseCandidateBuffer.uavIndex;
    indices.OutputIdx2 = useCustomRestirHeatmap ? fullScreenDebugTex.uavIndex : UINT(-1);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    {
        MICROPROFILE_SCOPEGPUI("GI_Diffuse_Temporal", MP_PURPLE);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_DiffuseReservoirBuffer[currentReservoir].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_DiffuseCandidateBuffer.resource.Get()),
        };
        cmdList->ResourceBarrier(2, barriers);
    }

    // --- Pass 2: Specular Temporal (RTR) ---
    // InputIdx0 = prev specular reservoirs, InputIdx1 = diffuse candidate buffer (SRV),
    // OutputIdx0 = curr specular reservoirs, OutputIdx1 = debug heatmap
    cmdList->SetPipelineState(m_SpecularTemporalPSO.Get());
    indices.InputIdx0  = m_SpecularReservoirBuffer[previousReservoir].srvIndex;
    indices.InputIdx1  = m_DiffuseCandidateBuffer.srvIndex;
    indices.OutputIdx0 = m_SpecularReservoirBuffer[currentReservoir].uavIndex;
    indices.OutputIdx1 = useCustomRestirHeatmap ? fullScreenDebugTex.uavIndex : UINT(-1);
    indices.OutputIdx2 = UINT(-1);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    {
        MICROPROFILE_SCOPEGPUI("GI_Specular_Temporal", MP_PURPLE);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_SpecularReservoirBuffer[currentReservoir].resource.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }

    // --- Pass 3: Diffuse Spatial ---
    // InputIdx0 = curr diffuse reservoirs, OutputIdx0 = diffuse intermediate
    cmdList->SetPipelineState(m_DiffuseSpatialPSO.Get());
    indices.InputIdx0  = m_DiffuseReservoirBuffer[currentReservoir].srvIndex;
    indices.InputIdx1  = UINT(-1);
    indices.OutputIdx0 = m_DiffuseReservoirIntermediate.uavIndex;
    indices.OutputIdx1 = UINT(-1);
    indices.OutputIdx2 = UINT(-1);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    {
        MICROPROFILE_SCOPEGPUI("GI_Diffuse_Spatial", MP_PURPLE);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_DiffuseReservoirIntermediate.resource.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }

    // --- Pass 4: Specular Spatial ---
    // InputIdx0 = curr specular reservoirs, OutputIdx0 = specular intermediate
    cmdList->SetPipelineState(m_SpecularSpatialPSO.Get());
    indices.InputIdx0  = m_SpecularReservoirBuffer[currentReservoir].srvIndex;
    indices.OutputIdx0 = m_SpecularReservoirIntermediate.uavIndex;
    indices.OutputIdx1 = UINT(-1);
    indices.OutputIdx2 = UINT(-1);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    {
        MICROPROFILE_SCOPEGPUI("GI_Specular_Spatial", MP_PURPLE);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_SpecularReservoirIntermediate.resource.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }

    // --- Pass 4b: GI Resolve Intermediates ---
    // Converts GI reservoir StructuredBuffers → raw float4 intermediates (BRDF eval + NRD normalize).
    // Always dispatched when GI is active; SSO always needs the intermediates to bridge to Final*.
    if (m_GIResolveIntermediatesPSO)
    {
        GraphicsHelper::TransitionResource(cmdList, m_DiffuseReservoirIntermediate,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_SpecularReservoirIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_GIDiffuseIntermediate,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_GISpecularIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        indices = {};
        indices.InputIdx0  = m_DiffuseReservoirIntermediate.srvIndex;
        indices.InputIdx1  = m_SpecularReservoirIntermediate.srvIndex;
        indices.OutputIdx0 = m_GIDiffuseIntermediate.uavIndex;
        indices.OutputIdx1 = m_GISpecularIntermediate.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        cmdList->SetPipelineState(m_GIResolveIntermediatesPSO.Get());
        {
            MICROPROFILE_SCOPEGPUI("GI_ResolveIntermediates", MP_BLUE);
            cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
        }

        D3D12_RESOURCE_BARRIER giResolveBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_GIDiffuseIntermediate.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_GISpecularIntermediate.resource.Get()),
        };
        cmdList->ResourceBarrier(_countof(giResolveBarriers), giResolveBarriers);
    }

    // --- Pass 4c: StoreShadingOutput Call 2 (GI contribution) ---
    // Reads GI intermediates and bridges into finalDiffuseTex / finalSpecularTex.
    // Always dispatched when GI is active (regardless of NRD state).
    // isFirstPass = 0 if DI ran this frame (additive blend), 1 if DI was off (overwrite).
    if (nrdStoreShadingOutputPSO && m_GIResolveIntermediatesPSO)
    {
        GraphicsHelper::TransitionResource(cmdList, m_GIDiffuseIntermediate,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_GISpecularIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, finalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, finalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        indices = {};
        indices.InputIdx0  = m_GIDiffuseIntermediate.srvIndex;
        indices.InputIdx1  = m_GISpecularIntermediate.srvIndex;
        indices.OutputIdx0 = finalDiffuseTex.uavIndex;
        indices.OutputIdx1 = finalSpecularTex.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        // isFirstPass=0 if DI ran this frame (additive blend), 1 if DI was off (overwrite)
        const UINT isFirstPass = (frame.enableRestirDI != 0u) ? 0u : 1u;
        cmdList->SetComputeRoot32BitConstants(13, 1, &isFirstPass, 0);
        cmdList->SetPipelineState(nrdStoreShadingOutputPSO);
        {
            MICROPROFILE_SCOPEGPUI("GI_StoreOutput", MP_BLUE);
            cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
        }

        D3D12_RESOURCE_BARRIER ssoBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(finalDiffuseTex.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(finalSpecularTex.resource.Get()),
        };
        cmdList->ResourceBarrier(_countof(ssoBarriers), ssoBarriers);
    }

    // FullScreenDebugTex UAV → SRV for raster debug (GI heatmap + DI debug).
    // GI field debug modes 1-4 are PT-only.
    // SHaRC debug was already handled in the early-return above.
    const bool rasterDebugActive = useCustomRestirHeatmap
        || frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF;
    if (rasterDebugActive)
    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(fullScreenDebugTex.resource.Get());
        cmdList->ResourceBarrier(1, &b);
        GraphicsHelper::TransitionResource(cmdList, fullScreenDebugTex,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }

    return false; // Normal GI passes ran (not the SHaRC debug overlay)
}
