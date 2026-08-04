#include "pch.h"

#include "Meshlet.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

void MeshletPass::CreateResources(uint32_t internalWidth, uint32_t internalHeight)
{
    static constexpr UINT MAX_VISIBLE_MESHLETS = 1 << 20;

    // ---- Binning resources (4-pass GPU sort) ----
    if (!CreateStructuredBuffer(m_MeshletCounts, sizeof(uint32_t), NUM_RASTER_BINS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_MeshletCounts"))
    {
        std::cerr << "[Meshlet] Failed to create MeshletCounts buffer" << std::endl;
        return;
    }

    if (!CreateStructuredBuffer(m_MeshletOffsetAndCounts, sizeof(uint32_t) * 4, NUM_RASTER_BINS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_MeshletOffsetAndCounts"))
    {
        std::cerr << "[Meshlet] Failed to create MeshletOffsetAndCounts buffer" << std::endl;
        return;
    }

    if (!CreateStructuredBuffer(m_DispatchMeshArgs, sizeof(uint32_t), NUM_RASTER_BINS * 3,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_DispatchMeshArgs"))
    {
        std::cerr << "[Meshlet] Failed to create DispatchMeshArgs buffer" << std::endl;
        return;
    }

    if (!CreateStructuredBuffer(m_BinnedMeshlets, sizeof(uint32_t), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BinnedMeshlets"))
    {
        std::cerr << "[Meshlet] Failed to create BinnedMeshlets buffer" << std::endl;
        return;
    }

    if (!CreateStructuredBuffer(m_GlobalMeshletCounter, sizeof(uint32_t), 1,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_GlobalMeshletCounter"))
    {
        std::cerr << "[Meshlet] Failed to create GlobalMeshletCounter buffer" << std::endl;
        return;
    }

    if (!CreateStructuredBuffer(m_ClassifyDispatchArgs, sizeof(D3D12_DISPATCH_ARGUMENTS), 1,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_ClassifyDispatchArgs"))
    {
        std::cerr << "[Meshlet] Failed to create ClassifyDispatchArgs buffer" << std::endl;
        return;
    }

    std::cout << "[Meshlet] Resources created (max " << MAX_VISIBLE_MESHLETS << " visible meshlets, "
              << NUM_RASTER_BINS << " raster bins)" << std::endl;

    // Visibility buffer for debug overlay (plan001)
    if (!CreateTexture(m_VisibilityBuffer, internalWidth, internalHeight,
                       DXGI_FORMAT_R32_UINT,
                       D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, 1, 1, "GBuffer_VisibilityBuffer"))
    {
        std::cerr << "[Meshlet] Failed to create visibility buffer" << std::endl;
    }

    // Culling, HZB, and debug-recording resources are now in GPUCulling.
}

void MeshletPass::RecreateVisibilityBuffer(uint32_t internalWidth, uint32_t internalHeight)
{
    if (!CreateTexture(m_VisibilityBuffer, internalWidth, internalHeight,
                       DXGI_FORMAT_R32_UINT,
                       D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, 1, 1, "GBuffer_VisibilityBuffer"))
    {
        std::cerr << "[Meshlet] Failed to recreate visibility buffer" << std::endl;
    }
    // HZB recreation is now handled by GPUCulling::RecreateHZB().
}

void MeshletPass::CreatePipelines(ID3D12Device* device, ID3D12Device2* device2,
                                   ID3D12RootSignature* mainRootSignature,
                                   bool meshShaderSupported)
{
    m_MeshShaderSupported = meshShaderSupported;

    // --- Binning PSOs (CS) — use MAIN root signature ---
    {
        auto compile = [&](const char* entry, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso, const char* label)
        {
            auto cs = GraphicsHelper::CompileShader("Shaders/MeshletBinning.hlsl", entry, "cs_6_6");
            if (!cs.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature;
                desc.CS = { cs.data(), cs.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), label);
            }
        };
        compile("PrepareArgsCS",        m_MeshletBinPrepareArgsPSO,    "[Meshlet] CreateComputePipelineState (bin prepare) failed");
        compile("ClassifyMeshletsCS",   m_MeshletClassifyPSO,          "[Meshlet] CreateComputePipelineState (bin classify) failed");
        compile("AllocateBinRangesCS",  m_MeshletAllocateBinRangesPSO, "[Meshlet] CreateComputePipelineState (bin allocate) failed");
        compile("WriteBinsCS",          m_MeshletWriteBinsPSO,         "[Meshlet] CreateComputePipelineState (bin write) failed");
    }

    // --- Mesh Shader Raster PSOs (MS+PS) — use MAIN root signature ---
    if (m_MeshShaderSupported)
    {
        auto buildMeshPSO = [&](
            const std::vector<char>& ms,
            const std::vector<char>& ps,
            D3D12_CULL_MODE cullMode,
            Microsoft::WRL::ComPtr<ID3D12PipelineState>& outPSO,
            const char* label)
        {
            if (ms.empty() || ps.empty()) return;

            struct MeshShaderPSOStream
            {
                CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE        RootSignature;
                CD3DX12_PIPELINE_STATE_STREAM_MS                    MS;
                CD3DX12_PIPELINE_STATE_STREAM_PS                    PS;
                CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER            RasterizerState;
                CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC            BlendState;
                CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL         DepthStencilState;
                CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT  DSVFormat;
                CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
                CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC           SampleDesc;
                CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY    PrimitiveTopologyType;
            } stream;

            stream.RootSignature = mainRootSignature;
            stream.MS = D3D12_SHADER_BYTECODE{ ms.data(), ms.size() };
            stream.PS = D3D12_SHADER_BYTECODE{ ps.data(), ps.size() };

            CD3DX12_RASTERIZER_DESC raster(D3D12_DEFAULT);
            raster.FrontCounterClockwise = TRUE;
            raster.CullMode = cullMode;
            stream.RasterizerState = raster;
            stream.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

            CD3DX12_DEPTH_STENCIL_DESC ds(D3D12_DEFAULT);
            ds.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            stream.DepthStencilState = ds;

            stream.DSVFormat = DXGI_FORMAT_D32_FLOAT;

            D3D12_RT_FORMAT_ARRAY rtFormats = {};
            rtFormats.NumRenderTargets = 4;
            rtFormats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            rtFormats.RTFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
            rtFormats.RTFormats[2] = DXGI_FORMAT_R8G8B8A8_UNORM;
            rtFormats.RTFormats[3] = DXGI_FORMAT_R32_UINT;
            stream.RTVFormats = rtFormats;

            DXGI_SAMPLE_DESC sampleDesc = { 1, 0 };
            stream.SampleDesc = sampleDesc;
            stream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

            D3D12_PIPELINE_STATE_STREAM_DESC streamDesc = { sizeof(stream), &stream };
            CHECK_HR(device2->CreatePipelineState(&streamDesc, IID_PPV_ARGS(&outPSO)), label);
        };

        for (uint32_t i = 0; i < NUM_RASTER_BINS; ++i)
        {
            D3D12_CULL_MODE cull = (i == 0) ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
            std::wstring alphaMaskVal = (i == 0) ? L"0" : L"1";
            {
                std::vector<std::pair<std::wstring,std::wstring>> defs = { {L"ALPHA_MASK", alphaMaskVal} };
                auto ms = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "MSMain", "ms_6_8", defs);
                auto ps = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "PSMain", "ps_6_8", defs);
                buildMeshPSO(ms, ps, cull, m_MeshletRasterPSO[i],
                             "[Meshlet] CreatePipelineState (mesh shader raster) failed");
            }
        }
    }

    // --- Indirect Dispatch command signature (for binning Classify/Write passes) ---
    {
        D3D12_INDIRECT_ARGUMENT_DESC dispatchArg = {};
        dispatchArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC sigDesc = {};
        sigDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        sigDesc.NumArgumentDescs = 1;
        sigDesc.pArgumentDescs = &dispatchArg;
        CHECK_HR(device->CreateCommandSignature(&sigDesc, nullptr,
                 IID_PPV_ARGS(&m_DispatchCommandSignatureCS)),
                 "[Meshlet] CreateCommandSignature (dispatch) failed");
    }

    // --- Indirect DispatchMesh command signature (for per-bin rasterize) ---
    if (m_MeshShaderSupported)
    {
        D3D12_INDIRECT_ARGUMENT_DESC meshArg = {};
        meshArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        D3D12_COMMAND_SIGNATURE_DESC sigDesc = {};
        sigDesc.ByteStride = sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
        sigDesc.NumArgumentDescs = 1;
        sigDesc.pArgumentDescs = &meshArg;
        CHECK_HR(device->CreateCommandSignature(&sigDesc, nullptr,
                 IID_PPV_ARGS(&m_DispatchMeshSignature)),
                 "[Meshlet] CreateCommandSignature (dispatch mesh) failed");
    }

    // --- Meshlet Debug View PSO (CS) for visibility buffer overlay ---
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/VisibilityDebugView.hlsl", "DebugRenderCS", "cs_6_6");
        if (!cs.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = mainRootSignature;
            desc.CS = { cs.data(), cs.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_MeshletDebugViewPSO)),
                     "[Meshlet] CreateComputePipelineState (debug view) failed");
        }
    }

    // HZB/two-pass-cull/debug-overlay PSOs are in GPUCulling.
    std::cout << "[Meshlet] Pipelines created" << std::endl;
}

void MeshletPass::Binning(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                           D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                           int visibleMeshletsSRVIdx, int visibleMeshletsCounterSRVIdx,
                           int visibleMeshletsCounterUAVIdx)
{
    if (!m_MeshletBinPrepareArgsPSO || !m_MeshletClassifyPSO ||
        !m_MeshletAllocateBinRangesPSO || !m_MeshletWriteBinsPSO)
        return;

    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);

    // --- Pass 1: PrepareArgsCS ---
    {
        GPU_MARKER(cmdList, L"Meshlet Binning - PrepareArgs");
        GraphicsHelper::TransitionResource(cmdList, m_MeshletCounts,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_GlobalMeshletCounter,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_ClassifyDispatchArgs,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BinningParams params = {};
        params.NumBins                    = NUM_RASTER_BINS;
        params.VisibleMeshletsCounterIdx  = (uint)visibleMeshletsCounterSRVIdx;
        params.RWMeshletCountsIdx         = (uint)m_MeshletCounts.uavIndex;
        params.RWGlobalMeshletCounterIdx  = (uint)m_GlobalMeshletCounter.uavIndex;
        params.RWDispatchArgumentsIdx     = (uint)m_ClassifyDispatchArgs.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BinningParams) / 4, &params, 0);

        cmdList->SetPipelineState(m_MeshletBinPrepareArgsPSO.Get());
        cmdList->Dispatch(1, 1, 1);

        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletCounts.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_ClassifyDispatchArgs.resource.Get()),
        };
        cmdList->ResourceBarrier(2, barriers);
    }

    // --- Pass 2: ClassifyMeshletsCS (indirect) ---
    {
        GPU_MARKER(cmdList, L"Meshlet Binning - Classify");
        GraphicsHelper::TransitionResource(cmdList, m_ClassifyDispatchArgs,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

        BinningParams params = {};
        params.NumBins                    = NUM_RASTER_BINS;
        params.VisibleMeshletsIdx         = (uint)visibleMeshletsSRVIdx;
        params.VisibleMeshletsCounterIdx  = (uint)visibleMeshletsCounterSRVIdx;
        params.RWMeshletCountsIdx         = (uint)m_MeshletCounts.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BinningParams) / 4, &params, 0);

        cmdList->SetPipelineState(m_MeshletClassifyPSO.Get());
        cmdList->ExecuteIndirect(m_DispatchCommandSignatureCS.Get(), 1,
                                 m_ClassifyDispatchArgs.resource.Get(), 0, nullptr, 0);

        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletCounts.resource.Get()),
        };
        cmdList->ResourceBarrier(1, barriers);
    }

    // --- Pass 3: AllocateBinRangesCS ---
    {
        GPU_MARKER(cmdList, L"Meshlet Binning - AllocateBinRanges");
        GraphicsHelper::TransitionResource(cmdList, m_MeshletCounts,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_MeshletOffsetAndCounts,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BinningParams params = {};
        params.NumBins                        = NUM_RASTER_BINS;
        params.MeshletCountsIdx               = (uint)m_MeshletCounts.srvIndex;
        params.RWMeshletOffsetAndCountsIdx    = (uint)m_MeshletOffsetAndCounts.uavIndex;
        params.RWGlobalMeshletCounterIdx      = (uint)m_GlobalMeshletCounter.uavIndex;
        params.RWDispatchMeshArgsIdx          = (uint)m_DispatchMeshArgs.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BinningParams) / 4, &params, 0);

        cmdList->SetPipelineState(m_MeshletAllocateBinRangesPSO.Get());
        cmdList->Dispatch((NUM_RASTER_BINS + 63) / 64, 1, 1);

        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletOffsetAndCounts.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_DispatchMeshArgs.resource.Get()),
        };
        cmdList->ResourceBarrier(2, barriers);
    }

    // --- Pass 4: WriteBinsCS (indirect) ---
    {
        GPU_MARKER(cmdList, L"Meshlet Binning - WriteBins");
        GraphicsHelper::TransitionResource(cmdList, m_ClassifyDispatchArgs,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        GraphicsHelper::TransitionResource(cmdList, m_BinnedMeshlets,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BinningParams params = {};
        params.NumBins                        = NUM_RASTER_BINS;
        params.VisibleMeshletsIdx             = (uint)visibleMeshletsSRVIdx;
        params.VisibleMeshletsCounterIdx      = (uint)visibleMeshletsCounterSRVIdx;
        params.RWMeshletOffsetAndCountsIdx    = (uint)m_MeshletOffsetAndCounts.uavIndex;
        params.RWBinnedMeshletsIdx            = (uint)m_BinnedMeshlets.uavIndex;
        params.RWDispatchMeshArgsIdx          = (uint)m_DispatchMeshArgs.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BinningParams) / 4, &params, 0);

        cmdList->SetPipelineState(m_MeshletWriteBinsPSO.Get());
        cmdList->ExecuteIndirect(m_DispatchCommandSignatureCS.Get(), 1,
                                 m_ClassifyDispatchArgs.resource.Get(), 0, nullptr, 0);

        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_BinnedMeshlets.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletOffsetAndCounts.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_DispatchMeshArgs.resource.Get()),
        };
        cmdList->ResourceBarrier(3, barriers);
    }
}

void MeshletPass::Rasterize(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                             D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model,
                             int visibleMeshletsSRVIdx)
{
    if (!model->IsMeshletReady() || !m_MeshShaderSupported)
        return;

    GPU_MARKER(cmdList, L"Meshlet Rasterize (Mesh Shader)");

    GraphicsHelper::TransitionResource(cmdList, m_BinnedMeshlets,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_MeshletOffsetAndCounts,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_DispatchMeshArgs,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

    cmdList->SetGraphicsRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetGraphicsRootConstantBufferView(0, frameCBAddress);
    cmdList->SetGraphicsRootShaderResourceView(1, model->GetMaterialBufferAddress());
    cmdList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));
    cmdList->SetGraphicsRootDescriptorTable(14, GraphicsHelper::GetSRVGPUHandle((UINT)model->GetMeshletStreamSRVBase()));

    for (uint32_t binIndex = 0; binIndex < NUM_RASTER_BINS; ++binIndex)
    {
        auto* pso = m_MeshletRasterPSO[binIndex].Get();
        if (!pso) continue;

        RasterParams rp = {};
        rp.BinIndex              = binIndex;
        rp.VisibleMeshletsIdx    = (uint)visibleMeshletsSRVIdx;
        rp.BinnedMeshletsIdx     = (uint)m_BinnedMeshlets.srvIndex;
        rp.MeshletBinDataIdx     = (uint)m_MeshletOffsetAndCounts.srvIndex;
        cmdList->SetGraphicsRoot32BitConstants(12, sizeof(RasterParams) / 4, &rp, 0);

        cmdList->SetPipelineState(pso);
        cmdList->ExecuteIndirect(
            m_DispatchMeshSignature.Get(), 1,
            m_DispatchMeshArgs.resource.Get(),
            sizeof(uint32_t) * 3 * binIndex,
            nullptr, 0);
    }

    // Restore to UAV for next frame
    GraphicsHelper::TransitionResource(cmdList, m_BinnedMeshlets,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_MeshletOffsetAndCounts,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DispatchMeshArgs,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
