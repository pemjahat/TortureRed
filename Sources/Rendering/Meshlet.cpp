#include "pch.h"

#include "Meshlet.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

void MeshletPass::CreateResources(uint32_t internalWidth, uint32_t internalHeight)
{
    static constexpr UINT MAX_VISIBLE_MESHLETS = 1 << 20; // 1M meshlet candidates

    // Visible meshlets list (RW)
    if (!CreateStructuredBuffer(m_VisibleMeshlets, sizeof(MeshletCandidate), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_VisibleMeshlets"))
    {
        std::cerr << "[Meshlet] Failed to create VisibleMeshlets buffer" << std::endl;
        return;
    }

    // Counter buffer (UAV)
    if (!CreateBuffer(m_VisibleMeshletsCounter, sizeof(uint32_t),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, true, "SB_VisibleMeshletsCounter"))
    {
        std::cerr << "[Meshlet] Failed to create VisibleMeshletsCounter buffer" << std::endl;
        return;
    }

    // DEBUG: per-visible-meshlet vertex/triangle counts — MeshletCandidateDebug[MAX_VISIBLE_MESHLETS]
    if (!CreateStructuredBuffer(m_VisibleMeshletsDebug, sizeof(MeshletCandidateDebug), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_VisibleMeshletsDebug"))
    {
        std::cerr << "[Meshlet] Failed to create VisibleMeshletsDebug" << std::endl;
        return;
    }

    // Indirect dispatch args for cull pass
    if (!CreateBuffer(m_CullDispatchArgs, sizeof(D3D12_DISPATCH_ARGUMENTS),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, false, true, "SB_CullDispatchArgs"))
    {
        std::cerr << "[Meshlet] Failed to create CullDispatchArgs buffer" << std::endl;
        return;
    }

    // Cull constants buffer (total meshlets)
    if (!CreateBuffer(m_CullConstantsBuffer, 256, D3D12_HEAP_TYPE_UPLOAD,
                      D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_CullConstants"))
    {
        std::cerr << "[Meshlet] Failed to create CullConstants buffer" << std::endl;
        return;
    }

    // ---- Binning resources (4-pass GPU sort) ----
    // Per-bin meshlet counts (NUM_RASTER_BINS = 2)
    if (!CreateStructuredBuffer(m_MeshletCounts, sizeof(uint32_t), NUM_RASTER_BINS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_MeshletCounts"))
    {
        std::cerr << "[Meshlet] Failed to create MeshletCounts buffer" << std::endl;
        return;
    }

    // Per-bin offset+count packed as uint4: (count, 1, 1, offset)
    // Read by the mesh shader as SRV to look up binOffset — never used as INDIRECT_ARGUMENT.
    if (!CreateStructuredBuffer(m_MeshletOffsetAndCounts, sizeof(uint32_t) * 4, NUM_RASTER_BINS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_MeshletOffsetAndCounts"))
    {
        std::cerr << "[Meshlet] Failed to create MeshletOffsetAndCounts buffer" << std::endl;
        return;
    }

    // Separate indirect dispatch args for DispatchMesh — uint3(count,1,1) per bin.
    // Kept exclusively as INDIRECT_ARGUMENT; never read as SRV by shaders.
    // Allocated as NUM_RASTER_BINS * 3 uints (3 uints = D3D12_DISPATCH_MESH_ARGUMENTS per bin).
    if (!CreateStructuredBuffer(m_DispatchMeshArgs, sizeof(uint32_t), NUM_RASTER_BINS * 3,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_DispatchMeshArgs"))
    {
        std::cerr << "[Meshlet] Failed to create DispatchMeshArgs buffer" << std::endl;
        return;
    }

    // Sorted indirection list: BinnedMeshlets[i] = index into VisibleMeshlets[]
    if (!CreateStructuredBuffer(m_BinnedMeshlets, sizeof(uint32_t), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BinnedMeshlets"))
    {
        std::cerr << "[Meshlet] Failed to create BinnedMeshlets buffer" << std::endl;
        return;
    }

    // Global meshlet counter scratch for prefix-sum
    if (!CreateStructuredBuffer(m_GlobalMeshletCounter, sizeof(uint32_t), 1,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_GlobalMeshletCounter"))
    {
        std::cerr << "[Meshlet] Failed to create GlobalMeshletCounter buffer" << std::endl;
        return;
    }

    // Indirect dispatch args for Classify/Write passes (built by PrepareArgsCS)
    // Created as a structured buffer so PrepareArgsCS can write it as RWStructuredBuffer<DispatchArgs>
    // and ExecuteIndirect can read it as INDIRECT_ARGUMENT.
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
}

void MeshletPass::CreatePipelines(ID3D12Device* device, ID3D12Device2* device2, ID3D12RootSignature* mainRootSignature, bool meshShaderSupported)
{
    m_MeshShaderSupported = meshShaderSupported;

    // --- Meshlet Cull Root Signature ---
    // Binds all SRV/UAV resources by GPU virtual address (root SRV/UAV) to avoid
    // descriptor table layout dependencies.
    //
    // Layout:
    //   [0] CBV  b0        — FrameConstants
    //   [1] CBV  b1        — CullConstants
    //   [2] SRV  t0 space3 — GlobalMeshletBounds
    //   [3] SRV  t1 space3 — GlobalMeshData
    //   [4] SRV  t2 space3 — GlobalInstanceData
    //   [5] UAV  u0        — VisibleMeshlets
    //   [6] UAV  u1        — VisibleMeshletsCounter
    //   [7] UAV  u2        — VisibleMeshletsDebug
    //   [8] SRV  t3 space3 — GlobalMeshlets
    {
        CD3DX12_ROOT_PARAMETER rootParams[9];
        rootParams[0].InitAsConstantBufferView(0);
        rootParams[1].InitAsConstantBufferView(1);
        rootParams[2].InitAsShaderResourceView(0, 3);
        rootParams[3].InitAsShaderResourceView(1, 3);
        rootParams[4].InitAsShaderResourceView(2, 3);
        rootParams[5].InitAsUnorderedAccessView(0);
        rootParams[6].InitAsUnorderedAccessView(1);
        rootParams[7].InitAsUnorderedAccessView(2); // u2: debug counter (uint2 — x=vertices, y=triangles)
        rootParams[8].InitAsShaderResourceView(3, 3); // t3 space3: GlobalMeshlets

        CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
        rsDesc.Init(9, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
        HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(hr))
        {
            if (error) std::cerr << "[Meshlet] Root signature error: " << (char*)error->GetBufferPointer() << std::endl;
            return;
        }
        CHECK_HR(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                              IID_PPV_ARGS(&m_MeshletRootSignature)), "[Meshlet] CreateRootSignature failed");
    }

    // --- Meshlet Cull PSO (CS) ---
    {
        auto cullCS = GraphicsHelper::CompileShader("Shaders/MeshletCull.hlsl", "CSMain", "cs_6_6");
        if (!cullCS.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = m_MeshletRootSignature.Get();
            desc.CS = { cullCS.data(), cullCS.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_MeshletCullPSO)),
                     "[Meshlet] CreateComputePipelineState (cull) failed");
        }
    }

    // --- Binning PSOs (CS) — use MAIN root signature (bindless heap) ---
    // PrepareArgsCS, ClassifyMeshletsCS, AllocateBinRangesCS, WriteBinsCS
    // All use the same root signature as the main pipeline (space0 bindless + root SRVs/UAVs via b1 constants).
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
    // Outputs: SV_TARGET0 = R16G16B16A16_FLOAT (HDR color), SV_TARGET1 = R32_UINT (visibility buffer)
    // Depth: D32_FLOAT reverse-Z
    // Uses D3D12 Pipeline State Stream (PSS) API — required for Mesh Shader PSOs.
    if (m_MeshShaderSupported)
    {
        // Helper lambda to build a mesh shader PSO via pipeline state stream
        auto buildMeshPSO = [&](
            const std::vector<char>& ms,
            const std::vector<char>& ps,
            D3D12_CULL_MODE cullMode,
            Microsoft::WRL::ComPtr<ID3D12PipelineState>& outPSO,
            const char* label)
        {
            if (ms.empty() || ps.empty()) return;

            // Build pipeline state stream manually
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
            ds.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
            stream.DepthStencilState = ds;

            stream.DSVFormat = DXGI_FORMAT_D32_FLOAT;

            D3D12_RT_FORMAT_ARRAY rtFormats = {};
            rtFormats.NumRenderTargets = 4;
            rtFormats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;      // SV_Target0: albedo
            rtFormats.RTFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;  // SV_Target1: packed normal
            rtFormats.RTFormats[2] = DXGI_FORMAT_R8G8B8A8_UNORM;      // SV_Target2: roughness|metallic
            rtFormats.RTFormats[3] = DXGI_FORMAT_R32_UINT;            // SV_Target3: visibility token (debug)
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

            // Normal render PSO
            {
                std::vector<std::pair<std::wstring,std::wstring>> defs = { {L"ALPHA_MASK", alphaMaskVal} };
                auto ms = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "MSMain", "ms_6_8", defs);
                auto ps = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "PSMain", "ps_6_8", defs);
                buildMeshPSO(ms, ps, cull, m_MeshletRasterPSO[i],
                             "[Meshlet] CreatePipelineState (mesh shader raster) failed");
            }

            // Debug PSO — same but with ENABLE_DEBUG_DATA=1
            {
                std::vector<std::pair<std::wstring,std::wstring>> defs = {
                    {L"ALPHA_MASK", alphaMaskVal},
                    {L"ENABLE_DEBUG_DATA", L"1"}
                };
                auto ms = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "MSMain", "ms_6_8", defs);
                auto ps = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "PSMain", "ps_6_8", defs);
                buildMeshPSO(ms, ps, cull, m_MeshletRasterDebugPSO[i],
                             "[Meshlet] CreatePipelineState (mesh shader raster debug) failed");
            }
        }

        // Direct CPU-driven debug PSO — MeshletRasterizeDebugMS.hlsl
        // No alpha-mask permutation needed: renders all meshlets as opaque for debugging.
        {
            std::vector<std::pair<std::wstring,std::wstring>> defs = { {L"ALPHA_MASK", L"0"} };
            auto ms = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeDebugMS.hlsl", "MSMain", "ms_6_8", defs);
            auto ps = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeDebugMS.hlsl", "PSMain", "ps_6_8", defs);
            buildMeshPSO(ms, ps, D3D12_CULL_MODE_NONE, m_MeshletRasterDirectPSO,
                         "[Meshlet] CreatePipelineState (mesh shader raster direct debug) failed");
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

    std::cout << "[Meshlet] Pipelines created" << std::endl;
}

void MeshletPass::Cull(ID3D12GraphicsCommandList* cmdList, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model)
{
    if (!m_MeshletCullPSO || !model->IsMeshletReady())
        return;

    size_t totalMeshlets = model->GetTotalMeshletCount();
    if (totalMeshlets == 0) return;

    GPU_MARKER(cmdList, L"Meshlet Cull");

    // Update cull constants
    {
        struct { uint totalMeshlets; uint instanceCount; uint pad[2]; } cullConsts;
        cullConsts.totalMeshlets = static_cast<uint>(totalMeshlets);
        cullConsts.instanceCount = static_cast<uint>(model->GetInstanceCount());
        memcpy(m_CullConstantsBuffer.cpuPtr, &cullConsts, sizeof(cullConsts));
    }

    cmdList->SetComputeRootSignature(m_MeshletRootSignature.Get());
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());

    // Bind CBVs
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootConstantBufferView(1, m_CullConstantsBuffer.gpuAddress);

    // Bind SRVs by GPU virtual address — no descriptor table / heap layout dependency
    cmdList->SetComputeRootShaderResourceView(2, model->GetGlobalMeshletBoundsBufferAddress());
    cmdList->SetComputeRootShaderResourceView(3, model->GetMeshDataBufferAddress());
    cmdList->SetComputeRootShaderResourceView(4, model->GetInstanceDataBufferAddress());
    cmdList->SetComputeRootShaderResourceView(8, model->GetGlobalMeshletsBufferAddress()); // t3 space3: GlobalMeshlets

    // Transition UAV resources
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshlets, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletsCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletsDebug, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Zero counter (ClearUnorderedAccessViewUint requires shader-visible GPU handle + CPU-only heap handle)
    const UINT zeroes[4] = { 0, 0, 0, 0 };
    cmdList->ClearUnorderedAccessViewUint(
        GraphicsHelper::GetSRVGPUHandle((UINT)m_VisibleMeshletsCounter.uavIndex),
        GraphicsHelper::GetCpuUAVHandle((UINT)m_VisibleMeshletsCounter.cpuUavIndex),
        m_VisibleMeshletsCounter.resource.Get(), zeroes, 0, nullptr);

    // Bind UAVs by GPU virtual address
    cmdList->SetComputeRootUnorderedAccessView(5, m_VisibleMeshlets.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(6, m_VisibleMeshletsCounter.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(7, m_VisibleMeshletsDebug.gpuAddress);

    cmdList->SetPipelineState(m_MeshletCullPSO.Get());
    {
        MICROPROFILE_SCOPEGPUI("MeshletCull", MP_GREEN);
        UINT threadGroups = static_cast<UINT>((totalMeshlets + 63) / 64);
        cmdList->Dispatch(threadGroups, 1, 1);
    }

    // UAV barriers
    D3D12_RESOURCE_BARRIER barriers[2] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_VisibleMeshlets.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_VisibleMeshletsDebug.resource.Get()),
    };
    cmdList->ResourceBarrier(2, barriers);
}

void MeshletPass::Binning(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress)
{
    // 4-pass GPU sort: PrepareArgs → Classify → AllocateBinRanges → WriteBins
    // Input:  m_VisibleMeshlets, m_VisibleMeshletsCounter (from Cull)
    // Output: m_BinnedMeshlets (sorted indirection), m_MeshletOffsetAndCounts (per-bin offset+count)
    //         m_ClassifyDispatchArgs (indirect dispatch args for Classify/Write passes)

    if (!m_MeshletBinPrepareArgsPSO || !m_MeshletClassifyPSO ||
        !m_MeshletAllocateBinRangesPSO || !m_MeshletWriteBinsPSO)
        return;

    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());

    // Bind FrameCB at root param 0
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);

    // Binning params passed via BindlessIndices (root param 12, b1):
    //   InputIdx0  = VisibleMeshlets SRV index
    //   InputIdx1  = VisibleMeshletsCounter SRV index
    //   InputIdx2  = MeshletCounts SRV index (for AllocateBinRanges read)
    //   OutputIdx0 = MeshletCounts UAV index
    //   OutputIdx1 = MeshletOffsetAndCounts UAV index
    //   OutputIdx2 = BinnedMeshlets UAV index
    //   PathVizLineBufferIdx = GlobalMeshletCounter UAV index (scratch)
    // ClassifyDispatchArgs is bound as root UAV at param 5 (u0).
    // (The binning shader reads NumBins as a push constant via b1.InputIdx0 high bits — see MeshletBinning.hlsl)

    // --- Pass 1: PrepareArgsCS ---
    // Zeros MeshletCounts, GlobalMeshletCounter; builds ClassifyDispatchArgs from VisibleMeshletsCounter.
    {
        GPU_MARKER(cmdList, L"Meshlet Binning - PrepareArgs");
        GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletsCounter,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_MeshletCounts,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_GlobalMeshletCounter,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_ClassifyDispatchArgs,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BinningParams params = {};
        params.NumBins                    = NUM_RASTER_BINS;
        params.VisibleMeshletsCounterIdx  = (uint)m_VisibleMeshletsCounter.srvIndex;
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

    // --- Pass 2: ClassifyMeshletsCS (indirect, driven by ClassifyDispatchArgs) ---
    // For each visible meshlet, looks up material.RasterBin and increments that bin's counter.
    {
        GPU_MARKER(cmdList, L"Meshlet Binning - Classify");
        GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshlets,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_ClassifyDispatchArgs,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

        BinningParams params = {};
        params.NumBins                    = NUM_RASTER_BINS;
        params.VisibleMeshletsIdx         = (uint)m_VisibleMeshlets.srvIndex;
        params.VisibleMeshletsCounterIdx  = (uint)m_VisibleMeshletsCounter.srvIndex;
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
    // Prefix-sum on MeshletCounts → writes MeshletOffsetAndCounts (offset per bin).
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

    // --- Pass 4: WriteBinsCS (indirect, driven by ClassifyDispatchArgs) ---
    // Writes each meshlet's index into BinnedMeshlets[] at its bin's offset.
    {
        GPU_MARKER(cmdList, L"Meshlet Binning - WriteBins");
        GraphicsHelper::TransitionResource(cmdList, m_ClassifyDispatchArgs,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        GraphicsHelper::TransitionResource(cmdList, m_BinnedMeshlets,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BinningParams params = {};
        params.NumBins                        = NUM_RASTER_BINS;
        params.VisibleMeshletsIdx             = (uint)m_VisibleMeshlets.srvIndex;
        params.VisibleMeshletsCounterIdx      = (uint)m_VisibleMeshletsCounter.srvIndex;
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

    // Restore VisibleMeshletsCounter to UAV for next frame
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletsCounter,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void MeshletPass::Rasterize(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model)
{
    if (!model->IsMeshletReady() || !m_MeshShaderSupported)
        return;

    GPU_MARKER(cmdList, L"Meshlet Rasterize (Mesh Shader)");

    // Transition binning outputs to SRV for the mesh shader to read
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshlets,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_BinnedMeshlets,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // MeshletOffsetAndCounts: SRV — read by mesh shader for binOffset lookup
    GraphicsHelper::TransitionResource(cmdList, m_MeshletOffsetAndCounts,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // DispatchMeshArgs: INDIRECT_ARGUMENT only — never read by shaders
    GraphicsHelper::TransitionResource(cmdList, m_DispatchMeshArgs,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

    // Use MAIN root signature
    cmdList->SetGraphicsRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());

    // Bind per-frame constants
    cmdList->SetGraphicsRootConstantBufferView(0, frameCBAddress);

    // Bind MaterialBuffer at root param 1 (t0 space1)
    cmdList->SetGraphicsRootShaderResourceView(1, model->GetMaterialBufferAddress());

    // Bind bindless texture table at root param 3 (t0 space0)
    cmdList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    // Bind meshlet stream descriptor table at root param 14 (t0-t8 space3)
    cmdList->SetGraphicsRootDescriptorTable(14, GraphicsHelper::GetSRVGPUHandle((UINT)model->GetMeshletStreamSRVBase()));

    // Rasterize each bin with its own PSO.
    // DispatchMeshArgs[binIndex] = uint3(count, 1, 1) — used as INDIRECT_ARGUMENT.
    // MeshletOffsetAndCounts[binIndex] = uint4(count, 1, 1, offset) — read as SRV by mesh shader for binOffset.
    for (uint32_t binIndex = 0; binIndex < NUM_RASTER_BINS; ++binIndex)
    {
        // Select PSO: debug mode uses the debug PSO (writes visibility buffer)
        auto* pso = (m_MeshletDebugMode > 0 && m_MeshletRasterDebugPSO[binIndex])
                    ? m_MeshletRasterDebugPSO[binIndex].Get()
                    : m_MeshletRasterPSO[binIndex].Get();
        if (!pso) continue;

        // Bind RasterParams (root param 12, b1): BinIndex + bindless SRV indices
        RasterParams rp = {};
        rp.BinIndex              = binIndex;
        rp.VisibleMeshletsIdx    = (uint)m_VisibleMeshlets.srvIndex;
        rp.BinnedMeshletsIdx     = (uint)m_BinnedMeshlets.srvIndex;
        rp.MeshletBinDataIdx     = (uint)m_MeshletOffsetAndCounts.srvIndex;
        cmdList->SetGraphicsRoot32BitConstants(12, sizeof(RasterParams) / 4, &rp, 0);

        cmdList->SetPipelineState(pso);

        // ExecuteIndirect: one DispatchMesh per bin.
        // DispatchMeshArgs[binIndex] = uint3(count, 1, 1) — written by AllocateBinRangesCS + WriteBinsCS.
        // Stride is 3 uints (= sizeof(D3D12_DISPATCH_MESH_ARGUMENTS)) per bin.
        cmdList->ExecuteIndirect(
            m_DispatchMeshSignature.Get(),
            1,
            m_DispatchMeshArgs.resource.Get(),
            sizeof(uint32_t) * 3 * binIndex,  // offset to this bin's uint3
            nullptr, 0);
    }

    // Restore resources to UAV for next frame's binning pass
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshlets,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_BinnedMeshlets,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_MeshletOffsetAndCounts,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_DispatchMeshArgs,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

// ---------------------------------------------------------------------------
// RasterizeDebug
//
// CPU-driven debug path: iterates every meshlet across all instances and issues
// one DispatchMesh(1, 1, 1) per meshlet.  No GPU culling or binning is involved.
// Use this to verify meshlet data correctness in isolation before enabling the
// full GPU-driven pipeline.
// ---------------------------------------------------------------------------
void MeshletPass::RasterizeDebug(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model)
{
    if (!model->IsMeshletReady() || !m_MeshShaderSupported)
        return;
    if (!m_MeshletRasterDirectPSO)
        return;

    // DispatchMesh requires ID3D12GraphicsCommandList6 — query it once up front.
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList6> cmdList6;
    CHECK_HR(cmdList->QueryInterface(IID_PPV_ARGS(&cmdList6)), "Failed to get ID3D12GraphicsCommandList6");

    // Use MAIN root signature
    cmdList->SetGraphicsRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());

    // Bind per-frame constants (root param 0)
    cmdList->SetGraphicsRootConstantBufferView(0, frameCBAddress);

    // Bind MaterialBuffer (root param 1, t0 space1)
    cmdList->SetGraphicsRootShaderResourceView(1, model->GetMaterialBufferAddress());

    // Bind bindless texture table (root param 3, t0 space0)
    cmdList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    // Bind meshlet stream descriptor table (root param 14, t0-t8 space3)
    cmdList->SetGraphicsRootDescriptorTable(14, GraphicsHelper::GetSRVGPUHandle((UINT)model->GetMeshletStreamSRVBase()));

    cmdList->SetPipelineState(m_MeshletRasterDirectPSO.Get());

    // Iterate every instance, then every meshlet within that instance
    const auto& instanceData = model->GetInstanceDataArray();
    const auto& meshDataArray = model->GetMeshDataArray();

    for (uint32_t instID = 0; instID < (uint32_t)instanceData.size(); ++instID)
    {
        const InstanceData& inst = instanceData[instID];
        const MeshData& md = meshDataArray[inst.MeshDataIndex];

        for (uint32_t meshletIdx = 0; meshletIdx < md.MeshletCount; ++meshletIdx)
        {
            // Write InstanceID + MeshletIndex into root param 12 (b1) as DebugRasterParams
            DebugRasterParams dp = {};
            dp.InstanceID   = instID;
            dp.MeshletIndex = meshletIdx;
            cmdList->SetGraphicsRoot32BitConstants(12, sizeof(DebugRasterParams) / 4, &dp, 0);

            // DispatchMesh requires ID3D12GraphicsCommandList6
            cmdList6->DispatchMesh(1, 1, 1);
        }
    }
}
