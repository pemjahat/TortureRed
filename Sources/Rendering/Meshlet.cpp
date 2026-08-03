#include "pch.h"

#include "Meshlet.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

#define A_CPU 1
#include "ffx_a.h"
#include "ffx_spd.h"

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

    // ----- Two-pass occlusion culling buffers -----
    if (!CreateStructuredBuffer(m_CandidateMeshlets, sizeof(MeshletCandidate), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_CandidateMeshlets"))
        std::cerr << "[Meshlet] Failed to create CandidateMeshlets buffer" << std::endl;
    if (!CreateBuffer(m_CandidateMeshletsCounter, sizeof(uint32_t),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, true, "SB_CandidateMeshletsCounter"))
        std::cerr << "[Meshlet] Failed to create CandidateMeshletsCounter" << std::endl;
    if (!CreateStructuredBuffer(m_OccludedInstances, sizeof(uint32_t), static_cast<UINT>(MAX_VISIBLE_MESHLETS / 4),
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_OccludedInstances"))
        std::cerr << "[Meshlet] Failed to create OccludedInstances buffer" << std::endl;
    if (!CreateBuffer(m_OccludedInstancesCounter, sizeof(uint32_t),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, true, "SB_OccludedInstancesCounter"))
        std::cerr << "[Meshlet] Failed to create OccludedInstancesCounter" << std::endl;
    if (!CreateStructuredBuffer(m_MeshletCullArgs, sizeof(uint32_t), 3,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_MeshletCullArgs"))
        std::cerr << "[Meshlet] Failed to create MeshletCullArgs" << std::endl;
    if (!CreateStructuredBuffer(m_InstanceCullArgs, sizeof(uint32_t), 3,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_InstanceCullArgs"))
        std::cerr << "[Meshlet] Failed to create InstanceCullArgs" << std::endl;
    // Double-buffered upload CB: Phase 1 writes to [0], Phase 2 writes to [1].
    // Prevents write-after-write hazard on the upload heap when both phases
    // record into the same command list — Phase 2's CPU memcpy would otherwise
    // overwrite Phase 1's data before the GPU executes Phase 1's dispatch.
    if (!CreateBuffer(m_TwoPassCullConstantsBuffer[0], 256, D3D12_HEAP_TYPE_UPLOAD,
                      D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_TwoPassCullConstants_0"))
        std::cerr << "[Meshlet] Failed to create TwoPassCullConstantsBuffer[0]" << std::endl;
    if (!CreateBuffer(m_TwoPassCullConstantsBuffer[1], 256, D3D12_HEAP_TYPE_UPLOAD,
                      D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_TwoPassCullConstants_1"))
        std::cerr << "[Meshlet] Failed to create TwoPassCullConstantsBuffer[1]" << std::endl;

    CreateHZBResources(internalWidth, internalHeight);
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

    CreateHZBResources(internalWidth, internalHeight);
}

// -----------------------------------------------------------------------------
// CreateHZBResources"
//
// Sizes the HZB at half the next-power-of-two of the view dimensions computes the mip count,
// and (re)creates the multi-mip HZB texture plus the small SPD support buffers.
// Called from CreateResources() and again whenever the internal resolution changes
// (RecreateVisibilityBuffer's call site), since the HZB is resolution-dependent.
// -----------------------------------------------------------------------------
void MeshletPass::CreateHZBResources(uint32_t internalWidth, uint32_t internalHeight)
{
    auto nextPow2 = [](uint32_t v) -> uint32_t
    {
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        return v + 1;
    };

    m_HZBWidth  = std::max(nextPow2(internalWidth)  >> 1u, 1u);
    m_HZBHeight = std::max(nextPow2(internalHeight) >> 1u, 1u);
    uint32_t maxDim = std::max(m_HZBWidth, m_HZBHeight);
    m_HZBMips = std::min<uint32_t>(static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(maxDim)))) + 1, SPD_MAX_MIPS);

    // R32_FLOAT: reverse-Z HZB stores the NEAREST (closest) depth per texel (min-reduce) — see HZB.hlsl header.
    if (!CreateTexture(m_HZB, m_HZBWidth, m_HZBHeight, DXGI_FORMAT_R32_FLOAT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, m_HZBMips, 1, "Tex_HZB"))
    {
        std::cerr << "[Meshlet] Failed to create HZB texture" << std::endl;
        return;
    }

    // Per-mip UAVs: CreateTexture() only creates a UAV for mip 0 (uavIndex). Allocate one bindless
    // UAV descriptor per additional mip so HZB.hlsl can address any mip via ResourceDescriptorHeap[].
    auto& ctx = GraphicsHelper::GetContext();
    ID3D12Device* device = ctx.device;
    m_HZBMipUAVIndices[0] = static_cast<int>(m_HZB.uavIndex);
    for (uint32_t mip = 1; mip < m_HZBMips; ++mip)
    {
        int uavIdx = static_cast<int>(GraphicsHelper::AllocateSRV());
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = mip;
        device->CreateUnorderedAccessView(m_HZB.resource.Get(), nullptr, &uavDesc,
                                           GraphicsHelper::GetSRVCPUHandle(static_cast<UINT>(uavIdx)));
        m_HZBMipUAVIndices[mip] = uavIdx;
    }
    for (uint32_t mip = m_HZBMips; mip < SPD_MAX_MIPS; ++mip)
        m_HZBMipUAVIndices[mip] = m_HZBMipUAVIndices[m_HZBMips - 1]; // unused slots — keep the SRV read in-bounds

    // Upload the mip-UAV-index table read by HZB.hlsl (StructuredBuffer<uint>[SPD_MAX_MIPS]).
    if (!CreateStructuredBuffer(m_HZBMipIndicesBuffer, sizeof(uint32_t), SPD_MAX_MIPS,
                                D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, "SB_HZBMipIndices"))
    {
        std::cerr << "[Meshlet] Failed to create HZBMipIndices buffer" << std::endl;
        return;
    }
    memcpy(m_HZBMipIndicesBuffer.cpuPtr, m_HZBMipUAVIndices, sizeof(uint32_t) * SPD_MAX_MIPS);

    // SPD's global atomic cross-workgroup-sync counter — one uint, self-resets to 0 at the end of
    // each SpdDownsample() dispatch (SPD v2.0+), so no per-frame clear is required after creation.
    if (!CreateStructuredBuffer(m_HZBSpdCounter, sizeof(uint32_t), 1,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_HZBSpdCounter"))
    {
        std::cerr << "[Meshlet] Failed to create HZBSpdCounter buffer" << std::endl;
        return;
    }

    std::cout << "[Meshlet] HZB resources created (" << m_HZBWidth << "x" << m_HZBHeight
              << ", " << m_HZBMips << " mips)" << std::endl;
}

void MeshletPass::CreatePipelines(ID3D12Device* device, ID3D12Device2* device2, ID3D12RootSignature* mainRootSignature, bool meshShaderSupported)
{
    m_MeshShaderSupported = meshShaderSupported;

    // --- Unified Meshlet Cull Root Signature ---
    // Superset of old frustum-only cull + two-pass occlusion cull. Must be created
    // BEFORE CreateTwoPassCullPipelines() below, since that function's PSOs reference
    // m_MeshletRootSignature.Get() as their pRootSignature.
    // All SRV/UAV at space0; static point-clamp sampler + CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
    // required by two-pass cull for bindless HZB access.
    //
    // CullTwoPass() binding (hierarchical cull — used for both occlusion and frustum-only modes):
    //   [0]  CBV b0 — FrameConstants
    //   [1]  CBV b1 — TwoPassCullConstants
    //   [2]  SRV t0 — InstanceData[]
    //   [3]  SRV t1 — InstanceBounds[]
    //   [4]  SRV t2 — MeshData[]
    //   [5]  SRV t3 — MeshletBounds[]
    //   [6]  SRV t4 — (unused; CandidateMeshlets read via UAV u0 instead, not a
    //                  separate SRV — avoids a resource-state hazard,
    //                  see MeshletTwoPassCull.hlsl)
    //   [7]  UAV u0 — CandidateMeshlets
    //   [8]  UAV u1 — CandidateMeshletsCounter
    //   [9]  UAV u2 — OccludedInstances[]
    //   [10] UAV u3 — OccludedInstancesCounter
    //   [11] UAV u4 — VisibleMeshlets[]
    //   [12] UAV u5 — VisibleMeshletsCounter
    //   [13] UAV u6 — MeshletCullArgs
    //   [14] UAV u7 — InstanceCullArgs
    {
        CD3DX12_ROOT_PARAMETER rootParams[15];
        rootParams[0].InitAsConstantBufferView(0);          // b0: FrameConstants
        rootParams[1].InitAsConstantBufferView(1);          // b1: shader-specific constants
        rootParams[2].InitAsShaderResourceView(0);          // t0
        rootParams[3].InitAsShaderResourceView(1);          // t1
        rootParams[4].InitAsShaderResourceView(2);          // t2
        rootParams[5].InitAsShaderResourceView(3);          // t3
        rootParams[6].InitAsShaderResourceView(4);          // t4
        rootParams[7].InitAsUnorderedAccessView(0);         // u0
        rootParams[8].InitAsUnorderedAccessView(1);         // u1
        rootParams[9].InitAsUnorderedAccessView(2);         // u2
        rootParams[10].InitAsUnorderedAccessView(3);        // u3
        rootParams[11].InitAsUnorderedAccessView(4);        // u4
        rootParams[12].InitAsUnorderedAccessView(5);        // u5
        rootParams[13].InitAsUnorderedAccessView(6);        // u6
        rootParams[14].InitAsUnorderedAccessView(7);        // u7

        CD3DX12_STATIC_SAMPLER_DESC pointClampSampler(0, D3D12_FILTER_MIN_MAG_MIP_POINT,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

        CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
        rsDesc.Init(15, rootParams, 1, &pointClampSampler,
                    D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);

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

    // Hierarchical cull PSOs (CullInstancesCS + CullMeshletsCS, two-phase + frustum-only) —
    // must come AFTER m_MeshletRootSignature is created above,
    // since these PSOs are built with pRootSignature = m_MeshletRootSignature.Get().
    CreateTwoPassCullPipelines(device);

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
            ds.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL; // Reverse-Z: closer = larger depth
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

    CreateHZBPipelines(device);

    std::cout << "[Meshlet] Pipelines created" << std::endl;
}

// -----------------------------------------------------------------------------
// CreateHZBPipelines
//
// HZB.hlsl accesses everything (source depth, per-mip HZB UAVs, SPD counter) via
// bindless ResourceDescriptorHeap[idx] indices carried in HZBConstants, so its
// dedicated root signature only needs a single root CBV (b0) plus the
// CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED flag — no descriptor tables or root SRV/UAVs.
// -----------------------------------------------------------------------------
void MeshletPass::CreateHZBPipelines(ID3D12Device* device)
{
    {
        CD3DX12_ROOT_PARAMETER rootParams[1];
        rootParams[0].InitAsConstantBufferView(0); // b0: HZBConstants

        CD3DX12_STATIC_SAMPLER_DESC pointClampSampler(0, D3D12_FILTER_MIN_MAG_MIP_POINT,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

        CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
        rsDesc.Init(1, rootParams, 1, &pointClampSampler,
                    D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);

        Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
        HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(hr))
        {
            if (error) std::cerr << "[Meshlet] HZB root signature error: " << (char*)error->GetBufferPointer() << std::endl;
            return;
        }
        CHECK_HR(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                              IID_PPV_ARGS(&m_HZBRootSignature)), "[Meshlet] CreateRootSignature (HZB) failed");
    }

    auto compile = [&](const char* entry, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso, const char* label)
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/HZB.hlsl", entry, "cs_6_6");
        if (!cs.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = m_HZBRootSignature.Get();
            desc.CS = { cs.data(), cs.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), label);
        }
    };
    compile("HZBInitCS",   m_HZBInitPSO,   "[Meshlet] CreateComputePipelineState (HZB init) failed");
    compile("HZBCreateCS", m_HZBCreatePSO, "[Meshlet] CreateComputePipelineState (HZB create) failed");
}

// -----------------------------------------------------------------------------
// BuildHZB"
//
// Two dispatches: HZBInitCS writes mip 0 (min-reduced 2x2 Gather() of the source
// depth buffer — reverse-Z: farthest=smallest, so min()), HZBCreateCS runs SPD to
// generate mips 1..NumMips-1 in one dispatch.
// Currently invoked once at the point in the frame where the old single-phase
// culling read depth (see docs/task004-2passcull.md "Risks and scoped-out items"
// for the still-pending two-phase Cull() rewrite that will consume this HZB).
// -----------------------------------------------------------------------------
void MeshletPass::BuildHZB(ID3D12GraphicsCommandList* cmdList, GPUTexture& depthBuffer)
{
    if (!m_HZBInitPSO || !m_HZBCreatePSO || m_HZB.resource == nullptr)
        return;

    GPU_MARKER(cmdList, L"HZB Build");

    HZBConstants hzbConstants = {};
    hzbConstants.DepthSRVIdx      = depthBuffer.srvIndex;
    hzbConstants.MipIndicesSRVIdx = static_cast<uint32_t>(m_HZBMipIndicesBuffer.srvIndex);
    hzbConstants.SpdCounterUAVIdx = static_cast<uint32_t>(m_HZBSpdCounter.uavIndex);
    hzbConstants.NumMips  = m_HZBMips;
    hzbConstants.Width    = m_HZBWidth;
    hzbConstants.Height   = m_HZBHeight;
    hzbConstants.DimensionsInvX = 1.0f / static_cast<float>(m_HZBWidth);
    hzbConstants.DimensionsInvY = 1.0f / static_cast<float>(m_HZBHeight);

    // SpdSetup() (CPU-side ffx_a.h macros) computes the dispatch dimensions for HZBCreateCS.
    varAU2(dispatchThreadGroupCountXY);
    varAU2(workGroupOffset);
    varAU2(numWorkGroupsAndMips);
    varAU4(rectInfo) = initAU4(0, 0, m_HZBWidth, m_HZBHeight);
    SpdSetup(dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips, rectInfo,
             static_cast<int>(m_HZBMips) - 1); // SPD's own mip count excludes mip 0 (already written by HZBInitCS)

    hzbConstants.NumWorkGroups    = numWorkGroupsAndMips[0];
    hzbConstants.WorkGroupOffsetX = workGroupOffset[0];
    hzbConstants.WorkGroupOffsetY = workGroupOffset[1];

    // Upload the constants for this frame via a small per-call scratch CBV.
    // Reuses the persistent upload-heap map + memcpy pattern (same as the cull constant buffers).
    if (m_HZBConstantsBuffer.resource == nullptr)
    {
        if (!CreateBuffer(m_HZBConstantsBuffer, 256, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_HZBConstants"))
        {
            std::cerr << "[Meshlet] Failed to create HZBConstants buffer" << std::endl;
            return;
        }
    }
    memcpy(m_HZBConstantsBuffer.cpuPtr, &hzbConstants, sizeof(hzbConstants));

    cmdList->SetComputeRootSignature(m_HZBRootSignature.Get());
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootConstantBufferView(0, m_HZBConstantsBuffer.gpuAddress);

    GraphicsHelper::TransitionResource(cmdList, depthBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_HZB, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_HZBSpdCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    {
        GPU_MARKER(cmdList, L"HZB Init (mip 0)");
        cmdList->SetPipelineState(m_HZBInitPSO.Get());
        UINT groupsX = (m_HZBWidth + 15) / 16;
        UINT groupsY = (m_HZBHeight + 15) / 16;
        cmdList->Dispatch(groupsX, groupsY, 1);

        D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_HZB.resource.Get());
        cmdList->ResourceBarrier(1, &uavBarrier);
    }

    if (m_HZBMips > 1)
    {
        GPU_MARKER(cmdList, L"HZB Create (SPD mips)");
        cmdList->SetPipelineState(m_HZBCreatePSO.Get());
        cmdList->Dispatch(dispatchThreadGroupCountXY[0], dispatchThreadGroupCountXY[1], 1);
    }

    // Transition the HZB to a shader-readable state. CullTwoPass()'s HZBCull() reads
    // it via bindless SRV (ResourceDescriptorHeap[HZBSRVIdx]) — both Phase 1 (reading
    // the HZB built at the end of the PREVIOUS frame) and Phase 2 (reading the HZB
    // just rebuilt above) require this. Without it, the resource stays in
    // UNORDERED_ACCESS, which is an invalid state for the SRV read.
    D3D12_RESOURCE_BARRIER hzbDoneBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_HZB.resource.Get());
    cmdList->ResourceBarrier(1, &hzbDoneBarrier);
    GraphicsHelper::TransitionResource(cmdList, m_HZB, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

// =============================================================================
// CullTwoPass — hierarchical two-stage culling (instance → meshlet), used for
// both two-phase occlusion culling and the frustum-only single-phase mode.
//
// Phase 0 (FIRST):  CullInstancesCS (direct) → BuildMeshletCullIndirectArgs →
//                   CullMeshletsCS (indirect). Input: all instances. HZB: previous frame.
// Phase 1 (SECOND): BuildInstanceCullIndirectArgs → CullInstancesCS (indirect,
//                   only OccludedInstances) → BuildMeshletCullIndirectArgs →
//                   CullMeshletsCS (indirect). HZB: freshly rebuilt this frame.
//
// occlusionEnabled=0 falls back to frustum-only (no HZB read), matching the old
// single-phase behavior byte-for-byte as a safe rollback path.
// =============================================================================
void MeshletPass::CullTwoPass(ID3D12GraphicsCommandList* cmdList, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                               Model* model, bool occlusionEnabled, int phase)
{
    if (!m_CullInstancesPSO || !m_CullMeshletsPSO || !model->IsMeshletReady())
        return;

    size_t totalInstances = model->GetInstanceCount();
    size_t totalMeshlets  = model->GetTotalMeshletCount();
    if (totalInstances == 0 || totalMeshlets == 0) return;

    const bool isFirstPhase = (phase == 0);

    // ---- Populate TwoPassCullConstants ----
    TwoPassCullConstants cullConsts = {};
    cullConsts.NumInstances      = static_cast<uint>(totalInstances);
    cullConsts.NumMeshlets        = static_cast<uint>(totalMeshlets);
    cullConsts.HZBSRVIdx          = static_cast<uint>(m_HZB.srvIndex);
    cullConsts.HZBMipCount        = m_HZBMips;
    cullConsts.HZBWidth           = m_HZBWidth;
    cullConsts.HZBHeight          = m_HZBHeight;
    cullConsts.CandidateMeshletsCounterIdx  = static_cast<uint>(m_CandidateMeshletsCounter.uavIndex);
    cullConsts.CandidateMeshletsUAVIdx      = static_cast<uint>(m_CandidateMeshlets.uavIndex);
    cullConsts.OccludedInstancesCounterIdx  = static_cast<uint>(m_OccludedInstancesCounter.uavIndex);
    cullConsts.OccludedInstancesUAVIdx      = static_cast<uint>(m_OccludedInstances.uavIndex);
    cullConsts.OccludedInstancesSRVIdx      = static_cast<uint>(m_OccludedInstances.srvIndex);
    cullConsts.VisibleMeshletsUAVIdx        = static_cast<uint>(m_VisibleMeshlets.uavIndex);
    cullConsts.VisibleMeshletsCounterUAVIdx = static_cast<uint>(m_VisibleMeshletsCounter.uavIndex);
    cullConsts.Phase              = isFirstPhase ? TWO_PASS_PHASE_FIRST : TWO_PASS_PHASE_SECOND;
    cullConsts.EnableOcclusion    = occlusionEnabled ? 1u : 0u;

    // Double-buffered: Phase 1 → buffer[0], Phase 2 → buffer[1].
    // Prevents Phase 2's CPU-side memcpy from overwriting Phase 1's data
    // before the GPU executes Phase 1's dispatch (both are in the same
    // command list, but upload-heap writes are CPU-side and not deferred).
    uint cbIdx = isFirstPhase ? 0u : 1u;
    memcpy(m_TwoPassCullConstantsBuffer[cbIdx].cpuPtr, &cullConsts, sizeof(cullConsts));

    cmdList->SetComputeRootSignature(m_MeshletRootSignature.Get());
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());

    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootConstantBufferView(1, m_TwoPassCullConstantsBuffer[cbIdx].gpuAddress);

    // Bind SRVs by GPU virtual address
    cmdList->SetComputeRootShaderResourceView(2, model->GetInstanceDataBufferAddress());           // t0
    cmdList->SetComputeRootShaderResourceView(3, model->GetInstanceBoundsBufferAddress());         // t1
    cmdList->SetComputeRootShaderResourceView(4, model->GetMeshDataBufferAddress());              // t2
    cmdList->SetComputeRootShaderResourceView(5, model->GetGlobalMeshletBoundsBufferAddress());   // t3
    // NOTE: CandidateMeshlets is intentionally NOT bound as a separate SRV — CullMeshletsCS
    // reads it directly through the UAV (u0) that CullInstancesCS wrote via, avoiding a
    // resource-state hazard (same buffer can't be UNORDERED_ACCESS and SRV-readable at once).

    // Transition UAVs
    GraphicsHelper::TransitionResource(cmdList, m_CandidateMeshlets, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_CandidateMeshletsCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_OccludedInstances, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_OccludedInstancesCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshlets, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletsCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_MeshletCullArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_InstanceCullArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Zero counters — CandidateMeshletsCounter and VisibleMeshletsCounter are always fresh per phase.
    // OccludedInstancesCounter is only cleared in Phase 1 (first call); Phase 2 reads the list
    // built up by Phase 1.
    const UINT zeroes[4] = { 0, 0, 0, 0 };
    cmdList->ClearUnorderedAccessViewUint(
        GraphicsHelper::GetSRVGPUHandle((UINT)m_CandidateMeshletsCounter.uavIndex),
        GraphicsHelper::GetCpuUAVHandle((UINT)m_CandidateMeshletsCounter.cpuUavIndex),
        m_CandidateMeshletsCounter.resource.Get(), zeroes, 0, nullptr);
    if (isFirstPhase)
    {
        cmdList->ClearUnorderedAccessViewUint(
            GraphicsHelper::GetSRVGPUHandle((UINT)m_OccludedInstancesCounter.uavIndex),
            GraphicsHelper::GetCpuUAVHandle((UINT)m_OccludedInstancesCounter.cpuUavIndex),
            m_OccludedInstancesCounter.resource.Get(), zeroes, 0, nullptr);
    }
    cmdList->ClearUnorderedAccessViewUint(
        GraphicsHelper::GetSRVGPUHandle((UINT)m_VisibleMeshletsCounter.uavIndex),
        GraphicsHelper::GetCpuUAVHandle((UINT)m_VisibleMeshletsCounter.cpuUavIndex),
        m_VisibleMeshletsCounter.resource.Get(), zeroes, 0, nullptr);

    // Bind UAVs
    cmdList->SetComputeRootUnorderedAccessView(7,  m_CandidateMeshlets.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(8,  m_CandidateMeshletsCounter.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(9,  m_OccludedInstances.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(10, m_OccludedInstancesCounter.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(11, m_VisibleMeshlets.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(12, m_VisibleMeshletsCounter.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(13, m_MeshletCullArgs.gpuAddress);
    cmdList->SetComputeRootUnorderedAccessView(14, m_InstanceCullArgs.gpuAddress);

    if (isFirstPhase)
    {
        // ---- Phase 1: CullInstancesCS (direct dispatch) ----
        {
            GPU_MARKER(cmdList, L"TwoPassCull Phase1 - CullInstances");
            cmdList->SetPipelineState(m_CullInstancesPSO.Get());
            UINT groups = static_cast<UINT>((totalInstances + 63) / 64);
            cmdList->Dispatch(groups, 1, 1);
        }

        D3D12_RESOURCE_BARRIER uavBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_CandidateMeshlets.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_OccludedInstances.resource.Get()),
        };
        cmdList->ResourceBarrier(2, uavBarriers);

        // ---- BuildMeshletCullIndirectArgs ----
        {
            cmdList->SetPipelineState(m_BuildMeshletCullIndirectArgsPSO.Get());
            cmdList->Dispatch(1, 1, 1);
        }

        D3D12_RESOURCE_BARRIER argsBarrier =
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletCullArgs.resource.Get());
        cmdList->ResourceBarrier(1, &argsBarrier);
    }
    else
    {
        // ---- Phase 2: transition OccludedInstances for bindless SRV read ----
        // Phase 1 wrote to this buffer as UAV. Phase 2's CullInstancesCS reads it
        // as bindless SRV (via ResourceDescriptorHeap[OccludedInstancesSRVIdx]).
        GraphicsHelper::TransitionResource(cmdList, m_OccludedInstances,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // ---- Phase 2: BuildInstanceCullIndirectArgs first ----
        {
            GPU_MARKER(cmdList, L"TwoPassCull Phase2 - BuildInstanceArgs");
            cmdList->SetPipelineState(m_BuildInstanceCullIndirectArgsPSO.Get());
            cmdList->Dispatch(1, 1, 1);
        }

        D3D12_RESOURCE_BARRIER argsBarrier =
            CD3DX12_RESOURCE_BARRIER::UAV(m_InstanceCullArgs.resource.Get());
        cmdList->ResourceBarrier(1, &argsBarrier);

        // ---- Phase 2: CullInstancesCS (indirect dispatch from OccludedInstances) ----
        GraphicsHelper::TransitionResource(cmdList, m_InstanceCullArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        {
            GPU_MARKER(cmdList, L"TwoPassCull Phase2 - CullInstances");
            cmdList->SetPipelineState(m_CullInstancesPSO.Get());
            cmdList->ExecuteIndirect(m_DispatchCommandSignatureCS.Get(), 1,
                                     m_InstanceCullArgs.resource.Get(), 0, nullptr, 0);
        }

        D3D12_RESOURCE_BARRIER uavBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_CandidateMeshlets.resource.Get()),
        };
        cmdList->ResourceBarrier(1, uavBarriers);

        // ---- BuildMeshletCullIndirectArgs ----
        GraphicsHelper::TransitionResource(cmdList, m_MeshletCullArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        {
            cmdList->SetPipelineState(m_BuildMeshletCullIndirectArgsPSO.Get());
            cmdList->Dispatch(1, 1, 1);
        }

        D3D12_RESOURCE_BARRIER argsBarrier2 =
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletCullArgs.resource.Get());
        cmdList->ResourceBarrier(1, &argsBarrier2);
    }

    // ---- CullMeshletsCS (indirect dispatch) — shared by both phases ----
    GraphicsHelper::TransitionResource(cmdList, m_MeshletCullArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    {
        GPU_MARKER(cmdList, isFirstPhase ? L"TwoPassCull Phase1 - CullMeshlets"
                                         : L"TwoPassCull Phase2 - CullMeshlets");
        cmdList->SetPipelineState(m_CullMeshletsPSO.Get());
        cmdList->ExecuteIndirect(m_DispatchCommandSignatureCS.Get(), 1,
                                 m_MeshletCullArgs.resource.Get(), 0, nullptr, 0);
    }

    // Final barriers
    D3D12_RESOURCE_BARRIER finalBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_VisibleMeshlets.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_VisibleMeshletsCounter.resource.Get()),
    };
    cmdList->ResourceBarrier(2, finalBarriers);
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

// =============================================================================
// CreateTwoPassCullPipelines — compiles 4 CS PSOs using the unified m_MeshletRootSignature.
// =============================================================================
void MeshletPass::CreateTwoPassCullPipelines(ID3D12Device* device)
{
    // --- Compile CS PSOs (root signature already created as unified m_MeshletRootSignature) ---
    auto compileWithDefines = [&](const char* entry, const std::vector<std::pair<std::wstring, std::wstring>>& defines,
                                   Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso, const char* label)
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/MeshletTwoPassCull.hlsl", entry, "cs_6_6", defines);
        if (!cs.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = m_MeshletRootSignature.Get();
            desc.CS = { cs.data(), cs.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), label);
        }
    };

    auto compileNoDefines = [&](const char* entry, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso, const char* label)
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/MeshletTwoPassCull.hlsl", entry, "cs_6_6");
        if (!cs.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = m_MeshletRootSignature.Get();
            desc.CS = { cs.data(), cs.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), label);
        }
    };

    std::vector<std::pair<std::wstring, std::wstring>> occlusionDefines = {
        { L"OCCLUSION_CULL", L"1" }
    };

    compileWithDefines("CullInstancesCS", occlusionDefines, m_CullInstancesPSO,
            "[Meshlet] CreateComputePipelineState (CullInstancesCS) failed");
    compileWithDefines("CullMeshletsCS", occlusionDefines, m_CullMeshletsPSO,
            "[Meshlet] CreateComputePipelineState (CullMeshletsCS) failed");
    compileNoDefines(  "BuildMeshletCullIndirectArgsCS", m_BuildMeshletCullIndirectArgsPSO,
            "[Meshlet] CreateComputePipelineState (BuildMeshletCullIndirectArgs) failed");
    compileNoDefines(  "BuildInstanceCullIndirectArgsCS", m_BuildInstanceCullIndirectArgsPSO,
            "[Meshlet] CreateComputePipelineState (BuildInstanceCullIndirectArgs) failed");
}
