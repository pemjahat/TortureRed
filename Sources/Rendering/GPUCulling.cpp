#include "pch.h"

#include "GPUCulling.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

#define A_CPU 1
#include "ffx_a.h"
#include "ffx_spd.h"

void GPUCulling::CreateResources(uint32_t internalWidth, uint32_t internalHeight)
{
    static constexpr UINT MAX_VISIBLE_MESHLETS = 1 << 20;

    // Visible meshlets list (RW) — output of CullMeshletsCS, input to MeshletPass::Binning
    if (!CreateStructuredBuffer(m_VisibleMeshlets, sizeof(MeshletCandidate), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_VisibleMeshlets"))
    {
        std::cerr << "[GPUCulling] Failed to create VisibleMeshlets buffer" << std::endl;
        return;
    }

    // 2 slots: [TWO_PASS_PHASE_FIRST]/[TWO_PASS_PHASE_SECOND] — see m_VisibleMeshletsCounter
    // comment in GPUCulling.h.
    if (!CreateBuffer(m_VisibleMeshletsCounter, sizeof(uint32_t) * 2,
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, true, "SB_VisibleMeshletsCounter"))
    {
        std::cerr << "[GPUCulling] Failed to create VisibleMeshletsCounter" << std::endl;
        return;
    }
    if (!CreateBuffer(m_VisibleInstancesCounter, sizeof(uint32_t) * 2,
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, true, "SB_VisibleInstancesCounter"))
        std::cerr << "[GPUCulling] Failed to create VisibleInstancesCounter" << std::endl;

    // ----- Two-pass occlusion culling buffers -----
    if (!CreateStructuredBuffer(m_CandidateMeshlets, sizeof(MeshletCandidate), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_CandidateMeshlets"))
        std::cerr << "[GPUCulling] Failed to create CandidateMeshlets buffer" << std::endl;
    if (!CreateBuffer(m_CandidateMeshletsCounter, sizeof(uint32_t),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, true, "SB_CandidateMeshletsCounter"))
        std::cerr << "[GPUCulling] Failed to create CandidateMeshletsCounter" << std::endl;
    if (!CreateStructuredBuffer(m_OccludedInstances, sizeof(uint32_t), static_cast<UINT>(MAX_VISIBLE_MESHLETS / 4),
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_OccludedInstances"))
        std::cerr << "[GPUCulling] Failed to create OccludedInstances buffer" << std::endl;
    if (!CreateBuffer(m_OccludedInstancesCounter, sizeof(uint32_t),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, true, "SB_OccludedInstancesCounter"))
        std::cerr << "[GPUCulling] Failed to create OccludedInstancesCounter" << std::endl;
    if (!CreateStructuredBuffer(m_MeshletCullArgs, sizeof(uint32_t), 3,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_MeshletCullArgs"))
        std::cerr << "[GPUCulling] Failed to create MeshletCullArgs" << std::endl;
    if (!CreateStructuredBuffer(m_InstanceCullArgs, sizeof(uint32_t), 3,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_InstanceCullArgs"))
        std::cerr << "[GPUCulling] Failed to create InstanceCullArgs" << std::endl;
    if (!CreateBuffer(m_TwoPassCullConstantsBuffer[0], 256, D3D12_HEAP_TYPE_UPLOAD,
                      D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_TwoPassCullConstants_0"))
        std::cerr << "[GPUCulling] Failed to create TwoPassCullConstantsBuffer[0]" << std::endl;
    if (!CreateBuffer(m_TwoPassCullConstantsBuffer[1], 256, D3D12_HEAP_TYPE_UPLOAD,
                      D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_TwoPassCullConstants_1"))
        std::cerr << "[GPUCulling] Failed to create TwoPassCullConstantsBuffer[1]" << std::endl;

    // Occluded-rect debug recording buffers
    if (!CreateStructuredBuffer(m_OccludedRects, sizeof(OccludedRectDebug), MAX_OCCLUDED_RECT_DEBUG,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_OccludedRectsDebug"))
        std::cerr << "[GPUCulling] Failed to create occluded-rects debug buffer" << std::endl;
    if (!CreateBuffer(m_OccludedRectsCounter, sizeof(uint32_t),
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      true, true, "SB_OccludedRectsDebugCounter"))
        std::cerr << "[GPUCulling] Failed to create occluded-rects debug counter" << std::endl;

    // Cull stats debug buffer — written by CopyCullStatsCS after each phase
    if (!CreateBuffer(m_CullStatsBuffer, sizeof(uint32_t) * CULL_STATS_COUNT,
                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      true, true, "SB_CullStats"))
        std::cerr << "[GPUCulling] Failed to create CullStats buffer" << std::endl;

    // Mip-selection tint sideband
    if (!CreateStructuredBuffer(m_VisibleMeshletMips, sizeof(uint32_t), MAX_VISIBLE_MESHLETS,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_VisibleMeshletMips"))
        std::cerr << "[GPUCulling] Failed to create VisibleMeshletMips buffer" << std::endl;

    CreateHZBResources(internalWidth, internalHeight);

    std::cout << "[GPUCulling] Resources created (max " << MAX_VISIBLE_MESHLETS << " visible meshlets)" << std::endl;
}

void GPUCulling::RecreateHZB(uint32_t internalWidth, uint32_t internalHeight)
{
    CreateHZBResources(internalWidth, internalHeight);
}

void GPUCulling::CreateHZBResources(uint32_t internalWidth, uint32_t internalHeight)
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

    if (!CreateTexture(m_HZB, m_HZBWidth, m_HZBHeight, DXGI_FORMAT_R32_FLOAT,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, m_HZBMips, 1, "Tex_HZB"))
    {
        std::cerr << "[GPUCulling] Failed to create HZB texture" << std::endl;
        return;
    }

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
        m_HZBMipUAVIndices[mip] = m_HZBMipUAVIndices[m_HZBMips - 1];

    if (!CreateStructuredBuffer(m_HZBMipIndicesBuffer, sizeof(uint32_t), SPD_MAX_MIPS,
                                D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, "SB_HZBMipIndices"))
    {
        std::cerr << "[GPUCulling] Failed to create HZBMipIndices buffer" << std::endl;
        return;
    }
    memcpy(m_HZBMipIndicesBuffer.cpuPtr, m_HZBMipUAVIndices, sizeof(uint32_t) * SPD_MAX_MIPS);

    if (!CreateStructuredBuffer(m_HZBSpdCounter, sizeof(uint32_t), 1,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_HZBSpdCounter"))
    {
        std::cerr << "[GPUCulling] Failed to create HZBSpdCounter buffer" << std::endl;
        return;
    }

    std::cout << "[GPUCulling] HZB created (" << m_HZBWidth << "x" << m_HZBHeight
              << ", " << m_HZBMips << " mips)" << std::endl;
}

void GPUCulling::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* mainRootSignature)
{
    // --- Unified Meshlet Cull Root Signature ---
    // Fully bindless — no root SRV/UAV descriptors. Every buffer MeshletTwoPassCull.hlsl
    // touches is looked up via ResourceDescriptorHeap[CullConst.*Idx] instead (see the
    // file-header comment in MeshletTwoPassCull.hlsl and docs/bug_flyingworld_meshlet_flicker.md).
    // Just the two CBVs + the static sampler HZBCull() needs.
    {
        CD3DX12_ROOT_PARAMETER rootParams[2];
        rootParams[0].InitAsConstantBufferView(0);          // b0: FrameConstants
        rootParams[1].InitAsConstantBufferView(1);          // b1: TwoPassCullConstants

        CD3DX12_STATIC_SAMPLER_DESC pointClampSampler(0, D3D12_FILTER_MIN_MAG_MIP_POINT,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

        CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
        rsDesc.Init(2, rootParams, 1, &pointClampSampler,
                    D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);

        Microsoft::WRL::ComPtr<ID3DBlob> signature, error;
        HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(hr))
        {
            if (error) std::cerr << "[GPUCulling] Root signature error: " << (char*)error->GetBufferPointer() << std::endl;
            return;
        }
        CHECK_HR(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                              IID_PPV_ARGS(&m_CullRootSignature)), "[GPUCulling] CreateRootSignature failed");
    }

    // Two-pass cull PSOs
    {
        auto compileWithDefines = [&](const char* entry, const std::vector<std::pair<std::wstring, std::wstring>>& defines,
                                       Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso, const char* label)
        {
            auto cs = GraphicsHelper::CompileShader("Shaders/MeshletTwoPassCull.hlsl", entry, "cs_6_6", defines);
            if (!cs.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = m_CullRootSignature.Get();
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
                desc.pRootSignature = m_CullRootSignature.Get();
                desc.CS = { cs.data(), cs.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), label);
            }
        };
        std::vector<std::pair<std::wstring, std::wstring>> occlusionDefines = { { L"OCCLUSION_CULL", L"1" } };
        compileWithDefines("CullInstancesCS", occlusionDefines, m_CullInstancesPSO,
                "[GPUCulling] CreateComputePipelineState (CullInstancesCS) failed");
        compileWithDefines("CullMeshletsCS", occlusionDefines, m_CullMeshletsPSO,
                "[GPUCulling] CreateComputePipelineState (CullMeshletsCS) failed");
        compileNoDefines("BuildMeshletCullIndirectArgsCS", m_BuildMeshletCullIndirectArgsPSO,
                "[GPUCulling] CreateComputePipelineState (BuildMeshletCullIndirectArgs) failed");
        compileNoDefines("BuildInstanceCullIndirectArgsCS", m_BuildInstanceCullIndirectArgsPSO,
                "[GPUCulling] CreateComputePipelineState (BuildInstanceCullIndirectArgs) failed");
    }

    // Debug overlay PSOs
    {
        auto compile = [&](const char* file, const char* entry, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso, const char* label)
        {
            auto cs = GraphicsHelper::CompileShader(file, entry, "cs_6_6");
            if (!cs.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature; // use cull root sig — these don't bind b1 root params
                desc.CS = { cs.data(), cs.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), label);
            }
        };
        // HZB mip viewer — uses MAIN root signature via dispatch caller
        {
            auto cs = GraphicsHelper::CompileShader("Shaders/HZBDebugView.hlsl", "HZBDebugViewCS", "cs_6_6");
            if (!cs.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature; // caller overrides
                desc.CS = { cs.data(), cs.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_HZBDebugViewPSO)),
                         "[GPUCulling] CreateComputePipelineState (HZB debug view) failed");
            }
        }
        // Occluded-rect overlay
        {
            auto csBg = GraphicsHelper::CompileShader("Shaders/OccludedRectDebug.hlsl", "OccludedRectBackgroundCS", "cs_6_6");
            if (!csBg.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature;
                desc.CS = { csBg.data(), csBg.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_OccludedRectBackgroundPSO)),
                         "[GPUCulling] CreateComputePipelineState (occluded-rect background) failed");
            }
            auto csRects = GraphicsHelper::CompileShader("Shaders/OccludedRectDebug.hlsl", "OccludedRectsCS", "cs_6_6");
            if (!csRects.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature;
                desc.CS = { csRects.data(), csRects.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_OccludedRectsPSO)),
                         "[GPUCulling] CreateComputePipelineState (occluded-rects) failed");
            }
        }
        // Depth readout
        {
            auto cs = GraphicsHelper::CompileShader("Shaders/DepthReadout.hlsl", "DepthReadoutCS", "cs_6_6");
            if (!cs.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature;
                desc.CS = { cs.data(), cs.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_DepthReadoutPSO)),
                         "[GPUCulling] CreateComputePipelineState (depth readout) failed");
            }
        }
        // Copy cull stats (1-thread CS: copies per-phase functional counters → debug stats buffer)
        {
            auto cs = GraphicsHelper::CompileShader("Shaders/CullStats.hlsl", "CopyCullStatsCS", "cs_6_6");
            if (!cs.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature;
                desc.CS = { cs.data(), cs.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_CopyCullStatsPSO)),
                         "[GPUCulling] CreateComputePipelineState (CopyCullStats) failed");
            }
        }
        // Cull stats overlay (1-thread CS: reads debug stats buffer, writes GPU debug text)
        {
            auto cs = GraphicsHelper::CompileShader("Shaders/CullStats.hlsl", "CullStatsCS", "cs_6_6");
            if (!cs.empty())
            {
                D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
                desc.pRootSignature = mainRootSignature;
                desc.CS = { cs.data(), cs.size() };
                CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_CullStatsPSO)),
                         "[GPUCulling] CreateComputePipelineState (CullStats) failed");
            }
        }
    }

    // HZB PSOs
    {
        CD3DX12_ROOT_PARAMETER rootParams[1];
        rootParams[0].InitAsConstantBufferView(0);
        CD3DX12_STATIC_SAMPLER_DESC pointClampSampler(0, D3D12_FILTER_MIN_MAG_MIP_POINT,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
        rsDesc.Init(1, rootParams, 1, &pointClampSampler,
                    D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);
        Microsoft::WRL::ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
        if (FAILED(hr))
        {
            if (err) std::cerr << "[GPUCulling] HZB root sig error: " << (char*)err->GetBufferPointer() << std::endl;
            return;
        }
        CHECK_HR(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                              IID_PPV_ARGS(&m_HZBRootSignature)), "[GPUCulling] CreateRootSignature (HZB) failed");

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
        compile("HZBInitCS",   m_HZBInitPSO,   "[GPUCulling] CreateComputePipelineState (HZB init) failed");
        compile("HZBCreateCS", m_HZBCreatePSO, "[GPUCulling] CreateComputePipelineState (HZB create) failed");
    }

    // Indirect Dispatch command signature (for CullMeshletsCS / CullInstancesCS ExecuteIndirect)
    {
        D3D12_INDIRECT_ARGUMENT_DESC dispatchArg = {};
        dispatchArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC sigDesc = {};
        sigDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        sigDesc.NumArgumentDescs = 1;
        sigDesc.pArgumentDescs = &dispatchArg;
        CHECK_HR(device->CreateCommandSignature(&sigDesc, nullptr,
                 IID_PPV_ARGS(&m_DispatchCommandSignatureCS)),
                 "[GPUCulling] CreateCommandSignature (dispatch) failed");
    }

    std::cout << "[GPUCulling] Pipelines created" << std::endl;
}

void GPUCulling::BuildHZB(ID3D12GraphicsCommandList* cmdList, GPUTexture& depthBuffer)
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

    varAU2(dispatchThreadGroupCountXY);
    varAU2(workGroupOffset);
    varAU2(numWorkGroupsAndMips);
    varAU4(rectInfo) = initAU4(0, 0, m_HZBWidth, m_HZBHeight);
    SpdSetup(dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips, rectInfo,
             static_cast<int>(m_HZBMips) - 1);

    hzbConstants.NumWorkGroups    = numWorkGroupsAndMips[0];
    hzbConstants.WorkGroupOffsetX = workGroupOffset[0];
    hzbConstants.WorkGroupOffsetY = workGroupOffset[1];

    if (m_HZBConstantsBuffer.resource == nullptr)
    {
        if (!CreateBuffer(m_HZBConstantsBuffer, 256, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_HZBConstants"))
        {
            std::cerr << "[GPUCulling] Failed to create HZBConstants buffer" << std::endl;
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

    D3D12_RESOURCE_BARRIER hzbDoneBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_HZB.resource.Get());
    cmdList->ResourceBarrier(1, &hzbDoneBarrier);
    GraphicsHelper::TransitionResource(cmdList, m_HZB, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

void GPUCulling::DebugViewHZB(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                               uint32_t outputUAVIdx, uint32_t outputWidth, uint32_t outputHeight, int mipLevel)
{
    if (!m_HZBDebugViewPSO || !m_HZB.resource)
        return;

    int clampedMip = mipLevel < 0 ? 0 : (mipLevel >= (int)m_HZBMips ? (int)m_HZBMips - 1 : mipLevel);

    HZBDebugParams params = {};
    params.HZBSRVIdx    = static_cast<uint32_t>(m_HZB.srvIndex);
    params.MipLevel     = static_cast<uint32_t>(clampedMip);
    params.OutputUAVIdx = outputUAVIdx;
    params.Width        = outputWidth;
    params.Height       = outputHeight;

    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetPipelineState(m_HZBDebugViewPSO.Get());
    cmdList->SetComputeRoot32BitConstants(13, sizeof(HZBDebugParams) / 4, &params, 0);
    cmdList->Dispatch((outputWidth + 7) / 8, (outputHeight + 7) / 8, 1);
}

void GPUCulling::DrawOccludedRects(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                                    D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, GPUTexture& output,
                                    uint32_t outputWidth, uint32_t outputHeight)
{
    if (!m_OccludedRectBackgroundPSO || !m_OccludedRectsPSO || !m_OccludedRects.resource)
        return;

    GraphicsHelper::TransitionResource(cmdList, m_OccludedRects, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_OccludedRectsCounter, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    OccludedRectDrawParams params = {};
    params.RectsSRVIdx      = static_cast<uint32_t>(m_OccludedRects.srvIndex);
    params.RectsCountSRVIdx = static_cast<uint32_t>(m_OccludedRectsCounter.srvIndex);
    params.OutputUAVIdx     = static_cast<uint32_t>(output.uavIndex);
    params.Width            = outputWidth;
    params.Height           = outputHeight;

    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRoot32BitConstants(13, sizeof(OccludedRectDrawParams) / 4, &params, 0);

    cmdList->SetPipelineState(m_OccludedRectBackgroundPSO.Get());
    cmdList->Dispatch((outputWidth + 7) / 8, (outputHeight + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(output.resource.Get());
    cmdList->ResourceBarrier(1, &uavBarrier);

    cmdList->SetPipelineState(m_OccludedRectsPSO.Get());
    cmdList->Dispatch(MAX_OCCLUDED_RECT_DEBUG / 64, 1, 1);
}

void GPUCulling::EmitDepthReadout(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                                   uint32_t dataUAVIdx, uint32_t glyphSRVIdx, float fontSize,
                                   uint32_t backbufferWidth, uint32_t backbufferHeight)
{
    if (!m_DepthReadoutPSO || !m_OccludedRects.resource)
        return;

    GraphicsHelper::TransitionResource(cmdList, m_OccludedRects, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_OccludedRectsCounter, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    DepthReadoutParams params = {};
    params.RectsSRVIdx      = static_cast<uint32_t>(m_OccludedRects.srvIndex);
    params.RectsCountSRVIdx = static_cast<uint32_t>(m_OccludedRectsCounter.srvIndex);
    params.DataUAVIdx       = dataUAVIdx;
    params.GlyphSRVIdx      = glyphSRVIdx;
    params.FontSize         = fontSize;
    params.BackbufferWidth  = static_cast<float>(backbufferWidth);
    params.BackbufferHeight = static_cast<float>(backbufferHeight);
    params.MaxLabels        = 64;

    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetPipelineState(m_DepthReadoutPSO.Get());
    cmdList->SetComputeRoot32BitConstants(13, sizeof(DepthReadoutParams) / 4, &params, 0);
    cmdList->Dispatch(1, 1, 1);
}

void GPUCulling::EmitCullStats(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                                D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                                uint32_t dataUAVIdx, uint32_t glyphSRVIdx, float fontSize,
                                uint32_t backbufferWidth, uint32_t backbufferHeight,
                                uint32_t totalInstances, uint32_t totalMeshlets)
{
    if (!m_CullStatsPSO || !m_CullStatsBuffer.resource)
        return;

    GraphicsHelper::TransitionResource(cmdList, m_CullStatsBuffer,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    CullStatsParams params = {};
    params.StatsBufferSRVIdx = static_cast<uint32_t>(m_CullStatsBuffer.srvIndex);
    params.DataUAVIdx        = dataUAVIdx;
    params.GlyphSRVIdx       = glyphSRVIdx;
    params.FontSize          = fontSize;
    params.TotalInstances    = totalInstances;
    params.TotalMeshlets     = totalMeshlets;
    params.BackbufferWidth   = backbufferWidth;
    params.BackbufferHeight  = backbufferHeight;
    params.StartX            = 10.0f;
    params.StartY            = 10.0f;

    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetPipelineState(m_CullStatsPSO.Get());
    cmdList->SetComputeRoot32BitConstants(13, sizeof(CullStatsParams) / 4, &params, 0);
    cmdList->Dispatch(1, 1, 1);
}

void GPUCulling::CullTwoPass(ID3D12GraphicsCommandList* cmdList, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                               Model* model, bool occlusionEnabled, int phase,
                               ID3D12RootSignature* mainRootSignature)
{
    if (!m_CullInstancesPSO || !m_CullMeshletsPSO || !model->IsMeshletReady())
        return;

    size_t totalInstances = model->GetInstanceCount();
    size_t totalMeshlets  = model->GetTotalMeshletCount();
    if (totalInstances == 0 || totalMeshlets == 0) return;

    const bool isFirstPhase = (phase == 0);

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
    cullConsts.DebugRecordOccluded        = m_DebugRecordOccluded ? 1u : 0u;
    cullConsts.OccludedRectsUAVIdx        = static_cast<uint>(m_OccludedRects.uavIndex);
    cullConsts.OccludedRectsCounterUAVIdx = static_cast<uint>(m_OccludedRectsCounter.uavIndex);
    // NOTE: mip-tint flag is set via SetDebugRecordMip() by the caller before dispatch.
    cullConsts.DebugRecordMip             = m_DebugRecordMipEnabled ? 1u : 0u;
    cullConsts.VisibleMeshletMipsUAVIdx   = static_cast<uint>(m_VisibleMeshletMips.uavIndex);

    // Fully-bindless resource indices (see MeshletTwoPassCull.hlsl's file-header comment
    // and docs/bug_flyingworld_meshlet_flicker.md) — no root SRV/UAV descriptors are bound
    // for these below; the shader looks them all up via ResourceDescriptorHeap[idx].
    cullConsts.InstanceDataSRVIdx             = static_cast<uint>(model->GetInstanceDataSRVIndex());
    cullConsts.InstanceBoundsSRVIdx           = static_cast<uint>(model->GetInstanceBoundsSRVIndex());
    cullConsts.MeshDataSRVIdx                 = static_cast<uint>(model->GetMeshDataSRVIndex());
    cullConsts.MeshletBoundsSRVIdx            = static_cast<uint>(model->GetGlobalMeshletBoundsSRVIndex());
    cullConsts.MeshletCullArgsUAVIdx          = static_cast<uint>(m_MeshletCullArgs.uavIndex);
    cullConsts.InstanceCullArgsUAVIdx         = static_cast<uint>(m_InstanceCullArgs.uavIndex);
    cullConsts.VisibleInstancesCounterUAVIdx  = static_cast<uint>(m_VisibleInstancesCounter.uavIndex);

    uint cbIdx = isFirstPhase ? 0u : 1u;
    memcpy(m_TwoPassCullConstantsBuffer[cbIdx].cpuPtr, &cullConsts, sizeof(cullConsts));

    cmdList->SetComputeRootSignature(m_CullRootSignature.Get());
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());

    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootConstantBufferView(1, m_TwoPassCullConstantsBuffer[cbIdx].gpuAddress);

    GraphicsHelper::TransitionResource(cmdList, m_CandidateMeshlets, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_CandidateMeshletsCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_OccludedInstances, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_OccludedInstancesCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_VisibleInstancesCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshlets, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletsCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_MeshletCullArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_InstanceCullArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    if (m_DebugRecordMipEnabled)
    {
        GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletMips, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

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

        // m_VisibleMeshletsCounter[2] must be cleared ONLY ONCE per frame (Phase 1's start).
        // Phase 2 must see Phase 1's final slot [TWO_PASS_PHASE_FIRST] value intact — it uses
        // that as the base offset when appending its own candidates into VisibleMeshlets, so
        // it never overwrites/aliases Phase 1's already-rasterized range.
        cmdList->ClearUnorderedAccessViewUint(
            GraphicsHelper::GetSRVGPUHandle((UINT)m_VisibleMeshletsCounter.uavIndex),
            GraphicsHelper::GetCpuUAVHandle((UINT)m_VisibleMeshletsCounter.cpuUavIndex),
            m_VisibleMeshletsCounter.resource.Get(), zeroes, 0, nullptr);
    }
    cmdList->ClearUnorderedAccessViewUint(
        GraphicsHelper::GetSRVGPUHandle((UINT)m_VisibleInstancesCounter.uavIndex),
        GraphicsHelper::GetCpuUAVHandle((UINT)m_VisibleInstancesCounter.cpuUavIndex),
        m_VisibleInstancesCounter.resource.Get(), zeroes, 0, nullptr);

    if (isFirstPhase && m_DebugRecordOccluded)
    {
        GraphicsHelper::TransitionResource(cmdList, m_OccludedRects, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(cmdList, m_OccludedRectsCounter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ClearUnorderedAccessViewUint(
            GraphicsHelper::GetSRVGPUHandle((UINT)m_OccludedRectsCounter.uavIndex),
            GraphicsHelper::GetCpuUAVHandle((UINT)m_OccludedRectsCounter.cpuUavIndex),
            m_OccludedRectsCounter.resource.Get(), zeroes, 0, nullptr);
    }
    if (isFirstPhase)
    {
        GPU_MARKER(cmdList, L"TwoPassCull Phase1 - CullInstances");
        cmdList->SetPipelineState(m_CullInstancesPSO.Get());
        UINT groups = static_cast<UINT>((totalInstances + 63) / 64);
        cmdList->Dispatch(groups, 1, 1);

        D3D12_RESOURCE_BARRIER uavBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_CandidateMeshlets.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_OccludedInstances.resource.Get()),
        };
        cmdList->ResourceBarrier(2, uavBarriers);

        cmdList->SetPipelineState(m_BuildMeshletCullIndirectArgsPSO.Get());
        cmdList->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER argsBarrier =
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletCullArgs.resource.Get());
        cmdList->ResourceBarrier(1, &argsBarrier);
    }
    else
    {
        GraphicsHelper::TransitionResource(cmdList, m_OccludedInstances,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        GPU_MARKER(cmdList, L"TwoPassCull Phase2 - BuildInstanceArgs");
        cmdList->SetPipelineState(m_BuildInstanceCullIndirectArgsPSO.Get());
        cmdList->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER argsBarrier =
            CD3DX12_RESOURCE_BARRIER::UAV(m_InstanceCullArgs.resource.Get());
        cmdList->ResourceBarrier(1, &argsBarrier);

        GraphicsHelper::TransitionResource(cmdList, m_InstanceCullArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        GPU_MARKER(cmdList, L"TwoPassCull Phase2 - CullInstances");
        cmdList->SetPipelineState(m_CullInstancesPSO.Get());
        cmdList->ExecuteIndirect(m_DispatchCommandSignatureCS.Get(), 1,
                                 m_InstanceCullArgs.resource.Get(), 0, nullptr, 0);

        D3D12_RESOURCE_BARRIER uavBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_CandidateMeshlets.resource.Get()),
        };
        cmdList->ResourceBarrier(1, uavBarriers);

        GraphicsHelper::TransitionResource(cmdList, m_MeshletCullArgs, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->SetPipelineState(m_BuildMeshletCullIndirectArgsPSO.Get());
        cmdList->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER argsBarrier2 =
            CD3DX12_RESOURCE_BARRIER::UAV(m_MeshletCullArgs.resource.Get());
        cmdList->ResourceBarrier(1, &argsBarrier2);
    }

    GraphicsHelper::TransitionResource(cmdList, m_MeshletCullArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    {
        GPU_MARKER(cmdList, isFirstPhase ? L"TwoPassCull Phase1 - CullMeshlets"
                                         : L"TwoPassCull Phase2 - CullMeshlets");
        cmdList->SetPipelineState(m_CullMeshletsPSO.Get());
        cmdList->ExecuteIndirect(m_DispatchCommandSignatureCS.Get(), 1,
                                 m_MeshletCullArgs.resource.Get(), 0, nullptr, 0);
    }

    D3D12_RESOURCE_BARRIER finalBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_VisibleMeshlets.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_VisibleMeshletsCounter.resource.Get()),
    };
    cmdList->ResourceBarrier(2, finalBarriers);

    // m_VisibleMeshletsCounter is read via SRV by TWO consumers after this point:
    //   1. MeshletPass::BuildDispatchMeshArgs (always-on — builds the mesh-shader
    //      dispatch args from the visible-meshlet count).
    //   2. CopyCullStatsCS below (debug-only overlay).
    // It was previously left tracked in UNORDERED_ACCESS state here (only a UAV
    // barrier above, never a state transition), so both SRV reads were illegal
    // (D3D12 debug layer: RESOURCE_STATE_MISMATCH — a resource must be in
    // NON_PIXEL_SHADER_RESOURCE/PIXEL_SHADER_RESOURCE state to be read via SRV).
    // Transition unconditionally — not gated on m_CopyCullStatsPSO — so
    // BuildDispatchMeshArgs's read stays legal even if the debug PSOs failed
    // to compile. CullTwoPass's own start-of-phase transition (back to
    // UNORDERED_ACCESS, above) already accounts for this before the next
    // culling dispatch writes it again.
    GraphicsHelper::TransitionResource(cmdList, m_VisibleMeshletsCounter,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Copy per-phase functional counters to the debug stats buffer.
    // CopyCullStatsCS is a 1-thread compute shader that reads the current
    // counter values and writes them to CullStatsBuffer at the appropriate slot.
    // Uses the MAIN root signature (not m_CullRootSignature) so it can access
    // root param 13 for CullStatsCopyParams.
    if (m_CopyCullStatsPSO)
    {
        GraphicsHelper::TransitionResource(cmdList, m_CandidateMeshletsCounter,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_OccludedInstancesCounter,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_VisibleInstancesCounter,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(cmdList, m_CullStatsBuffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        CullStatsCopyParams copyParams = {};
        copyParams.CandidateCounterSRVIdx           = static_cast<uint32_t>(m_CandidateMeshletsCounter.srvIndex);
        copyParams.VisibleCounterSRVIdx             = static_cast<uint32_t>(m_VisibleMeshletsCounter.srvIndex);
        copyParams.OccludedCounterSRVIdx            = static_cast<uint32_t>(m_OccludedInstancesCounter.srvIndex);
        copyParams.InstanceVisibleCounterSRVIdx     = static_cast<uint32_t>(m_VisibleInstancesCounter.srvIndex);
        copyParams.StatsBufferUAVIdx                = static_cast<uint32_t>(m_CullStatsBuffer.uavIndex);
        // Two-phase: Phase 0 → slots [0..3], Phase 1 (SECOND) → slots [4..7]
        copyParams.BaseSlot                         = (!isFirstPhase) ? 4u : 0u;

        cmdList->SetComputeRootSignature(mainRootSignature);
        cmdList->SetComputeRoot32BitConstants(13, sizeof(CullStatsCopyParams) / 4, &copyParams, 0);
        cmdList->SetPipelineState(m_CopyCullStatsPSO.Get());
        cmdList->Dispatch(1, 1, 1);
    }
}
