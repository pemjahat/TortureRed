#include "pch.h"

#include "DDGI.h"
#include "Core/Model.h"
#include "Graphics/GraphicsHelper.h"

#include <DirectXCollision.h>
#include <cmath>
#include <iostream>

// =============================================================================
// DDGI implementation.
//
// The atlas row length (m_ProbesPerRow) and the atlas sizes are derived from
// the probe count alone, so the C++ side and DDGI.hlsli cannot disagree about
// the layout. Every shader recomputes them with the same formulas.
// =============================================================================

namespace
{
// Must match DDGI.hlsli's DDGI_PROBE_IRRADIANCE_TEXELS / DDGI_PROBE_DEPTH_TEXELS
// plus the one-texel border ring on each side.
constexpr uint32_t kIrradianceTileExtent = 1 + DDGI::kIrradianceTexels + 1; // 8
constexpr uint32_t kDepthTileExtent      = 1 + DDGI::kDepthTexels + 1;      // 16

// Must match DDGI.hlsli's DDGI_RESOLVE_DISPATCH_X.
constexpr uint32_t kResolveDispatchX = 4096;

uint32_t ProbesPerRow(uint32_t probeCount)
{
    return std::max(1u, (uint32_t)std::ceil(std::sqrt((double)probeCount)));
}
} // namespace

void DDGI::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    auto createCompute = [&](const char* label, const char* file, const char* entry)
        -> Microsoft::WRL::ComPtr<ID3D12PipelineState>
    {
        std::cout << "[DDGI] compiling " << label << " shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader(file, entry, "cs_6_6");
        std::cout << "[DDGI] " << label << " shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
        if (!cs.empty())
        {
            desc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
        }
        return pso;
    };

    m_TracePSO            = createCompute("trace",            "Shaders/DDGI_Trace.hlsl",  "main");
    m_IrradiancePSO       = createCompute("irradiance",       "Shaders/DDGI_Update.hlsl", "UpdateIrradianceCS");
    m_DepthPSO            = createCompute("depth",            "Shaders/DDGI_Update.hlsl", "UpdateDepthCS");
    m_StatePSO            = createCompute("probe states",     "Shaders/DDGI_Update.hlsl", "UpdateProbeStatesCS");
    m_ClearIrradiancePSO  = createCompute("clear irradiance", "Shaders/DDGI_Update.hlsl", "ClearIrradianceCS");
    m_ClearDepthPSO       = createCompute("clear depth",      "Shaders/DDGI_Update.hlsl", "ClearDepthCS");
    m_ClearBuffersPSO     = createCompute("clear buffers",    "Shaders/DDGI_Update.hlsl", "ClearProbeBuffersCS");

    // --- Probe overlay (post-composite LDR, backbuffer RTV, no DSV) ---
    {
        std::cout << "[DDGI] compiling debug shaders..." << std::endl;
        auto vs = GraphicsHelper::CompileShader("Shaders/DDGI_Debug.hlsl", "VSMain", "vs_6_6");
        auto ps = GraphicsHelper::CompileShader("Shaders/DDGI_Debug.hlsl", "PSMain", "ps_6_6");
        std::cout << "[DDGI] debug shaders: " << (vs.empty() || ps.empty() ? "FAILED" : "ok") << std::endl;
        if (!vs.empty() && !ps.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature  = rootSignature;
            desc.VS              = { vs.data(), vs.size() };
            desc.PS              = { ps.data(), ps.size() };
            desc.BlendState      = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            desc.SampleMask      = UINT_MAX;
            desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            // Engine convention: geometry is CCW from outside.
            desc.RasterizerState.FrontCounterClockwise = TRUE;
            // The overlay is a closed sphere and this pass has no DSV, so both
            // hemispheres would rasterise and draw order would decide the pixel
            // winner: a ring of notches at the limb plus a vertical seam. The
            // engine convention makes the far hemisphere the back faces, so
            // culling BACK removes exactly it. Matches the BakedGI debug PSO.
            desc.RasterizerState.CullMode              = D3D12_CULL_MODE_BACK;
            desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets      = 1;
            desc.SampleDesc.Count      = 1;
            desc.RTVFormats[0]         = DXGI_FORMAT_R8G8B8A8_UNORM; // backbuffer
            desc.DepthStencilState.DepthEnable = FALSE; // manual reverse-Z compare in PSMain
            device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_DebugPSO));
        }
    }
}

bool DDGI::CreateResources()
{
    // Reuse the descriptor slots of a previous setup: the C++ and the HLSL
    // formulas agree, so the layout is stable and the views only need a
    // refresh. The caller synced the GPU, so replacing the buffers is safe.
    const uint32_t rows = (m_ProbeCount + m_ProbesPerRow - 1u) / m_ProbesPerRow;
    m_IrradianceWidth  = kIrradianceTileExtent * m_ProbesPerRow;
    m_IrradianceHeight = kIrradianceTileExtent * rows;
    m_DepthWidth       = kDepthTileExtent * m_ProbesPerRow;
    m_DepthHeight      = kDepthTileExtent * rows;

    if (m_IrradianceWidth  > kMaxAtlasDim || m_IrradianceHeight > kMaxAtlasDim ||
        m_DepthWidth       > kMaxAtlasDim || m_DepthHeight      > kMaxAtlasDim)
    {
        std::cerr << "[DDGI] Atlas " << m_IrradianceWidth << "x" << m_IrradianceHeight
                  << " / " << m_DepthWidth << "x" << m_DepthHeight
                  << " exceeds the " << kMaxAtlasDim << "-texel cap — reduce the probe count"
                  << std::endl;
        return false;
    }

    for (uint32_t i = 0; i < 2; ++i)
    {
        const char* irrName = (i == 0) ? "Tex_DDGIIrradiance0" : "Tex_DDGIIrradiance1";
        const char* depName = (i == 0) ? "Tex_DDGIDepth0" : "Tex_DDGIDepth1";
        if (!CreateTexture(m_Irradiance[i], m_IrradianceWidth, m_IrradianceHeight,
                           DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, irrName) ||
            !CreateTexture(m_Depth[i], m_DepthWidth, m_DepthHeight,
                           DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, depName))
        {
            std::cerr << "[DDGI] Failed to create the atlas pair" << std::endl;
            return false;
        }
    }

    if (!CreateStructuredBuffer(m_RayHits, sizeof(float) * 4,
                                (uint64_t)m_ProbeCount * kRaysPerProbe,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_DDGIRayHits") ||
        !CreateStructuredBuffer(m_ProbeOffsets, sizeof(float) * 4, m_ProbeCount,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_DDGIProbeOffsets") ||
        !CreateStructuredBuffer(m_ProbeStates, sizeof(uint32_t), m_ProbeCount,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_DDGIProbeStates"))
    {
        std::cerr << "[DDGI] Failed to create the probe buffers" << std::endl;
        return false;
    }
    return true;
}

void DDGI::RecordClears(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                        D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress)
{
    DDGIParams p = {};
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
    p.numRaysPerProbe = kRaysPerProbe;
    p.maxRaysPerProbe = kRaysPerProbe;
    p.irradianceTargetUAVIdx = (uint32_t)m_Irradiance[0].uavIndex;
    p.depthTargetUAVIdx      = (uint32_t)m_Depth[0].uavIndex;
    p.probeOffsetUAVIdx      = (uint32_t)m_ProbeOffsets.uavIndex;
    p.probeStateUAVIdx       = (uint32_t)m_ProbeStates.uavIndex;

    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(DDGIParams) / 4, &p, 0);

    const uint32_t irrGroupsX = (m_IrradianceWidth  + 7u) / 8u;
    const uint32_t irrGroupsY = (m_IrradianceHeight + 7u) / 8u;
    const uint32_t depGroupsX = (m_DepthWidth  + 7u) / 8u;
    const uint32_t depGroupsY = (m_DepthHeight + 7u) / 8u;

    for (uint32_t i = 0; i < 2; ++i)
    {
        // --- irradiance atlas ---
        p.irradianceTargetUAVIdx = (uint32_t)m_Irradiance[i].uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(DDGIParams) / 4, &p, 0);
        cmdList->SetPipelineState(m_ClearIrradiancePSO.Get());
        cmdList->Dispatch(irrGroupsX, irrGroupsY, 1);

        D3D12_RESOURCE_BARRIER irrBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Irradiance[i].resource.Get());
        cmdList->ResourceBarrier(1, &irrBarrier);

        // --- depth atlas ---
        p.depthTargetUAVIdx = (uint32_t)m_Depth[i].uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(DDGIParams) / 4, &p, 0);
        cmdList->SetPipelineState(m_ClearDepthPSO.Get());
        cmdList->Dispatch(depGroupsX, depGroupsY, 1);

        D3D12_RESOURCE_BARRIER depBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Depth[i].resource.Get());
        cmdList->ResourceBarrier(1, &depBarrier);
    }

    // --- probe offsets and states ---
    if (m_ClearBuffersPSO)
    {
        cmdList->SetPipelineState(m_ClearBuffersPSO.Get());
        cmdList->Dispatch((m_ProbeCount + 63u) / 64u, 1, 1);

        D3D12_RESOURCE_BARRIER bufferBarriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_ProbeOffsets.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_ProbeStates.resource.Get()),
        };
        cmdList->ResourceBarrier(2, bufferBarriers);
    }

    // The clears leave the atlases writable. RecordFrame reads them as
    // shader resources, so hand them over in a compute-read state.
    D3D12_RESOURCE_BARRIER done[4] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_Irradiance[0].resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_Irradiance[1].resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_Depth[0].resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_Depth[1].resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
    };
    cmdList->ResourceBarrier(4, done);
    m_Irradiance[0].state = m_Irradiance[1].state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    m_Depth[0].state      = m_Depth[1].state      = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
}

bool DDGI::RecordSetup(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                       ID3D12RootSignature* rootSignature, Model* model, float spacing,
                       D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress)
{
    m_Valid = false;
    if (!device || !cmdList || !rootSignature || !model || !m_TracePSO || !m_IrradiancePSO ||
        !m_DepthPSO || !m_StatePSO || !m_ClearIrradiancePSO || !m_ClearDepthPSO)
        return false;

    // --- Grid from the scene bounds (auto-grow the spacing to fit the caps).
    // The arithmetic mirrors BakedGI.cpp so the two grids stay comparable. ---
    const DirectX::BoundingBox bounds = model->GetSceneWorldBounds();
    DirectX::XMFLOAT3 mn;
    {
        const DirectX::XMFLOAT3 c = bounds.Center;
        const DirectX::XMFLOAT3 e = bounds.Extents;
        std::cout << "[DDGI] Scene bounds: min=(" << c.x - e.x << ", " << c.y - e.y << ", " << c.z - e.z
                  << ") max=(" << c.x + e.x << ", " << c.y + e.y << ", " << c.z + e.z
                  << ") size=(" << 2.0f * e.x << ", " << 2.0f * e.y << ", " << 2.0f * e.z << " m)" << std::endl;
        mn = { c.x - e.x - spacing, c.y - e.y - spacing, c.z - e.z - spacing };
        const DirectX::XMFLOAT3 mx = { c.x + e.x + spacing, c.y + e.y + spacing, c.z + e.z + spacing };

        m_Spacing = spacing;
        for (uint32_t guard = 0; guard < 32; ++guard)
        {
            m_DimX = std::max(2u, (uint32_t)std::ceil((mx.x - mn.x) / m_Spacing) + 1u);
            m_DimY = std::max(2u, (uint32_t)std::ceil((mx.y - mn.y) / m_Spacing) + 1u);
            m_DimZ = std::max(2u, (uint32_t)std::ceil((mx.z - mn.z) / m_Spacing) + 1u);
            const uint64_t count = (uint64_t)m_DimX * m_DimY * m_DimZ;
            if (m_DimX <= kMaxDim && m_DimY <= kMaxDim && m_DimZ <= kMaxDim && count <= kMaxProbeCount)
                break;
            m_Spacing *= 1.25f;
        }
    }
    m_ProbeCount   = m_DimX * m_DimY * m_DimZ;
    m_ProbesPerRow = ProbesPerRow(m_ProbeCount);
    m_GridMin      = mn;
    m_FrameSerial  = 0;

    if (m_Spacing != spacing)
        std::cout << "[DDGI] Spacing auto-grew " << spacing << "m -> " << m_Spacing
                  << "m to fit the probe cap (" << kMaxProbeCount << " probes / "
                  << kMaxDim << " per axis)" << std::endl;

    if (!CreateResources())
        return false;

    std::cout << "[DDGI] Probe grid: " << m_DimX << "x" << m_DimY << "x" << m_DimZ
              << " @ " << m_Spacing << "m (" << m_ProbeCount << " probes, "
              << kRaysPerProbe << " rays each, " << kStableRays << " stable)"
              << std::endl;
    std::cout << "[DDGI] Atlases: irradiance " << m_IrradianceWidth << "x" << m_IrradianceHeight
              << " RGBA16F, depth " << m_DepthWidth << "x" << m_DepthHeight << " RG16F, "
              << m_ProbesPerRow << " probes per row — "
              << (GetResourceBytes() >> 20) << " MB total" << std::endl;

    RecordClears(cmdList, rootSignature, frameCBAddress);

    m_Valid = true;
    return true;
}

void DDGI::RecordFrame(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                       D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                       D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                       D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                       D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                       D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                       D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                       const DirectX::XMFLOAT3& sunIrradiance,
                       const DirectX::XMFLOAT3& sunToDir)
{
    if (!m_Valid || !m_TracePSO || !m_IrradiancePSO || !m_DepthPSO || !m_StatePSO)
        return;

    const uint32_t target  = m_FrameSerial & 1u;
    const uint32_t history = 1u - target;

    // --- Atlas state transitions. The target pair must be writable; the
    // history pair must be readable by the trace pass and the resolves. ---
    m_Irradiance[history].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_Depth[history].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_Irradiance[target].Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_Depth[target].Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    DDGIParams p = {};
    p.randomVectorX = 0.5f; p.randomVectorY = 1.0f; p.randomVectorZ = 0.5f; // fixed axis
    // Per-frame angle so the rotating rays sweep the sphere over time.
    p.randomAngle = (float)m_FrameSerial * 2.39996323f; // golden angle
    // Frame 0 sent zero: the setup cleared the atlases, and a full replace
    // removes any dependence on the cleared value.
    p.historyBlendWeight = (m_FrameSerial == 0u) ? 0.0f : kHistoryBlendWeight;
    p.sunIrradianceX = sunIrradiance.x; p.sunIrradianceY = sunIrradiance.y; p.sunIrradianceZ = sunIrradiance.z;
    p.sunDirX = sunToDir.x; p.sunDirY = sunToDir.y; p.sunDirZ = sunToDir.z;
    p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
    p.spacing  = m_Spacing;
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
    p.numRaysPerProbe = kRaysPerProbe;
    p.maxRaysPerProbe = kRaysPerProbe;
    p.irradianceHistorySRVIdx = (uint32_t)m_Irradiance[history].srvIndex;
    p.depthHistorySRVIdx      = (uint32_t)m_Depth[history].srvIndex;
    p.irradianceTargetUAVIdx  = (uint32_t)m_Irradiance[target].uavIndex;
    p.depthTargetUAVIdx       = (uint32_t)m_Depth[target].uavIndex;
    p.rayHitUAVIdx            = (uint32_t)m_RayHits.uavIndex;
    p.rayHitSRVIdx            = (uint32_t)m_RayHits.srvIndex;
    p.probeOffsetSRVIdx       = (uint32_t)m_ProbeOffsets.srvIndex;
    p.probeOffsetUAVIdx       = (uint32_t)m_ProbeOffsets.uavIndex;
    p.probeStateSRVIdx        = (uint32_t)m_ProbeStates.srvIndex;
    p.probeStateUAVIdx        = (uint32_t)m_ProbeStates.uavIndex;

    // --- Common compute bindings. The trace pass traces rays, so it needs the
    // TLAS and the scene buffers on the same root slots BakedGI's bake uses. ---
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, materialsAddress);
    cmdList->SetComputeRootShaderResourceView(2, drawNodesAddress);
    cmdList->SetComputeRootShaderResourceView(3, tlasAddress);
    cmdList->SetComputeRootShaderResourceView(4, indicesAddress);
    cmdList->SetComputeRootShaderResourceView(5, verticesAddress);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(DDGIParams) / 4, &p, 0);

    // --- 1. Trace ---
    cmdList->SetPipelineState(m_TracePSO.Get());
    cmdList->Dispatch((m_ProbeCount + 7u) / 8u, (kRaysPerProbe + 7u) / 8u, 1);

    {
        // The resolves read the ray buffer and the probe buffers as shader
        // resources; the state resolve writes the probe buffers. One barrier
        // set orders all three.
        D3D12_RESOURCE_BARRIER traceBarriers[3] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_RayHits.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_ProbeOffsets.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_ProbeStates.resource.Get()),
        };
        cmdList->ResourceBarrier(3, traceBarriers);
    }

    // --- 2. Irradiance and depth resolves. One thread group for each probe,
    // the probe index spread over both dispatch axes. ---
    const uint32_t resolveX = std::min(kResolveDispatchX, std::max(1u, m_ProbeCount));
    const uint32_t resolveY = (m_ProbeCount + kResolveDispatchX - 1u) / kResolveDispatchX;

    cmdList->SetPipelineState(m_IrradiancePSO.Get());
    cmdList->Dispatch(resolveX, resolveY, 1);

    D3D12_RESOURCE_BARRIER irrBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Irradiance[target].resource.Get());
    cmdList->ResourceBarrier(1, &irrBarrier);

    cmdList->SetPipelineState(m_DepthPSO.Get());
    cmdList->Dispatch(resolveX, resolveY, 1);

    D3D12_RESOURCE_BARRIER depBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Depth[target].resource.Get());
    cmdList->ResourceBarrier(1, &depBarrier);

    // --- 3. Probe relocation and state classification ---
    cmdList->SetPipelineState(m_StatePSO.Get());
    cmdList->Dispatch((m_ProbeCount + 31u) / 32u, 1, 1);

    {
        D3D12_RESOURCE_BARRIER stateBarriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_ProbeOffsets.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_ProbeStates.resource.Get()),
        };
        cmdList->ResourceBarrier(2, stateBarriers);
    }

    // --- Hand the target pair to the deferred consumer. ---
    m_Irradiance[target].Transition(cmdList, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    m_Depth[target].Transition(cmdList, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    ++m_FrameSerial;
}

uint64_t DDGI::GetResourceBytes() const
{
    const uint64_t irradiance = 2ull * m_IrradianceWidth * m_IrradianceHeight * 8ull;
    const uint64_t depth      = 2ull * m_DepthWidth * m_DepthHeight * 4ull;
    const uint64_t rayHits    = (uint64_t)m_ProbeCount * kRaysPerProbe * 16ull;
    const uint64_t offsets    = (uint64_t)m_ProbeCount * 16ull;
    const uint64_t states     = (uint64_t)m_ProbeCount * 4ull;
    return irradiance + depth + rayHits + offsets + states;
}
