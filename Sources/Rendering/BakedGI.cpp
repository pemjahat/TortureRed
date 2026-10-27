#include "pch.h"

#include "BakedGI.h"
#include "Core/Model.h"
#include "Graphics/GraphicsHelper.h"

#include <DirectXCollision.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

// =============================================================================
// Probes/ disk cache — the ShaderCache pattern applied to the bake outputs.
// =============================================================================
namespace
{

constexpr uint32_t kProbeCacheMagic   = 0x42504744; // 'BGPD'
constexpr uint32_t kProbeCacheVersion = 1; // 0 -> 1 (2026-10-26): the method-3 sun
                                           // table is a second per-probe record set.
                                           // It is NOT in the cache today because the
                                           // cache is gated off (kProbeCacheEnabled),
                                           // but a re-enable must not read a version-0
                                           // file into a layout that expects it.

#pragma pack(push, 4)
struct ProbeCacheHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t dimX, dimY, dimZ;
    uint32_t probeCount;
    uint32_t finalResponse; // ping-pong index holding the final table
    float    spacing;
    float    gridMinX, gridMinY, gridMinZ;
};
#pragma pack(pop)

std::filesystem::path ProbeCachePath(const std::string& scenePath, float spacing)
{
    const std::filesystem::path dir = GraphicsHelper::GetCacheDir(L"Probes");
    if (dir.empty())
        return {};
    char sp[32];
    snprintf(sp, sizeof(sp), "%.2f", static_cast<double>(spacing));
    return dir / (std::filesystem::path(scenePath).stem().string() + "_s" + sp + ".probebin");
}

} // namespace

void BakedGI::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    // --- Bake (compute, uses inline ray queries via CommonTracing) ---
    {
        std::cout << "[BakedGI] compiling bake shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_Bake.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] bake shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_BakePSO));
        }
    }

    // --- Step-3 visibility bake: octahedral depth moments (compute, ray queries) ---
    {
        std::cout << "[BakedGI] compiling visibility bake shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_Visibility.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] visibility bake shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_VisBakePSO));
        }
    }

    // --- Per-frame lit-probe update (compute) ---
    {
        std::cout << "[BakedGI] compiling update shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_Update.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] update shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_UpdatePSO));
        }
    }

    // --- Fullscreen backface-contamination leak map (post-composite, alpha-blended) ---
    {
        std::cout << "[BakedGI] compiling leak-map shaders..." << std::endl;
        auto vs = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "LeakVS", "vs_6_6");
        auto ps = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "LeakPS", "ps_6_6");
        std::cout << "[BakedGI] leak-map shaders: " << (vs.empty() || ps.empty() ? "FAILED" : "ok") << std::endl;
        if (!vs.empty() && !ps.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature         = rootSignature;
            desc.VS                      = { vs.data(), vs.size() };
            desc.PS                      = { ps.data(), ps.size() };
            desc.BlendState              = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            desc.SampleMask              = UINT_MAX;
            desc.RasterizerState         = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT); // FillMode=SOLID etc.
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;                  // fullscreen triangle
            desc.PrimitiveTopologyType   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets        = 1;
            desc.SampleDesc.Count        = 1;
            desc.RTVFormats[0]           = DXGI_FORMAT_R8G8B8A8_UNORM; // backbuffer (post-composite)
            desc.DepthStencilState.DepthEnable = FALSE;
            // Alpha blend: sky pixels write a=0 so the tonemapped scene shows
            // through; geometry pixels (a=1) replace.
            desc.BlendState.RenderTarget[0].BlendEnable    = TRUE;
            desc.BlendState.RenderTarget[0].SrcBlend       = D3D12_BLEND_SRC_ALPHA;
            desc.BlendState.RenderTarget[0].DestBlend      = D3D12_BLEND_INV_SRC_ALPHA;
            desc.BlendState.RenderTarget[0].BlendOp        = D3D12_BLEND_OP_ADD;
            desc.BlendState.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
            desc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
            desc.BlendState.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
            HRESULT hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_LeakDebugPSO));
            if (FAILED(hr))
                std::cerr << "[BakedGI] leak-map PSO creation failed (hr = 0x"
                          << std::hex << hr << std::dec << ")" << std::endl;
        }
    }

    // --- Probe placement debug cubes (post-composite LDR overlay) ---
    {
        std::cout << "[BakedGI] compiling debug shaders..." << std::endl;
        auto vs = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "VSMain", "vs_6_6");
        auto ps = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "PSMain", "ps_6_6");
        std::cout << "[BakedGI] debug shaders: " << (vs.empty() || ps.empty() ? "FAILED" : "ok") << std::endl;
        if (!vs.empty() && !ps.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature         = rootSignature;
            desc.VS                      = { vs.data(), vs.size() };
            desc.PS                      = { ps.data(), ps.size() };
            desc.BlendState              = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            desc.SampleMask              = UINT_MAX;
            desc.RasterizerState         = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            // Engine convention (GBuffer/DeferredLighting/Shadow/Transparency/
            // Meshlet all set this): geometry is CCW-from-outside.
            desc.RasterizerState.FrontCounterClockwise = TRUE;
            desc.PrimitiveTopologyType   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets        = 1;
            desc.SampleDesc.Count        = 1;
            desc.RTVFormats[0]           = DXGI_FORMAT_R8G8B8A8_UNORM; // backbuffer (post-composite)
            // No DSV: occlusion is a manual reverse-Z compare in PSMain against
            // the GBuffer depth SRV (internal-res, mapped in the shader).
            desc.DepthStencilState.DepthEnable = FALSE;
            device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_DebugPSO));
        }
    }

    // --- Ray-debug re-trace (compute, on demand — debug only) ---
    {
        std::cout << "[BakedGI] compiling ray-debug shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_RayDebug.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] ray-debug shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_RayDebugPSO));
        }
    }

    // --- Per-term response-ball fold (compute, debug view 8) ---
    // Rides the bake list, so it is compiled here with the bake: a shader error
    // then shows up in this log line rather than only in a missing view.
    {
        std::cout << "[BakedGI] compiling iteration-fold shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_IterFold.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] iteration-fold shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_IterFoldPSO));
        }
    }
}

bool BakedGI::RecordBake(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                         ID3D12RootSignature* rootSignature, Model* model,
                         D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                         float spacing, bool singleDirectionSun,
                         const DirectX::XMFLOAT3& sunToDir)
{
    if (!device || !cmdList || !rootSignature || !model || !m_BakePSO)
        return false;

    // The caller synced the GPU — release the previous bake's cache staging
    // now (the upload buffer must only die between executions, never while a
    // recorded list still references it).
    m_CacheUpload.Reset();
    m_CacheReadback.Reset();
    m_LoadedFromCache = false;
    // The sun table belongs to the previous bake's sun and grid. Clear the flag
    // first, so an early return cannot leave a stale sun table readable.
    m_SunResponseValid = false;
    // The per-term capture is produced by the dispatch path below. Cleared here
    // so the cache-load early return cannot leave a previous bake's balls
    // standing as if they described this grid.
    m_EfbBallsValid   = false;

    // --- Grid from the scene bounds (auto-grow spacing to fit the caps) ---
    const DirectX::BoundingBox bounds = model->GetSceneWorldBounds();
    DirectX::XMFLOAT3 mn, mx;
    {
        DirectX::XMFLOAT3 c = bounds.Center, e = bounds.Extents;
        DirectX::XMFLOAT3 sceneMin = { c.x - e.x, c.y - e.y, c.z - e.z };
        DirectX::XMFLOAT3 sceneMax = { c.x + e.x, c.y + e.y, c.z + e.z };
        std::cout << "[BakedGI] Scene bounds: min=(" << sceneMin.x << ", " << sceneMin.y << ", " << sceneMin.z
                  << ") max=(" << sceneMax.x << ", " << sceneMax.y << ", " << sceneMax.z << ") "
                  << "size=(" << sceneMax.x - sceneMin.x << ", " << sceneMax.y - sceneMin.y << ", "
                  << sceneMax.z - sceneMin.z << " m)" << std::endl;
        mn = { sceneMin.x - spacing, sceneMin.y - spacing, sceneMin.z - spacing };
        mx = { sceneMax.x + spacing, sceneMax.y + spacing, sceneMax.z + spacing };
    }

    m_Spacing = spacing;
    for (uint32_t guard = 0; guard < 32; ++guard)
    {
        m_DimX = std::max(2u, (uint32_t)std::ceil((mx.x - mn.x) / m_Spacing) + 1u);
        m_DimY = std::max(2u, (uint32_t)std::ceil((mx.y - mn.y) / m_Spacing) + 1u);
        m_DimZ = std::max(2u, (uint32_t)std::ceil((mx.z - mn.z) / m_Spacing) + 1u);
        uint64_t count = (uint64_t)m_DimX * m_DimY * m_DimZ;
        if (m_DimX <= kMaxDim && m_DimY <= kMaxDim && m_DimZ <= kMaxDim && count <= kMaxProbeCount)
            break;
        m_Spacing *= 1.25f;
    }
    m_ProbeCount = m_DimX * m_DimY * m_DimZ;
    m_GridMin = mn;
    m_Valid = false; // set at the end; the buffers below are the source of truth

    // The grid identity is now (re)established. Bumped here rather than on the
    // success path so the cache-load early return below is covered too: both
    // paths end up with this grid. A bump on a bake that later fails only costs
    // the UI one rebuild.
    ++m_GridRevision;

    if (m_Spacing != spacing)
        std::cout << "[BakedGI] Spacing auto-grew " << spacing << "m -> " << m_Spacing
                  << "m to fit the probe cap (" << kMaxProbeCount << " probes / "
                  << kMaxDim << " per axis)" << std::endl;

    std::cout << "[BakedGI] Baking probe grid: " << m_DimX << "x" << m_DimY << "x" << m_DimZ
              << " : " << m_Spacing << "m (" << m_ProbeCount << " probes, "
              << kBakeIterations << " series iterations, "
              << kBakeGatherRays << " gather rays for each of " << kDirections << " directions";
    // Name the sun pass when it runs. Without this the line understates the work
    // by one whole direction and never mentions the dispatch that the method
    // exists for, which is how a stale mental model survives a session.
    if (singleDirectionSun)
        std::cout << " + 1 sun pass";
    std::cout << ")"
              << " — world " << (mx.x - mn.x) << "x" << (mx.y - mn.y) << "x" << (mx.z - mn.z) << "m" << std::endl;

    // --- Probes/ disk cache: on a hit (file exists + grid header matches),
    // restore the outputs with plain buffer copies and skip the dispatches
    // entirely. Gated off right now (BakedGI::kProbeCacheEnabled) so the bake
    // always runs; the load path itself is unchanged. ---
    if constexpr (kProbeCacheEnabled)
    {
        if (TryLoadCache(device, cmdList, model->GetSourcePath()))
            return true;
    }

    if (!CreateProbeBuffers(singleDirectionSun))
        return false;

    // Debug-only capture buffers (term snapshots + folded balls). A failure here
    // does not void the bake: the capture is a debug aid, so record without it
    // and let the "Efb Contribution" view report that it has nothing to read.
    const bool iterCapture = CreateIterationBuffers();
    if (!iterCapture)
        std::cerr << "[BakedGI] iteration capture unavailable — the Efb Contribution view will be empty" << std::endl;

    // --- Record the series-expansion iterations (ping-pong) ---
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, materialsAddress);
    cmdList->SetComputeRootShaderResourceView(2, drawNodesAddress);
    cmdList->SetComputeRootShaderResourceView(3, tlasAddress);
    cmdList->SetComputeRootShaderResourceView(4, indicesAddress);
    cmdList->SetComputeRootShaderResourceView(5, verticesAddress);

    // --- Step-3 visibility bake FIRST: the transport iterations' feedback
    // reads are visibility-weighted (BakedGI_Bake.hlsl EvalResponseIndirect
    // gates each feedback cage probe with backface + Chebyshev weights read
    // from these moment maps), so the moments must exist before iteration 1.
    // Rides the same list and root bindings (CBV b0 + material/draw-node/
    // TLAS/index/vertex SRVs) as the transport dispatches; only the b2
    // root-constants block and PSO change. One 16x16-thread group per probe;
    // probes tile across (x = kVisDispatchX, y) to respect the 65535-per-axis
    // dispatch cap (kMaxProbeCount is 131072).
    if (m_VisBakePSO)
    {
        BakedGIVisParams vp = {};
        vp.gridMinX = m_GridMin.x; vp.gridMinY = m_GridMin.y; vp.gridMinZ = m_GridMin.z;
        vp.spacing  = m_Spacing;
        vp.dimX = m_DimX; vp.dimY = m_DimY; vp.dimZ = m_DimZ;
        vp.probeCount = m_ProbeCount;
        vp.visUAVIdx  = (uint32_t)m_Visibility.uavIndex;
        vp.metaUAVIdx = (uint32_t)m_Meta.uavIndex;

        cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIVisParams) / 4, &vp, 0);
        cmdList->SetPipelineState(m_VisBakePSO.Get());
        cmdList->Dispatch(kVisDispatchX, (m_ProbeCount + kVisDispatchX - 1) / kVisDispatchX, 1);

        // Barrier BOTH outputs: the transport iterations read the meta bits
        // the classification lane wrote (the visibility barrier alone would
        // leave that write-read unordered).
        D3D12_RESOURCE_BARRIER visBarriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_Visibility.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_Meta.resource.Get()),
        };
        cmdList->ResourceBarrier(2, visBarriers);
    }
    else
    {
        std::cerr << "[BakedGI] Visibility bake PSO missing — Chebyshev weights skipped" << std::endl;
    }

    uint32_t cur = 0;
    for (uint32_t it = 0; it < kBakeIterations; ++it)
    {
        BakedGIBakeParams p = {};
        p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
        p.spacing  = m_Spacing;
        p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
        p.probeCount = m_ProbeCount;
        p.responseUAVIdx    = (uint32_t)m_Response[cur].uavIndex;
        p.responsePrevSRVIdx = (uint32_t)m_Response[1 - cur].srvIndex;
        p.metaUAVIdx         = (uint32_t)m_Meta.uavIndex;
        p.iteration          = it;
        p.visSRVIdx          = (uint32_t)m_Visibility.srvIndex;
        p.sunMode            = 0u; // 16-bin transport: unchanged

        cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIBakeParams) / 4, &p, 0);
        cmdList->SetPipelineState(m_BakePSO.Get());
        cmdList->Dispatch((m_ProbeCount + 7) / 8, kDirections / 8, 1);

        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Response[cur].resource.Get());
        cmdList->ResourceBarrier(1, &barrier);

        // ---- Method 3: the sun's own response, toward the REAL sun direction.
        //
        // This is the whole point of the method. The binned transport asks "how
        // does the probe respond to a delta light from W_j"; the runtime fold then
        // has to pretend R_14 describes the sun, which is 14.66 degrees away, and
        // the fold ends up averaging visibility over a 23 degree cone. This pass
        // asks the right question instead, so there is nothing to interpolate.
        //
        // ONE dispatch, not 16: there is one sun. Same PSO, same feedback cage,
        // same visibility weighting — sunMode only redirects wDelta, the buffer
        // and the record stride. The series still converges against its own
        // operator, which is what a sky-fed shortcut could not do.
        if (singleDirectionSun && m_SunResponse[0].resource)
        {
            BakedGIBakeParams sp = p;
            sp.sunMode            = 1u;
            sp.sunDirX            = sunToDir.x;
            sp.sunDirY            = sunToDir.y;
            sp.sunDirZ            = sunToDir.z;
            sp.responseUAVIdx     = (uint32_t)m_SunResponse[cur].uavIndex;
            sp.responsePrevSRVIdx = (uint32_t)m_SunResponse[1 - cur].srvIndex;

            cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIBakeParams) / 4, &sp, 0);
            cmdList->SetPipelineState(m_BakePSO.Get());
            cmdList->Dispatch((m_ProbeCount + 7) / 8, 1, 1);

            D3D12_RESOURCE_BARRIER sunBarrier =
                CD3DX12_RESOURCE_BARRIER::UAV(m_SunResponse[cur].resource.Get());
            cmdList->ResourceBarrier(1, &sunBarrier);
        }

        // Snapshot this term's table for the debug view. The LAST term needs no
        // copy: it is m_Response[m_FinalResponse] once the loop ends, so
        // snapshot[kBakeIterations - 1] is deliberately absent.
        if (iterCapture && it + 1u < kBakeIterations)
            SnapshotResponseTable(cmdList, cur, it);

        m_FinalResponse = cur;
        cur = 1 - cur;
    }

    m_Valid = true;

    // The sun table ping-pongs on the same schedule as the binned one, so the
    // final sun record sits in the same index the final response does.
    m_SunResponseValid = singleDirectionSun && (m_SunResponse[0].resource != nullptr);
    m_FinalSunResponse = m_FinalResponse;
    // What the resident table describes. HasSunResponseFor compares against this,
    // so a moved sun is rejected instead of reused.
    if (m_SunResponseValid)
        m_SunBakeDir = sunToDir;

    // The table just changed, so any previous probe-table read-back describes
    // the OLD bake. Clear the flag: the sun-bin panel then shows no contribution
    // column rather than another bake's numbers.
    m_ProbeTableValid = false;

    // --- Per-term response balls (debug view 8: the Efb contribution) ---
    // One fold for each term. The view differences ball[k] and ball[0]; by the
    // bake's own identity R^k = R^0 + G(Efb_k) that difference IS the series-
    // feedback contribution of term k. Uses the bindings already on this list
    // (descriptor heap, root signature, b0, b2) — only the PSO and the b2 block
    // change, the same way the visibility bake rides the transport dispatches.
    if (iterCapture && m_IterFoldPSO)
    {
        // The snapshots were written by CopyBufferRegion, so they are COPY_DEST;
        // the fold reads them as an SRV. The final table needs no transition here
        // because the transport iterations already read their tables the same
        // way (a UAV barrier, no state change).
        for (uint32_t s = 0; s < kIterSnapshotCount; ++s)
            m_ResponseIter[s].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        for (uint32_t t = 0; t < kBakeIterations; ++t)
        {
            BakedGIIterFoldParams fp = {};
            fp.probeCount  = m_ProbeCount;
            fp.tableSRVIdx = (t + 1u < kBakeIterations)
                           ? (uint32_t)m_ResponseIter[t].srvIndex
                           : (uint32_t)m_Response[m_FinalResponse].srvIndex;
            fp.ballUAVIdx  = (uint32_t)m_EfbBall[t].uavIndex;

            cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIIterFoldParams) / 4, &fp, 0);
            cmdList->SetPipelineState(m_IterFoldPSO.Get());
            cmdList->Dispatch((m_ProbeCount + 63u) / 64u, 1, 1);
        }

        // One barrier for all three outputs: the debug view transitions them for
        // read anyway, but the bake must not end with unordered writes.
        D3D12_RESOURCE_BARRIER ballBarriers[kBakeIterations];
        for (uint32_t t = 0; t < kBakeIterations; ++t)
            ballBarriers[t] = CD3DX12_RESOURCE_BARRIER::UAV(m_EfbBall[t].resource.Get());
        cmdList->ResourceBarrier(kBakeIterations, ballBarriers);

        m_EfbBallsValid = true;
    }

    std::cout << "[BakedGI] Bake recorded (" << m_ProbeCount << " probes, "
              << "visibility: 16x16 depth moments x" << kVisStride << " texels/probe, "
              << (m_EfbBallsValid ? "per-term response balls captured"
                                  : "no per-term capture")
              << ")" << std::endl;
    return true;
}

bool BakedGI::RecordMetaReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList)
{
    if (!m_Valid || !device || !cmdList || !m_Meta.resource)
        return false;

    const UINT64 byteSize = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&m_MetaReadback))))
    {
        std::cerr << "[BakedGI] Failed to create the meta readback buffer" << std::endl;
        return false;
    }
    GraphicsHelper::SetObjectName(m_MetaReadback.Get(), "RB_BakedGIMeta");

    // Appended to the bake list: meta (UAV after the dispatches) -> readback.
    // Round-trips back to UAV so the runtime update pass — which reads the
    // meta SRV without transitions — is unaffected.
    m_Meta.Transition(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->CopyBufferRegion(m_MetaReadback.Get(), 0, m_Meta.resource.Get(), 0, byteSize);
    m_Meta.Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return true;
}

void BakedGI::LogBakeStats() const
{
    if (!m_Valid || !m_MetaReadback)
        return;

    uint32_t* meta = nullptr;
    if (FAILED(m_MetaReadback->Map(0, nullptr, reinterpret_cast<void**>(&meta))))
    {
        std::cerr << "[BakedGI] Failed to map the meta readback buffer" << std::endl;
        return;
    }

    uint32_t ddgiValid = 0, oldValid = 0, newlyBad = 0, newlyValid = 0;
    uint32_t deadBackface = 0, deadFar = 0;
    for (uint32_t i = 0; i < m_ProbeCount; ++i)
    {
        const uint32_t m = meta[i];
        const bool v  = (m & 0x1u) != 0; // BAKED_GI_META_VALID (DDGI criteria)
        const bool ov = (m & 0x2u) != 0; // BAKED_GI_META_OLD_VALID (1cm test)
        ddgiValid += v ? 1u : 0u;
        oldValid  += ov ? 1u : 0u;
        newlyBad   += (!v && ov) ? 1u : 0u;
        newlyValid += (v && !ov) ? 1u : 0u;
        deadBackface += (m & 0x4u) ? 1u : 0u;
        deadFar      += (m & 0x8u) ? 1u : 0u;
    }
    m_MetaReadback->Unmap(0, nullptr);

    const float pctNew = 100.0f * (float)ddgiValid / (float)m_ProbeCount;
    const float pctOld = 100.0f * (float)oldValid / (float)m_ProbeCount;
    std::cout << "[BakedGI] Probe coverage (DDGI criteria): " << ddgiValid << "/" << m_ProbeCount
              << " valid (" << pctNew << "%) — dead: " << deadBackface
              << " by >= 25% backfaces, " << deadFar << " by > 3x spacing far rule" << std::endl;
    std::cout << "[BakedGI] Old 1cm test for comparison:      " << oldValid << "/" << m_ProbeCount
              << " valid (" << pctOld << "%)" << std::endl;
    std::cout << "[BakedGI] Criteria swap flips: " << newlyBad
              << " valid -> BAD (blue in placement view), " << newlyValid
              << " bad -> valid (yellow)" << std::endl;
}

// -----------------------------------------------------------------------------
// Ray-debug tool — debug only, on demand
// -----------------------------------------------------------------------------

namespace
{

// The shader's luminance weights (CommonTracing.hlsl Luminance). Kept local so
// this debug path shares no symbol with the renderer.
inline float RayDebugLuminance(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// One float4 of the response table for a direction and an SH term.
inline const float* RayDebugTable(const uint8_t* base, uint32_t dir, uint32_t lm)
{
    return reinterpret_cast<const float*>(base) + ((size_t)dir * 10u + lm) * 4u;
}

// rec3.w packs TWO values into one float, because the record is a fixed 96 B and
// no field is spare:  status + 4 * sourceRayIndex. BakedGI_RayDebug.hlsl owns the
// encode. Status keeps its old meaning and its old three values (0 = sky miss,
// 1 = backface reject, 2 = shaded), so a reader decodes instead of comparing.
//
// Exactness: fp32 represents every integer up to 2^24, and the largest packed
// value is 2 + 4 * 255 = 1022, so std::fmod is exact and the round trip loses
// nothing. Read rec3.w directly and you get the packed value, not the status.
inline float RayDebugStatus(const float* rec)
{
    return std::fmod(rec[12 + 3], 4.0f);
}

inline uint32_t RayDebugSourceRay(const float* rec)
{
    return static_cast<uint32_t>(std::lround((rec[12 + 3] - RayDebugStatus(rec)) * 0.25f));
}

} // namespace

bool BakedGI::RecordRayDebug(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                             ID3D12RootSignature* rootSignature,
                             D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                             uint32_t probeIndex)
{
    if (!device || !cmdList || !rootSignature || !m_RayDebugPSO)
        return false;
    if (!m_Valid || m_ProbeCount == 0)
        return false;

    // Clamp into the grid: a stale UI value must not read outside the table.
    if (probeIndex >= m_ProbeCount)
        probeIndex = m_ProbeCount - 1u;
    m_RayDebugProbe = probeIndex;
    // Clear the verdict for this dump. A skipped assignment then yields no
    // verdict, whatever the index value is. The pairing check alone cannot catch
    // a re-bake that flips the verdict at the SAME index.
    m_RayDebugValidProbe = UINT32_MAX;

    // The Cage Trace pool describes the OLD records. Clear it here, not in the
    // readout: a dump whose readout never runs — no readback, or a failed Map —
    // must leave an EMPTY pool, so the UI cannot offer a ray index that belongs
    // to a buffer which no longer exists.
    for (auto& list : m_RayCandidates)
        list.clear();
    m_RayCandidateDirs = 0u;

    // The caller synced the GPU before recording, so releasing the previous
    // readback here is safe (the meta-readback pattern). m_RayDebugBuf is NOT
    // cleared: CreateStructuredBuffer reuses the existing srvIndex/uavIndex, so
    // a repeated dump does not leak descriptor slots (the heap has no free).
    m_RayDebugReadback.Reset();

    // --- GPU output: records, then the re-accumulated table (one UAV). ---
    // CreateStructuredBuffer allocates the SRV and UAV role descriptors, so the
    // shader reads the UAV through ResourceDescriptorHeap[m_RayDebugBuf.uavIndex].
    if (!CreateStructuredBuffer(m_RayDebugBuf, 16u, kRayDebugBufF4s,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_BakedGIRayDebug"))
    {
        std::cerr << "[BakedGI] Failed to create the ray-debug buffer" << std::endl;
        return false;
    }

    // --- CPU readback: debug buffer | stored table. ---
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(kRayDebugReadbackBytes);
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&m_RayDebugReadback))))
        {
            std::cerr << "[BakedGI] Failed to create the ray-debug readback buffer" << std::endl;
            return false;
        }
        GraphicsHelper::SetObjectName(m_RayDebugReadback.Get(), "RB_BakedGIRayDebug");
    }

    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, materialsAddress);
    cmdList->SetComputeRootShaderResourceView(2, drawNodesAddress);
    cmdList->SetComputeRootShaderResourceView(3, tlasAddress);
    cmdList->SetComputeRootShaderResourceView(4, indicesAddress);
    cmdList->SetComputeRootShaderResourceView(5, verticesAddress);

    BakedGIRayDebugParams p = {};
    p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
    p.spacing  = m_Spacing;
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
    p.probeIndex         = m_RayDebugProbe;
    p.rayRecUAVIdx       = (uint32_t)m_RayDebugBuf.uavIndex;
    // The final iteration (kBakeIterations - 1 = 2) reads the previous
    // iteration's table for its feedback term. With the 3-iteration odd
    // ping-pong, that buffer is 1 - m_FinalResponse.
    p.responsePrevSRVIdx = (uint32_t)m_Response[1 - m_FinalResponse].srvIndex;
    p.visSRVIdx          = (uint32_t)m_Visibility.srvIndex;

    cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIRayDebugParams) / 4, &p, 0);
    cmdList->SetPipelineState(m_RayDebugPSO.Get());
    cmdList->Dispatch(1, kDirections, 1);

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_RayDebugBuf.resource.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }

    // Debug buffer -> readback.
    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
            m_RayDebugBuf.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b);
        cmdList->CopyBufferRegion(m_RayDebugReadback.Get(), 0, m_RayDebugBuf.resource.Get(), 0, kRayDebugBufBytes);
        D3D12_RESOURCE_BARRIER b2 = CD3DX12_RESOURCE_BARRIER::Transition(
            m_RayDebugBuf.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);
    }

    // The probe's slice of the stored table -> readback, for the self-check.
    {
        ID3D12Resource* stored = m_Response[m_FinalResponse].resource.Get();
        const UINT64 srcOffset = (UINT64)m_RayDebugProbe * kDirections * kResponseFloat4s * 16u;
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
            stored, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b);
        cmdList->CopyBufferRegion(m_RayDebugReadback.Get(), kRayDebugStoredOff, stored,
                                  srcOffset, kRayDebugTableBytes);
        D3D12_RESOURCE_BARRIER b2 = CD3DX12_RESOURCE_BARRIER::Transition(
            stored, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);
    }

    return true;
}

void BakedGI::TransitionRayDebugForRead(ID3D12GraphicsCommandList* cmdList)
{
    if (!cmdList || !m_RayDebugBuf.resource)
        return;
    // PLACEMENT. The transition lives at the read site, not inside
    // RecordRayDebug. Two reasons:
    //   1. CreateStructuredBuffer calls CreateCommittedResource on every dump,
    //      so each dump builds a NEW resource with state = UNORDERED_ACCESS.
    //      The tracker therefore cannot go stale between dumps, and the next
    //      dispatch needs no repair.
    //   2. The read site owns its own read barrier, the way
    //      Renderer::DrawBakedGIProbeDebug transitions the GBuffer depth.
    // GPUResource::Transition guards on a state change, so drawing the fan on
    // two frames in a row emits one barrier, not two.
    m_RayDebugBuf.Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

BakedGI::ProbeMetaStatus BakedGI::ReadProbeMeta(uint32_t probe, uint32_t& metaWord) const
{
    // The meta readback is created by RecordMetaReadback on both the cold-bake
    // and the cache-load path, and it persists for the life of the bake. Map it
    // again here: the buffer is CPU-readable and nothing writes it after the
    // bake, so a second Map is safe (LogBakeStats uses the same buffer).
    if (!m_MetaReadback)
        return ProbeMetaStatus::NoReadback;
    if (probe >= m_ProbeCount)
        return ProbeMetaStatus::ProbeOutOfRange;

    uint32_t* meta = nullptr;
    if (FAILED(m_MetaReadback->Map(0, nullptr, reinterpret_cast<void**>(&meta))))
        return ProbeMetaStatus::MapFailed;
    metaWord = meta[probe];
    m_MetaReadback->Unmap(0, nullptr);
    return ProbeMetaStatus::Ok;
}

const std::vector<BakedGI::RayCandidate>& BakedGI::GetRayCandidates(uint32_t dir) const
{
    // An out-of-range direction gets an empty list rather than a clamp: the
    // caller asked about a direction that does not exist, and answering with
    // direction 0's pool would put another direction's rays under that label.
    static const std::vector<RayCandidate> kEmpty;
    return (dir < kDirections) ? m_RayCandidates[dir] : kEmpty;
}

void BakedGI::LogRayDebugReadout(const DirectX::XMFLOAT3* sunToDir) const
{
    if (!m_RayDebugReadback)
    {
        std::cerr << "[BakedGI RayDebug] no readback buffer — run RecordRayDebug first" << std::endl;
        return;
    }

    uint8_t* mapped = nullptr;
    if (FAILED(m_RayDebugReadback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
    {
        std::cerr << "[BakedGI RayDebug] failed to map the readback buffer" << std::endl;
        return;
    }

    const uint8_t* tableReacc  = mapped + (UINT64)kRayDebugTableBase * 16u;
    const uint8_t* tableStored = mapped + kRayDebugStoredOff;

    std::cout << "[BakedGI RayDebug] probe " << m_RayDebugProbe
              << " (grid " << m_DimX << "x" << m_DimY << "x" << m_DimZ
              << " @ " << m_Spacing << "m)" << std::endl;
    // The two counts differ on purpose, so print both. Confusing them is how a
    // reader would size a ranking against the wrong set.
    std::cout << "    " << kBakeGatherRays << " gather rays for each direction were replayed ("
              << kBakeGatherRays * kDirections << " rays); the " << kDebugGatherRays
              << " RECORD slots of each direction hold the rays that carry a direct term first"
              << std::endl;

    // --- Validity gate. Read the probe's meta bit up front, so the reader has
    // the verdict before step 1. Do NOT infer invalidity from an all-zero
    // stored row. That inference is false in two cases:
    //   1. METAL. A fully metallic surface has diffAlbedo = albedo *
    //      (1 - metallic) = 0 (BakedGI_Bake.hlsl:241), so a probe in a closed
    //      cavity lined with metal stores an exactly zero row. That probe is
    //      still VALID under the DDGI backface rule. The re-trace reproduces
    //      the zero, so the row prints MATCH. A row test would stamp "invalid"
    //      on a MATCH, which is a false claim.
    //   2. SEALED VOLUME. A valid probe inside a sealed volume stores zero too,
    //      because a sky miss contributes nothing under a delta environment.
    // The bake zeroes EVERY stored value of a genuinely invalid probe
    // (BakedGI_Bake.hlsl:147-155), so for such a probe the step-1 ranking
    // describes nothing and the step-3 comparison must fail. That failure is
    // not a drifted RNG stream, and a debug tool must not invent a defect. ---
    uint32_t   probeMeta  = 0;
    const ProbeMetaStatus metaStatus = ReadProbeMeta(m_RayDebugProbe, probeMeta);
    const bool haveMeta   = (metaStatus == ProbeMetaStatus::Ok);
    const bool probeValid = haveMeta && (probeMeta & 0x1u) != 0u; // BAKED_GI_META_VALID
    // Complete the verdict for the Ray Fan through GetRayDebugKnownInvalid.
    // RecordRayDebug already cleared the pairing for this dump, so writing both
    // here is what makes the verdict exist. An early return above (no buffer, or
    // a failed map on the debug readback) leaves the pairing cleared, and the
    // getter then reports no verdict.
    // Unknown validity means "do NOT dim". Dimming states that the probe is
    // degenerate, and that claim needs the meta bit. The text readout assumes a
    // valid probe for this same case, so the picture agrees with the text.
    m_RayDebugValid      = !haveMeta || probeValid;
    m_RayDebugValidProbe = m_RayDebugProbe;
    if (haveMeta && !probeValid)
    {
        std::cout << "[BakedGI RayDebug] WARNING — probe " << m_RayDebugProbe
                  << " is INVALID under the DDGI criteria (meta 0x" << std::hex << probeMeta
                  << std::dec << ")." << std::endl;
        std::cout << "    The bake zeroed this probe's stored table, so step 1 and step 3 describe"
                  << " nothing." << std::endl;
        std::cout << "    The re-traced rays themselves are live, so step 2 and the Ray Fan view"
                  << " still show the geometry." << std::endl;
    }

    // --- Step 1: rank the 16 directions by the luminance of the DC term. ---
    std::vector<std::pair<float, uint32_t>> dirRank;
    dirRank.reserve(kDirections);
    for (uint32_t h = 0; h < kDirections; ++h)
    {
        const float* dc = RayDebugTable(tableStored, h, 0u);
        dirRank.emplace_back(RayDebugLuminance(dc[0], dc[1], dc[2]), h);
    }
    std::sort(dirRank.begin(), dirRank.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "[BakedGI RayDebug] step 1 — 16 directions, stored DC luminance (high to low):" << std::endl;
    for (size_t i = 0; i < dirRank.size(); ++i)
    {
        const uint32_t h = dirRank[i].second;
        const float* v = RayDebugTable(tableStored, h, 9u); // slot 9 = V_i
        std::cout << "    dir " << (h < 10u ? " " : "") << h
                  << "  dcLum " << dirRank[i].first
                  << "  V_i " << v[0] << std::endl;
    }

    // Name the direction the Ray Fan draws in THIS readout, next to the DC
    // ranking. The two picks differ on purpose, see the step-2 comment: step 1
    // ranks by baked DC luminance, and the fan follows the sun fold's argmax.
    // The kernel-weight log names the same direction again, as a second check.
    if (sunToDir)
    {
        float sunWeight = 0.0f;
        const uint32_t sunDominant = DominantSunDirection(*sunToDir, &sunWeight);
        if (sunWeight > 0.0f)
        {
            std::cout << "    the Ray Fan draws direction " << sunDominant
                      << " (sun weight " << sunWeight << ")" << std::endl;
        }
        else
        {
            // Weight 0 means no sun, or a sun below the horizon. The fan then
            // falls back to direction 0, so do not present it as a winner.
            std::cout << "    the Ray Fan has no sun. It falls back to direction "
                      << sunDominant << "." << std::endl;
        }
    }

    // --- Step 2: rank the RECORDS of the highest-DC-luminance direction, step
    // 1's winner, by the luminance of L (record component 2). ---
    // This winner is a DC-luminance pick. It is NOT the sun-dominant direction
    // the Ray Fan draws (BakedGI::DominantSunDirection, the sun fold's argmax).
    // The two are different picks on purpose; do not join them.
    //
    // A record is NOT a source ray. The bake traces kBakeGatherRays for each
    // direction and the debug pass keeps kDebugGatherRays of them, qualifiers
    // first (BakedGI_RayDebug.hlsl). So record 0..63 are an ORDERED SUBSET of
    // rays 0..255, and "ray" below is the source index read from rec3.w. A
    // reader who assumes record r is ray r will mis-name every ray.
    const uint32_t dcDir = dirRank.empty() ? 0u : dirRank.front().second;
    std::vector<std::pair<float, uint32_t>> rayRank;
    rayRank.reserve(kDebugGatherRays);
    for (uint32_t r = 0; r < kDebugGatherRays; ++r)
    {
        // The record index is in float4 units, so multiply by 4 for the float
        // offset (the same conversion RayDebugTable applies).
        const float* rec = reinterpret_cast<const float*>(mapped)
                         + ((size_t)dcDir * kDebugGatherRays + r) * kRayRecordF4 * 4u;
        const float* L = rec + 8; // rec2 = (L.rgb, Vhit)
        rayRank.emplace_back(RayDebugLuminance(L[0], L[1], L[2]), r);
    }
    std::sort(rayRank.begin(), rayRank.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "[BakedGI RayDebug] step 2 — highest-DC-luminance direction " << dcDir
              << ", " << kDebugGatherRays << " records by L luminance (high to low), top 16:"
              << std::endl;
    for (size_t i = 0; i < rayRank.size() && i < 16u; ++i)
    {
        const uint32_t r = rayRank[i].second;
        const float* rec = reinterpret_cast<const float*>(mapped)
                         + ((size_t)dcDir * kDebugGatherRays + r) * kRayRecordF4 * 4u;
        const float* gdir = rec + 0;
        const float* alb  = rec + 4;
        const float* L    = rec + 8;
        // rec4 = (hit normal.xyz, instance index), rec5 = (hit world pos.xyz,
        // primitive index). -1 in w means the ray carried no surface.
        const float* idn  = rec + 16;
        const float* idp  = rec + 20;
        std::cout << "    rec " << (r < 10u ? " " : "") << r
                  << "  ray " << RayDebugSourceRay(rec)
                  << "  L " << rayRank[i].first
                  << "  dist " << gdir[3]
                  << "  NdotD " << alb[3]
                  << "  Vhit " << L[3]
                  << "  status " << RayDebugStatus(rec)
                  << "  inst " << (int)idn[3]
                  << "  prim " << (int)idp[3]
                  << "  Lrgb(" << L[0] << ", " << L[1] << ", " << L[2] << ")"
                  << std::endl;
    }

    // --- Step 2b: the hit-surface census. ---
    // Group every HIT of the probe's 16 x 64 re-traced rays by the instance it
    // landed on, and print that group's ray count, mean diffuse albedo, mean hit
    // position and mean hit normal. This is what rec4/rec5 exist for: a record
    // used to carry no surface identity, so naming the mesh behind a coloured ray
    // meant reproducing EvalResponseIndirect offline and inferring the normal from
    // NdotD. It also makes a tinted surface visible as a NUMBER — the instance
    // whose mean albedo is red is the one to go looking for.
    //
    // The count covers every non-miss ray (status >= 0.5), backface rejects
    // included: a reject names a real surface too, and it is often the one being
    // hunted. The albedo mean is valid for a reject as well, because the record
    // fills rec1 BEFORE the backface test.
    // CAUTION: rec1 carries diffAlbedo = albedo * (1 - metallic), so a METAL
    // reads as black here. A black albedo is not a missing surface.
    struct HitGroup
    {
        uint32_t count = 0;
        float    albedo[3] = { 0.0f, 0.0f, 0.0f };
        float    pos[3]    = { 0.0f, 0.0f, 0.0f };
        float    nrm[3]    = { 0.0f, 0.0f, 0.0f };
    };
    std::vector<std::pair<uint32_t, HitGroup>> groups; // (instance index, group)
    for (uint32_t h = 0; h < kDirections; ++h)
    {
        for (uint32_t r = 0; r < kDebugGatherRays; ++r)
        {
            const float* rec = reinterpret_cast<const float*>(mapped)
                             + ((size_t)h * kDebugGatherRays + r) * kRayRecordF4 * 4u;
            if (RayDebugStatus(rec) < 0.5f) // status 0 = sky miss
                continue;
            const float* alb = rec + 4;
            const float* nrm = rec + 16;
            const float* pos = rec + 20;
            const uint32_t inst = (uint32_t)std::max(0.0f, nrm[3]);
            HitGroup* g = nullptr;
            for (auto& e : groups)
            {
                if (e.first == inst) { g = &e.second; break; }
            }
            if (!g)
            {
                groups.emplace_back(inst, HitGroup{});
                g = &groups.back().second;
            }
            ++g->count;
            for (uint32_t c = 0; c < 3u; ++c)
            {
                g->albedo[c] += alb[c];
                g->nrm[c]    += nrm[c];
                g->pos[c]    += pos[c];
            }
        }
    }
    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.second.count > b.second.count; });

    std::cout << "[BakedGI RayDebug] step 2b — hit surfaces, " << kDebugGatherRays
              << " records of " << kBakeGatherRays << " replayed rays for each of "
              << kDirections << " directions, grouped by instance:" << std::endl;
    if (groups.empty())
        std::cout << "    no hit rays — this probe sees sky only" << std::endl;
    for (size_t i = 0; i < groups.size() && i < 12u; ++i)
    {
        const uint32_t inst = groups[i].first;
        const HitGroup& g   = groups[i].second;
        const float    inv  = 1.0f / (float)g.count;
        std::cout << "    inst " << inst
                  << "  rays " << g.count
                  << "  albedo(" << g.albedo[0] * inv << ", " << g.albedo[1] * inv
                                 << ", " << g.albedo[2] * inv << ")"
                  << "  pos(" << g.pos[0] * inv << ", " << g.pos[1] * inv << ", " << g.pos[2] * inv << ")"
                  << "  normal(" << g.nrm[0] * inv << ", " << g.nrm[1] * inv << ", " << g.nrm[2] * inv << ")"
                  << std::endl;
    }
    if (groups.size() > 12u)
        std::cout << "    ... " << (groups.size() - 12u) << " more instance(s)" << std::endl;

    // --- Step 2c: the Cage Trace pool — the CPU side of the ray picker (view
    // 9). One entry for each shaded ray whose feedback term beats its direct term
    // at the hit ("direct < Efb"), for EVERY direction. ---
    //
    // The view's direction is the sun fold's argmax, so building all 16 is what
    // stops a sun change after the dump from leaving the combo on another
    // direction's rays. The cost is 16 x 64 records, read once per dump.
    //
    // The score is luminance(diffAlbedo * Efb) — the shader's own key. The bake's
    // band-0 coefficient is one constant times the ray's radiance (all 64 rays
    // share the solid angle 4*PI/64), so ranking by carried radiance IS ranking by
    // contribution to the stored response; diffAlbedo stays INSIDE the product,
    // because a ray onto a dark or blue surface carries a large Efb and
    // contributes almost nothing.
    {
        for (uint32_t h = 0; h < kDirections; ++h)
        {
            m_RayCandidates[h].clear();
            for (uint32_t r = 0; r < kDebugGatherRays; ++r)
            {
                const float* rec = reinterpret_cast<const float*>(mapped)
                                 + ((size_t)h * kDebugGatherRays + r) * kRayRecordF4 * 4u;
                // rec3.w packs status + 4 * sourceRayIndex; decode the status.
                // The shader's test is |status - 2| <= 0.5.
                const float status = RayDebugStatus(rec);
                if (status < 1.5f || status > 2.5f)
                    continue; // a sky miss or a backface reject carries no L and no Efb

                const float* alb  = rec + 4;  // (diffAlbedo.rgb, NdotD)
                const float* L    = rec + 8;  // (L.rgb, Vhit)
                const float* efb  = rec + 12; // (Efb.rgb, status)
                const float  ndotd = alb[3] * L[3]; // NdotD * Vhit, the direct term's weight

                RayCandidate c;
                c.ray       = r;
                c.sourceRay = RayDebugSourceRay(rec);
                c.albedoLum = RayDebugLuminance(alb[0], alb[1], alb[2]);
                c.directLum = RayDebugLuminance(alb[0] * ndotd, alb[1] * ndotd, alb[2] * ndotd);
                c.efbLum    = RayDebugLuminance(alb[0] * efb[0], alb[1] * efb[1], alb[2] * efb[2]);
                c.score     = c.efbLum; // identical on purpose: the key IS the carried feedback
                if (c.efbLum <= c.directLum)
                    continue; // the direct delta at the hit wins: this ray is not a borrow
                m_RayCandidates[h].push_back(c);
            }
        }
        m_RayCandidateDirs = kDirections;
    }

    if (sunToDir)
    {
        const uint32_t sunDominant = DominantSunDirection(*sunToDir);
        // Sorted HERE only, on a copy: the stored list keeps ray order, so the
        // UI combo's rows do not re-sort under the user's cursor.
        std::vector<RayCandidate> rows = m_RayCandidates[sunDominant];
        std::sort(rows.begin(), rows.end(),
                  [](const RayCandidate& a, const RayCandidate& b) { return a.score > b.score; });

        std::cout << "[BakedGI RayDebug] step 2c — Cage Trace pool, direction " << sunDominant
                  << ": " << rows.size() << " of " << kDebugGatherRays
                  << " records carry Efb > NdotD*Vhit (the picker's list, by score):" << std::endl;
        for (const RayCandidate& c : rows)
            std::cout << "    rec " << (c.ray < 10u ? " " : "") << c.ray
                      << "  ray " << c.sourceRay
                      << "  score " << c.score
                      << "  efb " << c.efbLum
                      << "  direct " << c.directLum
                      << "  alb " << c.albedoLum << std::endl;
        if (rows.empty())
            std::cout << "    none — the trace draws nothing in this direction" << std::endl;
    }

    // --- Step 3: the self-check. Compare the re-accumulated table with the
    // stored table, element by element. ---
    float maxAbs = 0.0f, maxStored = 0.0f;
    uint32_t worstDir = 0, worstLm = 0;
    for (uint32_t h = 0; h < kDirections; ++h)
    {
        for (uint32_t lm = 0; lm < kResponseFloat4s; ++lm)
        {
            const float* a = RayDebugTable(tableReacc,  h, lm);
            const float* b = RayDebugTable(tableStored, h, lm);
            for (uint32_t c = 0; c < 3u; ++c) // xyz carry the data; w is always zero
            {
                const float d = std::fabs(a[c] - b[c]);
                if (d > maxAbs) { maxAbs = d; worstDir = h; worstLm = lm; }
                maxStored = std::max(maxStored, std::fabs(b[c]));
            }
        }
    }
    const float tol = 1e-4f * std::max(1.0f, maxStored);
    const bool  pass = (maxAbs <= tol);

    std::cout << "[BakedGI RayDebug] step 3 — self-check: re-accumulated table vs stored table"
              << std::endl;
    std::cout << "    max |diff| " << maxAbs << "  (worst at dir " << worstDir
              << ", lm " << worstLm << ")   max |stored| " << maxStored
              << "   tol " << tol << std::endl;
    if (haveMeta && !probeValid)
        std::cout << "    NOT A VERDICT — the probe is invalid, so its stored table is all zero."
                  << " The mismatch above is expected." << std::endl;
    else
    {
        std::cout << "    " << (pass ? "MATCH — the re-trace reproduced the bake"
                                     : "MISMATCH — the RNG stream drifted; read the rays with care")
                  << std::endl;
        if (!haveMeta)
        {
            // Name the cause. The two causes need different fixes, so one
            // shared "no meta readback" line would name the wrong reason after
            // a rebuild with a smaller grid.
            if (metaStatus == ProbeMetaStatus::NoReadback)
                std::cout << "    note: no meta readback buffer (the bake has not run), so the verdict"
                             " assumes a valid probe" << std::endl;
            else if (metaStatus == ProbeMetaStatus::ProbeOutOfRange)
                std::cout << "    note: probe " << m_RayDebugProbe << " is at or past the probe count "
                          << m_ProbeCount << " (the grid changed), so the verdict assumes a valid probe"
                          << std::endl;
            else
                std::cout << "    note: the meta readback could not be mapped, so the verdict assumes"
                             " a valid probe" << std::endl;
        }
    }

    m_RayDebugReadback->Unmap(0, nullptr);
}

// ONE CPU mirror of BakedGIDeltaDirection (BakedGI.hlsli:27-33):
//
//     float y   = ((float)i + 0.5f) / (float)BAKED_GI_DIRECTIONS;
//     float phi = (float)i * 2.39996323f;      // golden angle
//     float r   = sqrt(max(0.0f, 1.0f - y * y));
//     return normalize(float3(r * cos(phi), y, r * sin(phi)));
//
// The SHADER is the authority. A change there MUST change this function. No
// compile-time check can catch that, because the formula is runtime float math,
// and a static_assert cannot compare it. What holds instead: ComputeSunKernelWeights
// and the sun-bin panel's angle column both CALL this function, so the weights
// and the angles cannot disagree with each other even if both drift from the
// shader. The guard on a drift from the shader is the known-good reference below.
//
// KNOWN-GOOD REFERENCE, from the captured FrameCB with sunToDir = (0, 0.906106, 0.42305):
//   w[14] = 0.451283 at 14.66 deg, w[11] = 0.334494 at 21.26 deg,
//   weight-weighted mean angle 23.03 deg, 54.87 percent of the weight beyond 15 deg.
// If the panel no longer reproduces those numbers for that sun, the mirror has
// drifted from the shader and every weight in the engine is suspect.
DirectX::XMFLOAT3 BakedGI::DeltaDirectionCPU(uint32_t j)
{
    const float y   = ((float)j + 0.5f) / (float)kDirections;
    const float phi = (float)j * 2.39996323f;
    const float r   = std::sqrt(std::max(0.0f, 1.0f - y * y));
    float dx = r * std::cos(phi), dy = y, dz = r * std::sin(phi);
    const float l = std::sqrt(dx * dx + dy * dy + dz * dz);
    return DirectX::XMFLOAT3(dx / l, dy / l, dz / l);
}

// CPU mirror of the update pass's sun fold: bw[j] = pow(max(0, dot(dir_j,
// sunDir)), 8), normalised by the caller (BakedGI_Update.hlsl:73-78). The
// direction set comes from DeltaDirectionCPU above.
//
// ONE mirror, and the shader is the authority. ONE caller: SunFoldWeights adds
// the normalisation and the argmax on top, and every other reader goes through
// SunFoldWeights. A second copy of the kernel — or of the normalisation, or of
// the argmax — would drift, and the weight log, the Ray Fan and the Sun
// Directions view would then describe three different folds. That is the same
// one-value-two-places defect that produced the stale validity flag, the
// probe/fan divergence and the frame-order fault.
bool BakedGI::ComputeSunKernelWeights(const DirectX::XMFLOAT3& sunToDir,
                                      float (&bw)[kDirections], float& wsum)
{
    wsum = 0.0f;
    const DirectX::XMFLOAT3 s = sunToDir;
    const float len = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
    if (len < 1e-6f)
        return false; // zero sun direction: the kernel has no meaning
    const float sx = s.x / len, sy = s.y / len, sz = s.z / len;

    for (uint32_t j = 0; j < kDirections; ++j)
    {
        const DirectX::XMFLOAT3 dir = DeltaDirectionCPU(j);
        const float d = dir.x * sx + dir.y * sy + dir.z * sz;
        bw[j] = std::pow(std::max(0.0f, d), 8.0f);
        wsum += bw[j];
    }
    return true;
}

uint32_t BakedGI::SunFoldWeights(const DirectX::XMFLOAT3& sunToDir,
                                 float (&w)[kDirections], float* outMax)
{
    if (outMax)
        *outMax = 0.0f;
    // Clear first: ComputeSunKernelWeights returns false BEFORE it fills bw, so
    // without this a no-sun frame would carry the previous frame's weights into
    // the caller's array — and the debug view would paint a stale fold.
    for (uint32_t j = 0; j < kDirections; ++j)
        w[j] = 0.0f;

    float bw[kDirections];
    float wsum = 0.0f;
    if (!ComputeSunKernelWeights(sunToDir, bw, wsum))
        return 0u; // zero sun direction: no fold, so no sun-dominant direction

    // Sun below the horizon: every weight is 0, so every direction ties. Report
    // direction 0 with weight 0.0. A caller must print "no sun" then, and must
    // not present direction 0 as a winner.
    if (wsum <= 1e-6f)
        return 0u;

    for (uint32_t j = 0; j < kDirections; ++j)
        w[j] = bw[j] / wsum;

    // The strict compare keeps the LOWEST index on a tie. A tie is rare above
    // the horizon, and a stable pick keeps the fan still across frames. The Sun
    // Directions view ramps on the weights, so a tie paints more than one
    // direction green: that is the weights being reported, not a conflict with
    // this single index.
    uint32_t best = 0u;
    for (uint32_t j = 1; j < kDirections; ++j)
        if (w[j] > w[best])
            best = j;
    if (outMax)
        *outMax = w[best];
    return best;
}

uint32_t BakedGI::DominantSunDirection(const DirectX::XMFLOAT3& sunToDir, float* outWeight)
{
    // The weight array is discarded: the argmax rule lives in SunFoldWeights, so
    // the fan's direction and the debug view's green direction are one pick.
    float w[kDirections];
    return SunFoldWeights(sunToDir, w, outWeight);
}

uint32_t BakedGI::FindNearestValidProbes(const DirectX::XMFLOAT3& point, uint32_t k,
                                         std::vector<NearestProbe>& out) const
{
    out.clear();
    if (k == 0u || !m_Valid || m_ProbeCount == 0u || m_DimX == 0u || m_DimY == 0u)
        return 0u;
    // The meta readback is the validity source. Without it (the bake has not
    // run) there is no verified probe to report, and an unverified probe must
    // not be offered as valid.
    if (!m_MetaReadback)
        return 0u;

    // The grid is regular, so the cell of a probe index is a pure function of
    // the index. ONE definition, used by both the scan and the output pass, so
    // the two cannot disagree about which cell a row describes.
    const uint32_t layerSize = m_DimX * m_DimY; // probes in one z slice
    const auto cellOf = [this, layerSize](uint32_t p, uint32_t& cx, uint32_t& cy, uint32_t& cz)
    {
        cx = p % m_DimX;
        cy = (p / m_DimX) % m_DimY;
        cz = p / layerSize;
    };

    uint32_t* meta = nullptr;
    if (FAILED(m_MetaReadback->Map(0, nullptr, reinterpret_cast<void**>(&meta))))
        return 0u;

    // Keep the k smallest distances WITHOUT sorting the grid. A bounded max-heap
    // keyed on SQUARED distance: the front is the farthest survivor, so a closer
    // probe replaces it. k is a UI count (a handful); the grid is up to 128K.
    // Comparing d2 avoids a sqrt on every probe — the sqrt happens once per
    // surviving row, in the output pass below.
    std::vector<std::pair<float, uint32_t>> heap; // (distance^2, probe index)
    heap.reserve(k + 1u);

    for (uint32_t p = 0; p < m_ProbeCount; ++p)
    {
        if ((meta[p] & 0x1u) == 0u) // BAKED_GI_META_VALID
            continue;

        uint32_t cx = 0, cy = 0, cz = 0;
        cellOf(p, cx, cy, cz);
        const float dx = m_GridMin.x + (float)cx * m_Spacing - point.x;
        const float dy = m_GridMin.y + (float)cy * m_Spacing - point.y;
        const float dz = m_GridMin.z + (float)cz * m_Spacing - point.z;
        const float d2 = dx * dx + dy * dy + dz * dz;

        if (heap.size() < k)
        {
            heap.emplace_back(d2, p);
            std::push_heap(heap.begin(), heap.end());
        }
        else if (d2 < heap.front().first)
        {
            std::pop_heap(heap.begin(), heap.end()); // farthest lands at the back
            heap.back() = { d2, p };
            std::push_heap(heap.begin(), heap.end());
        }
    }

    m_MetaReadback->Unmap(0, nullptr);

    // Heap order is not distance order. sort_heap turns the max-heap into an
    // ascending run, which is the order the list prints.
    std::sort_heap(heap.begin(), heap.end());

    out.reserve(heap.size());
    for (const std::pair<float, uint32_t>& e : heap)
    {
        NearestProbe np;
        np.index  = e.second;
        cellOf(np.index, np.cellX, np.cellY, np.cellZ);
        np.center = { m_GridMin.x + (float)np.cellX * m_Spacing,
                      m_GridMin.y + (float)np.cellY * m_Spacing,
                      m_GridMin.z + (float)np.cellZ * m_Spacing };
        np.distance = std::sqrt(e.first);
        out.push_back(np);
    }
    return (uint32_t)out.size();
}

void BakedGI::LogKernelWeights(const DirectX::XMFLOAT3& sunToDir)
{
    float w[kDirections];
    float wMax = 0.0f;
    const uint32_t sunDominant = SunFoldWeights(sunToDir, w, &wMax);

    std::vector<std::pair<float, uint32_t>> rank;
    rank.reserve(kDirections);
    for (uint32_t j = 0; j < kDirections; ++j)
        rank.emplace_back(w[j], j);
    std::sort(rank.begin(), rank.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "[BakedGI RayDebug] kernel weights — the fold's cos^8 weights, normalised"
              << std::endl;
    if (wMax <= 0.0f)
    {
        // SunFoldWeights reports one all-zero array for both no-sun causes, so
        // name both instead of guessing which one happened.
        std::cout << "    no sun: every weight is zero (either the light direction is zero, or"
                     " the sun is below the horizon)" << std::endl;
        return;
    }
    // Mark the direction the fan and the DeltaDirections channel both single
    // out, so the reader does not have to find the maximum in the weight column
    // by eye.
    for (size_t i = 0; i < rank.size(); ++i)
        std::cout << "    dir " << (rank[i].second < 10u ? " " : "") << rank[i].second
                  << "  weight " << rank[i].first
                  << (rank[i].second == sunDominant
                          ? "   <- the Ray Fan draws this direction, and the DeltaDirections"
                            " channel paints it white"
                          : "")
                  << std::endl;
}

// -----------------------------------------------------------------------------
// Sun-bin panel — the CPU side
// -----------------------------------------------------------------------------

void BakedGI::BuildSunBinPanel(const DirectX::XMFLOAT3* sunToDir, uint32_t probe,
                               SunBinPanel& out) const
{
    out = SunBinPanel{};

    // --- Weights. The SAME call the update pass's mirror and the Ray Fan use. ---
    uint32_t weightArgmax = 0u;
    float    wMax = 0.0f;
    if (sunToDir)
        weightArgmax = SunFoldWeights(*sunToDir, out.w, &wMax);
    out.weightArgmax  = weightArgmax;
    out.weightArgmaxW = wMax;
    out.haveSun       = (sunToDir != nullptr) && (wMax > 0.0f);

    // --- Sun orientation, for the header. Elevation above the horizontal
    // plane, azimuth with the direction set's own convention, atan2(z, x). ---
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    if (sunToDir)
    {
        const float len = std::sqrt(sunToDir->x * sunToDir->x + sunToDir->y * sunToDir->y
                                    + sunToDir->z * sunToDir->z);
        if (len > 1e-6f)
        {
            sx = sunToDir->x / len;
            sy = sunToDir->y / len;
            sz = sunToDir->z / len;
            out.haveDirection = true;
            out.elevationDeg  = DirectX::XMConvertToDegrees(std::asin(std::max(-1.0f, std::min(1.0f, sy))));
            out.azimuthDeg    = DirectX::XMConvertToDegrees(std::atan2(sz, sx));
        }
    }

    // --- Angles and the cone statistics. The angles come from the SAME mirror
    // the weights come from, so the two columns cannot disagree even if both
    // drift from the shader. ---
    float meanAngle = 0.0f;
    for (uint32_t j = 0; j < kDirections; ++j)
    {
        const DirectX::XMFLOAT3 dir = DeltaDirectionCPU(j);
        float angleDeg = 0.0f;
        if (out.haveDirection)
        {
            const float d = dir.x * sx + dir.y * sy + dir.z * sz;
            angleDeg = DirectX::XMConvertToDegrees(std::acos(std::max(-1.0f, std::min(1.0f, d))));
        }

        SunBinRow& row = out.rows[j];
        row.bin      = j;
        row.weight   = out.w[j];
        row.angleDeg = angleDeg;

        meanAngle += out.w[j] * angleDeg;
        if (out.w[j] > 1e-6f)
            ++out.nonzeroBins;
        if (angleDeg > 15.0f)
            out.weightBeyond15 += out.w[j];
    }
    // The weights sum to 1 whenever haveSun, so this is a true weighted mean.
    // With no sun they are all zero and the mean is 0, which the header omits.
    out.meanAngleDeg = meanAngle;

    // --- Per-probe contribution column. Empty unless the read-back belongs to
    // THIS probe: a stale read-back must never be labelled as this one. ---
    out.responseProbe = m_ProbeTableProbe;
    out.haveResponse  = m_ProbeTableValid && (m_ProbeTableProbe == probe);

    if (out.haveResponse)
    {
        // The DC-luminance argmax uses the same weights the ray-debug readout's
        // step 1 uses, so the panel and the log rank the same way. It is NOT the
        // weight argmax; the two markers are deliberately separate.
        const DirectX::XMFLOAT3 kLum(0.2126f, 0.7152f, 0.0722f);
        float dcBest = -1.0f;
        float total  = 0.0f;
        for (uint32_t j = 0; j < kDirections; ++j)
        {
            // Slot 0 is the DC SH coefficient of the response to a unit delta
            // from W_j. Slots 1..8 are the directional bands and are not shown.
            const DirectX::XMFLOAT4& r = m_ProbeTableCPU[j * kResponseFloat4s + 0u];
            SunBinRow& row = out.rows[j];
            row.dcLum = kLum.x * r.x + kLum.y * r.y + kLum.z * r.z;
            row.contrib    = row.weight * row.dcLum;
            row.contribRGB = DirectX::XMFLOAT3(row.weight * r.x, row.weight * r.y, row.weight * r.z);
            total += row.contrib;
            if (row.dcLum > dcBest)
            {
                dcBest = row.dcLum;
                out.dcArgmax = j;
            }
        }
        out.contribTotal = total;
        for (uint32_t j = 0; j < kDirections; ++j)
            out.rows[j].contribPct = (total > 1e-12f) ? (out.rows[j].contrib / total) : 0.0f;
    }

    // --- Sort. Contribution descending is the useful order when the per-probe
    // column exists, because that is the question the panel answers. Without it,
    // fall back to weight descending and the UI says so. The bin index survives
    // in SunBinRow::bin, so a row is never ambiguous. ---
    std::sort(out.rows, out.rows + kDirections, [](const SunBinRow& a, const SunBinRow& b) {
        return (a.contrib != b.contrib) ? (a.contrib > b.contrib) : (a.weight > b.weight);
    });
}

bool BakedGI::RecordProbeTableReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                                       uint32_t probe)
{
    m_ProbeTableValid = false; // a failed read-back must not leave the old one labelled as new

    if (!device || !cmdList || !m_Valid || !m_Response[m_FinalResponse].resource)
        return false;

    if (probe >= m_ProbeCount)
        probe = m_ProbeCount - 1u;
    m_ProbeTableProbe = probe;

    // Same lifetime rule as the ray-debug read-back: the caller synced the GPU,
    // so releasing the previous one here is safe.
    m_ProbeTableReadback.Reset();

    const UINT64 bytes = (UINT64)kDirections * kResponseFloat4s * 16u;
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(bytes);
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&m_ProbeTableReadback))))
        {
            std::cerr << "[BakedGI] Failed to create the probe-table readback buffer" << std::endl;
            return false;
        }
        GraphicsHelper::SetObjectName(m_ProbeTableReadback.Get(), "RB_BakedGIProbeTable");
    }

    // The response table sits in UNORDERED_ACCESS between bakes, which is the
    // state RecordRayDebug also finds it in. Transition out and back so this
    // call leaves the buffer exactly as it found it.
    ID3D12Resource* src = m_Response[m_FinalResponse].resource.Get();
    const UINT64 srcOffset = (UINT64)probe * kDirections * kResponseFloat4s * 16u;
    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
            src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b);
        cmdList->CopyBufferRegion(m_ProbeTableReadback.Get(), 0, src, srcOffset, bytes);
        D3D12_RESOURCE_BARRIER b2 = CD3DX12_RESOURCE_BARRIER::Transition(
            src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);
    }
    return true;
}

void BakedGI::ConsumeProbeTableReadback()
{
    if (!m_ProbeTableReadback)
        return;

    void* mapped = nullptr;
    if (FAILED(m_ProbeTableReadback->Map(0, nullptr, &mapped)) || !mapped)
    {
        std::cerr << "[BakedGI] Failed to map the probe-table readback buffer" << std::endl;
        return;
    }
    std::memcpy(m_ProbeTableCPU, mapped, (size_t)kDirections * kResponseFloat4s * 16u);
    m_ProbeTableReadback->Unmap(0, nullptr);

    m_ProbeTableValid = true;
}

bool BakedGI::HasSunResponseFor(const DirectX::XMFLOAT3& sunToDir) const
{
    if (!m_SunResponseValid)
        return false;

    // Normalised dot, so the test is about DIRECTION and not magnitude. 1e-6 in a
    // unit dot is about 0.08 degrees: far tighter than any real sun move, and far
    // looser than float renormalisation noise, so a genuine move cannot slip past.
    const float la = std::sqrt(m_SunBakeDir.x * m_SunBakeDir.x + m_SunBakeDir.y * m_SunBakeDir.y
                               + m_SunBakeDir.z * m_SunBakeDir.z);
    const float lb = std::sqrt(sunToDir.x * sunToDir.x + sunToDir.y * sunToDir.y
                               + sunToDir.z * sunToDir.z);
    if (la < 1e-6f || lb < 1e-6f)
        return false; // a zero direction describes nothing

    const float d = (m_SunBakeDir.x * sunToDir.x + m_SunBakeDir.y * sunToDir.y
                     + m_SunBakeDir.z * sunToDir.z) / (la * lb);
    return d >= 1.0f - 1e-6f;
}

// -----------------------------------------------------------------------------
// Probes/ disk cache
// -----------------------------------------------------------------------------

bool BakedGI::CreateProbeBuffers(bool wantSunTable)
{
    // The caller synced the GPU before recording, so releasing the previous
    // bake's buffers here is safe (same pattern as
    // Renderer::BuildAccelerationStructures).
    m_Response[0]   = {};
    m_Response[1]   = {};
    m_SunResponse[0] = {};
    m_SunResponse[1] = {};
    m_LitProbes     = {};
    m_LitProbesSky  = {};
    m_LitProbesSun  = {};
    m_Meta          = {};
    m_Visibility    = {};

    const uint64_t responseElems = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s;
    // Method-3 sun table: ONE record per probe, not 16. 160 B per probe per
    // buffer. Created only when the method asks for it, so the Probe method's
    // VRAM is unchanged.
    const uint64_t sunElems = (uint64_t)m_ProbeCount * kResponseFloat4s;
    if (!CreateStructuredBuffer(m_Response[0], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse0") ||
        !CreateStructuredBuffer(m_Response[1], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse1") ||
        (wantSunTable &&
         (!CreateStructuredBuffer(m_SunResponse[0], sizeof(float) * 4, sunElems,
                                  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGISun0") ||
          !CreateStructuredBuffer(m_SunResponse[1], sizeof(float) * 4, sunElems,
                                  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGISun1"))) ||
        !CreateStructuredBuffer(m_LitProbes, sizeof(float) * 4, (uint64_t)m_ProbeCount * kLitFloat4s,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGILit") ||
        !CreateStructuredBuffer(m_LitProbesSky, sizeof(float) * 4, (uint64_t)m_ProbeCount * kLitFloat4s,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGILitSky") ||
        !CreateStructuredBuffer(m_LitProbesSun, sizeof(float) * 4, (uint64_t)m_ProbeCount * kLitFloat4s,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGILitSun") ||
        !CreateStructuredBuffer(m_Meta, sizeof(uint32_t), m_ProbeCount,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIMeta") ||
        !CreateStructuredBuffer(m_Visibility, sizeof(float) * 2, (uint64_t)m_ProbeCount * kVisStride,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIVisibility"))
    {
        std::cerr << "[BakedGI] Failed to create probe buffers" << std::endl;
        return false;
    }
    return true;
}

// Debug-only capture, created on the dispatch path only: the term snapshots and
// the folded per-probe response balls that the "Efb Contribution" view reads.
//
// SIZING. One snapshot is a whole transport table: probeCount x 16 x 10 float4,
// 2560 B for each probe (about 30 MB at 11,616 probes, 56 MB at 21,952). The
// kIterSnapshotCount = kBakeIterations - 1 snapshots plus kBakeIterations balls
// (144 B for each probe — the ball is a per-probe SH9) is the whole cost, and it
// is what the view costs when it is never opened. It is paid on a bake, which is
// a user action, not on the frame path.
//
// The GPUBuffer structs are deliberately NOT reset first. CreateStructuredBuffer
// reuses an existing descriptor slot when it has one, so a re-bake replaces the
// resource without consuming a fresh SRV/UAV pair from a heap that has no free —
// the pattern the ray-debug buffer documents in RecordRayDebug.
bool BakedGI::CreateIterationBuffers()
{
    const uint64_t responseElems = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s;

    for (uint32_t s = 0; s < kIterSnapshotCount; ++s)
    {
        char name[48];
        snprintf(name, sizeof(name), "SB_BakedGIResponseIter%u", s);
        // COPY_DEST: these buffers only ever receive a CopyBufferRegion during
        // the bake, and the fold pass transitions them for its SRV read.
        if (!CreateStructuredBuffer(m_ResponseIter[s], sizeof(float) * 4, responseElems,
                                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, name))
        {
            std::cerr << "[BakedGI] Failed to create the term snapshot " << s << std::endl;
            return false;
        }
    }

    for (uint32_t t = 0; t < kBakeIterations; ++t)
    {
        char name[48];
        snprintf(name, sizeof(name), "SB_BakedGIEfbBall%u", t);
        if (!CreateStructuredBuffer(m_EfbBall[t], sizeof(float) * 4, (uint64_t)m_ProbeCount * kEfbBallFloat4s,
                                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, name))
        {
            std::cerr << "[BakedGI] Failed to create the response ball " << t << std::endl;
            return false;
        }
    }
    return true;
}

void BakedGI::SnapshotResponseTable(ID3D12GraphicsCommandList* cmdList, uint32_t src, uint32_t snapshot)
{
    GPUBuffer& source = m_Response[src];
    GPUBuffer& dest   = m_ResponseIter[snapshot];
    const UINT64 bytes = (UINT64)m_ProbeCount * kDirections * kResponseFloat4s * 16u;

    // The source is in UNORDERED_ACCESS (the bake dispatch wrote it), the
    // destination in COPY_DEST (CreationIterationBuffers). Round-trip the source
    // so the next transport iteration still finds the state it expects — the same
    // record-copy-record shape as the ray-debug dump's stored-table copy.
    source.Transition(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->CopyBufferRegion(dest.resource.Get(), 0, source.resource.Get(), 0, bytes);
    source.Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void BakedGI::TransitionEfbBallsForRead(ID3D12GraphicsCommandList* cmdList)
{
    if (!cmdList)
        return;
    // See TransitionRayDebugForRead for why this sits at the read site and not in
    // the bake: every bake builds NEW resources (CreateIterationBuffers calls
    // CreateCommittedResource), so the state tracker cannot go stale between
    // bakes, and GPUResource::Transition makes the call free after the first.
    for (uint32_t t = 0; t < kBakeIterations; ++t)
        m_EfbBall[t].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // The view's single-direction mode reads the TRANSPORT TABLES, not the balls:
    // one direction cannot be recovered from a 16-direction sum. The bake leaves
    // them tracked as UNORDERED_ACCESS and reads them as SRVs behind a UAV barrier
    // alone — the convention RecordBake documents at :373-374, which is fine for a
    // compute read on the bake's own list. This read happens on a LATER FRAME and
    // from the VERTEX stage, so the read site takes the real barrier rather than
    // inheriting that assumption. Transition guards the change, so this costs one
    // barrier per bake, not one per frame.
    for (uint32_t s = 0; s < kIterSnapshotCount; ++s)
        m_ResponseIter[s].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_Response[m_FinalResponse].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

bool BakedGI::TryLoadCache(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, const std::string& scenePath)
{
    namespace fs = std::filesystem;
    if (scenePath.empty())
        return false;

    const fs::path cachePath = ProbeCachePath(scenePath, m_Spacing);

    // Hit = the file exists + the header validates against the grid this
    // run computed (the map stem + spacing are already in the path). No
    // mtime freshness during development — delete the cache or bump the
    // version after editing the bake.

    // Read + validate the header against the grid RecordBake just computed.
    std::ifstream in(cachePath, std::ios::binary);
    if (!in.is_open())
        return false;
    ProbeCacheHeader h = {};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (h.magic != kProbeCacheMagic || h.version != kProbeCacheVersion ||
        h.dimX != m_DimX || h.dimY != m_DimY || h.dimZ != m_DimZ ||
        h.probeCount != m_ProbeCount || h.finalResponse > 1 ||
        std::fabs(h.spacing - m_Spacing) > 1e-6f ||
        std::fabs(h.gridMinX - m_GridMin.x) > 1e-4f ||
        std::fabs(h.gridMinY - m_GridMin.y) > 1e-4f ||
        std::fabs(h.gridMinZ - m_GridMin.z) > 1e-4f)
    {
        return false;
    }

    const uint64_t responseBytes = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s * 16;
    const uint64_t metaBytes     = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    const uint64_t visBytes      = (uint64_t)m_ProbeCount * kVisStride * sizeof(float) * 2;
    const uint64_t total         = responseBytes + metaBytes + visBytes;

    std::vector<char> blob(static_cast<size_t>(total));
    in.read(blob.data(), static_cast<std::streamsize>(total));
    if (in.fail() || in.gcount() != static_cast<std::streamsize>(total))
    {
        std::cerr << "[BakedGI] Probe cache truncated: " << cachePath.string() << std::endl;
        return false;
    }

    if (!CreateProbeBuffers())
        return false;
    m_FinalResponse = h.finalResponse;

    // One UPLOAD staging buffer for all three sections — kept alive as a
    // member until the next bake (the recorded list outlives this call).
    D3D12_HEAP_PROPERTIES up = {};
    up.Type = D3D12_HEAP_TYPE_UPLOAD;
    const D3D12_RESOURCE_DESC upDesc = CD3DX12_RESOURCE_DESC::Buffer(total);
    if (FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &upDesc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_CacheUpload))))
    {
        std::cerr << "[BakedGI] Failed to create the cache upload buffer" << std::endl;
        return false;
    }
    GraphicsHelper::SetObjectName(m_CacheUpload.Get(), "UP_BakedGICacheLoad");

    {
        uint8_t* mapped = nullptr;
        if (FAILED(m_CacheUpload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
            return false;
        memcpy(mapped, blob.data(), static_cast<size_t>(total));
        m_CacheUpload->Unmap(0, nullptr);
    }

    struct Copy { GPUBuffer* dst; uint64_t offset; uint64_t bytes; };
    const Copy copies[] = {
        { &m_Response[m_FinalResponse], 0,                          responseBytes },
        { &m_Meta,                      responseBytes,              metaBytes },
        { &m_Visibility,                responseBytes + metaBytes,  visBytes },
    };
    for (const Copy& c : copies)
    {
        c.dst->Transition(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);
        cmdList->CopyBufferRegion(c.dst->resource.Get(), 0, m_CacheUpload.Get(), c.offset, c.bytes);
        c.dst->Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    m_LoadedFromCache = true;
    m_Valid = true;
    std::cout << "[BakedGI] Probe cache hit: " << m_ProbeCount << " probes ("
              << m_DimX << "x" << m_DimY << "x" << m_DimZ << " @ " << m_Spacing
              << "m) — " << cachePath.string() << std::endl;
    return true;
}

bool BakedGI::RecordCacheReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList)
{
    if (!m_Valid || !device || !cmdList)
        return false;

    const uint64_t responseBytes = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s * 16;
    const uint64_t metaBytes     = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    const uint64_t visBytes      = (uint64_t)m_ProbeCount * kVisStride * sizeof(float) * 2;
    const uint64_t total         = responseBytes + metaBytes + visBytes;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(total);
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_CacheReadback))))
    {
        std::cerr << "[BakedGI] Failed to create the cache readback buffer" << std::endl;
        return false;
    }
    GraphicsHelper::SetObjectName(m_CacheReadback.Get(), "RB_BakedGICache");

    // response | meta | vis -> one readback, three copies.
    struct Copy { GPUBuffer* src; uint64_t dstOffset; uint64_t bytes; };
    const Copy copies[] = {
        { &m_Response[m_FinalResponse], 0,                          responseBytes },
        { &m_Meta,                      responseBytes,              metaBytes },
        { &m_Visibility,                responseBytes + metaBytes,  visBytes },
    };
    for (const Copy& c : copies)
    {
        c.src->Transition(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->CopyBufferRegion(m_CacheReadback.Get(), c.dstOffset, c.src->resource.Get(), 0, c.bytes);
        c.src->Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    return true;
}

void BakedGI::WriteCacheFile(const std::string& scenePath)
{
    if (!m_CacheReadback || scenePath.empty())
        return;

    const std::filesystem::path cachePath = ProbeCachePath(scenePath, m_Spacing);
    if (cachePath.empty())
        return;

    ProbeCacheHeader h = {};
    h.magic = kProbeCacheMagic;
    h.version = kProbeCacheVersion;
    h.dimX = m_DimX; h.dimY = m_DimY; h.dimZ = m_DimZ;
    h.probeCount = m_ProbeCount;
    h.finalResponse = m_FinalResponse;
    h.spacing = m_Spacing;
    h.gridMinX = m_GridMin.x; h.gridMinY = m_GridMin.y; h.gridMinZ = m_GridMin.z;

    const uint64_t responseBytes = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s * 16;
    const uint64_t metaBytes     = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    const uint64_t visBytes      = (uint64_t)m_ProbeCount * kVisStride * sizeof(float) * 2;
    const uint64_t total         = responseBytes + metaBytes + visBytes;

    uint8_t* mapped = nullptr;
    if (FAILED(m_CacheReadback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
    {
        std::cerr << "[BakedGI] Failed to map the cache readback buffer" << std::endl;
        return;
    }

    std::ofstream out(cachePath, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "[BakedGI] Failed to open the probe cache for writing: "
                  << cachePath.string() << std::endl;
        m_CacheReadback->Unmap(0, nullptr);
        return;
    }
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(mapped + 0),
              static_cast<std::streamsize>(responseBytes));
    out.write(reinterpret_cast<const char*>(mapped + responseBytes),
              static_cast<std::streamsize>(metaBytes));
    out.write(reinterpret_cast<const char*>(mapped + responseBytes + metaBytes),
              static_cast<std::streamsize>(visBytes));
    m_CacheReadback->Unmap(0, nullptr);

    std::cout << "[BakedGI] Probe cache written: " << cachePath.string()
              << " (" << (total >> 20) << " MB)" << std::endl;
}

void BakedGI::RecordUpdate(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                           D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                           const DirectX::XMFLOAT3& sunIrradiance, const DirectX::XMFLOAT3& sunToDir,
                           bool singleDirectionSun)
{
    if (!m_Valid || !m_UpdatePSO)
        return;

    BakedGIUpdateParams p = {};
    p.sunIrradianceX = sunIrradiance.x;
    p.sunIrradianceY = sunIrradiance.y;
    p.sunIrradianceZ = sunIrradiance.z;
    p.sunDirX = sunToDir.x;
    p.sunDirY = sunToDir.y;
    p.sunDirZ = sunToDir.z;
    // Method 3: the sun column comes from the one baked sun term, not from the
    // 16-bin kernel. The sky column is untouched — it still needs the binned
    // table for its quadrature.
    p.sunMode            = singleDirectionSun ? 1u : 0u;
    p.sunResponseSRVIdx  = singleDirectionSun ? GetSunResponseSRVIndex() : 0u;
    if (singleDirectionSun && p.sunResponseSRVIdx == 0u)
    {
        // No sun table: the fold would read descriptor 0 through the wrong view.
        // Report and leave the lit buffer alone rather than write garbage.
        std::cerr << "[BakedGI] single-direction fold requested with no sun table — skipped"
                  << std::endl;
        return;
    }
    p.responseSRVIdx = (uint32_t)m_Response[m_FinalResponse].srvIndex;
    p.metaSRVIdx      = (uint32_t)m_Meta.srvIndex;
    p.litUAVIdx       = (uint32_t)m_LitProbes.uavIndex;
    p.litSkyUAVIdx    = (uint32_t)m_LitProbesSky.uavIndex;
    p.litSunUAVIdx    = (uint32_t)m_LitProbesSun.uavIndex;
    p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
    p.spacing  = m_Spacing;
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;

    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIUpdateParams) / 4, &p, 0);
    cmdList->SetPipelineState(m_UpdatePSO.Get());
    cmdList->Dispatch((m_ProbeCount + 63) / 64, 1, 1);

    D3D12_RESOURCE_BARRIER barriers[3] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_LitProbes.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_LitProbesSky.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_LitProbesSun.resource.Get()),
    };
    cmdList->ResourceBarrier(3, barriers);
}
