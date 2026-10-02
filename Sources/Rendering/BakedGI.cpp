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

// =============================================================================
// Probes/ disk cache — the ShaderCache pattern applied to the bake outputs.
// =============================================================================
namespace
{

constexpr uint32_t kProbeCacheMagic   = 0x42504744; // 'BGPD'
constexpr uint32_t kProbeCacheVersion = 0; // development: format/semantics not finalized — no
                                           // compatibility across changes; freeze at 1 once stable

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
}

bool BakedGI::RecordBake(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                         ID3D12RootSignature* rootSignature, Model* model,
                         D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                         float spacing)
{
    if (!device || !cmdList || !rootSignature || !model || !m_BakePSO)
        return false;

    // The caller synced the GPU — release the previous bake's cache staging
    // now (the upload buffer must only die between executions, never while a
    // recorded list still references it).
    m_CacheUpload.Reset();
    m_CacheReadback.Reset();
    m_LoadedFromCache = false;

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

    if (m_Spacing != spacing)
        std::cout << "[BakedGI] Spacing auto-grew " << spacing << "m -> " << m_Spacing
                  << "m to fit the probe cap (" << kMaxProbeCount << " probes / "
                  << kMaxDim << " per axis)" << std::endl;

    std::cout << "[BakedGI] Baking probe grid: " << m_DimX << "x" << m_DimY << "x" << m_DimZ
              << " : " << m_Spacing << "m (" << m_ProbeCount << " probes, "
              << kBakeIterations << " series iterations)"
              << " — world " << (mx.x - mn.x) << "x" << (mx.y - mn.y) << "x" << (mx.z - mn.z) << "m" << std::endl;

    // --- Probes/ disk cache: on a hit (file exists + grid header matches),
    // restore the outputs with plain buffer copies and skip the dispatches
    // entirely. ---
    if (TryLoadCache(device, cmdList, model->GetSourcePath()))
        return true;

    if (!CreateProbeBuffers())
        return false;

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

        cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIBakeParams) / 4, &p, 0);
        cmdList->SetPipelineState(m_BakePSO.Get());
        cmdList->Dispatch((m_ProbeCount + 7) / 8, kDirections / 8, 1);

        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Response[cur].resource.Get());
        cmdList->ResourceBarrier(1, &barrier);

        m_FinalResponse = cur;
        cur = 1 - cur;
    }

    m_Valid = true;
    std::cout << "[BakedGI] Bake recorded (" << m_ProbeCount << " probes, "
              << "visibility: 16x16 depth moments x" << kVisStride << " texels/probe)"
              << std::endl;
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
// Probes/ disk cache
// -----------------------------------------------------------------------------

bool BakedGI::CreateProbeBuffers()
{
    // The caller synced the GPU before recording, so releasing the previous
    // bake's buffers here is safe (same pattern as
    // Renderer::BuildAccelerationStructures).
    m_Response[0]   = {};
    m_Response[1]   = {};
    m_LitProbes     = {};
    m_LitProbesSky  = {};
    m_LitProbesSun  = {};
    m_Meta          = {};
    m_Visibility    = {};

    const uint64_t responseElems = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s;
    if (!CreateStructuredBuffer(m_Response[0], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse0") ||
        !CreateStructuredBuffer(m_Response[1], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse1") ||
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
                           const DirectX::XMFLOAT3& sunIrradiance, const DirectX::XMFLOAT3& sunToDir)
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
