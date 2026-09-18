#include "pch.h"

#include "BakedGI.h"
#include "Core/Model.h"
#include "Graphics/GraphicsHelper.h"

#include <DirectXCollision.h>
#include <iostream>

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

    // --- (Re)create buffers. The caller synced the GPU before recording, so
    // releasing the previous bake's buffers here is safe (same pattern as
    // Renderer::BuildAccelerationStructures). ---
    m_Response[0] = {};
    m_Response[1] = {};
    m_LitProbes   = {};
    m_Meta        = {};

    const uint64_t responseElems = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s;
    if (!CreateStructuredBuffer(m_Response[0], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse0") ||
        !CreateStructuredBuffer(m_Response[1], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse1") ||
        !CreateStructuredBuffer(m_LitProbes, sizeof(float) * 4, (uint64_t)m_ProbeCount * kLitFloat4s,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGILit") ||
        !CreateStructuredBuffer(m_Meta, sizeof(uint32_t), m_ProbeCount,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIMeta"))
    {
        std::cerr << "[BakedGI] Failed to create probe buffers" << std::endl;
        return false;
    }

    // --- Record the series-expansion iterations (ping-pong) ---
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, materialsAddress);
    cmdList->SetComputeRootShaderResourceView(2, drawNodesAddress);
    cmdList->SetComputeRootShaderResourceView(3, tlasAddress);
    cmdList->SetComputeRootShaderResourceView(4, indicesAddress);
    cmdList->SetComputeRootShaderResourceView(5, verticesAddress);

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

        cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIBakeParams) / 4, &p, 0);
        cmdList->SetPipelineState(m_BakePSO.Get());
        cmdList->Dispatch((m_ProbeCount + 7) / 8, kDirections / 8, 1);

        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Response[cur].resource.Get());
        cmdList->ResourceBarrier(1, &barrier);

        m_FinalResponse = cur;
        cur = 1 - cur;
    }

    m_Valid = true;
    std::cout << "[BakedGI] Bake recorded (" << m_ProbeCount << " probes)" << std::endl;
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

    uint32_t valid = 0;
    for (uint32_t i = 0; i < m_ProbeCount; ++i)
        valid += (meta[i] & 1u);
    m_MetaReadback->Unmap(0, nullptr);

    const uint32_t invalid = m_ProbeCount - valid;
    const float    pct     = 100.0f * (float)valid / (float)m_ProbeCount;
    std::cout << "[BakedGI] Probe coverage: " << valid << "/" << m_ProbeCount
              << " valid (" << pct << "%) : " << invalid
              << " inside geometry (excluded from interpolation)" << std::endl;
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
    p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
    p.spacing  = m_Spacing;
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;

    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIUpdateParams) / 4, &p, 0);
    cmdList->SetPipelineState(m_UpdatePSO.Get());
    cmdList->Dispatch((m_ProbeCount + 63) / 64, 1, 1);

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_LitProbes.resource.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
