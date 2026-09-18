#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;

// =============================================================================
// BakedGI — probe-based baked GI: uniform probe grid + runtime transport bake.
//
// Exclusive indirect-GI source (either this or ReSTIR GI, never both). The
// bake computes, per probe and per delta direction (16 fibonacci-hemisphere
// dirs), the INDIRECT-ONLY directional-irradiance SH9 response to a unit
// delta light plus the direct-delta visibility V_i (multi-bounce via a
// series-expansion ping-pong with exact direct light at every trace hit —
// the retrace-sun discipline). Per frame, a cheap update pass folds the live
// sky (cubemap quadrature) and sun (direction-kernel interpolation) into a
// lit-probe buffer the deferred pass fetches with validity-renormalized
// trilinear. See docs/task017-baked-gi-probe-system-design.md.
// =============================================================================
class BakedGI
{
public:
    static constexpr uint32_t kDirections      = 16; // delta/sky-quadrature directions
    static constexpr uint32_t kResponseFloat4s = 10; // per (probe, dir): 9 SH9 RGB + V_i
    static constexpr uint32_t kLitFloat4s     = 9;  // per probe: 9 SH9 RGB
    static constexpr uint32_t kBakeIterations  = 3;  // series-expansion bounces
    static constexpr uint32_t kMaxProbeCount   = 128 * 1024;
    static constexpr uint32_t kMaxDim          = 256; // per-axis grid cap

    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Records the full bake (grid setup, buffer (re)creation, series
    // iterations) into cmdList. The caller closes/executes/waits — the
    // Renderer::BakeGIProbes wrapper follows the BuildAccelerationStructures
    // reset/record/ExecuteCommandList pattern, so buffers are never in flight
    // when released.
    bool RecordBake(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                    ID3D12RootSignature* rootSignature, Model* model,
                    D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                    float spacing);

    // Per-frame: fold the live sun/sky state into the lit-probe buffer.
    void RecordUpdate(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                      D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                      const DirectX::XMFLOAT3& sunIrradiance, const DirectX::XMFLOAT3& sunToDir);

    bool              IsValid() const { return m_Valid; }
    float             GetSpacing() const { return m_Spacing; }
    uint32_t          GetProbeCount() const { return m_ProbeCount; }
    DirectX::XMFLOAT3 GetGridMin() const { return m_GridMin; }
    uint32_t          GetDimX() const { return m_DimX; }
    uint32_t          GetDimY() const { return m_DimY; }
    uint32_t          GetDimZ() const { return m_DimZ; }
    uint32_t          GetLitSRVIndex() const { return (uint32_t)m_LitProbes.srvIndex; }
    uint32_t          GetMetaSRVIndex() const { return (uint32_t)m_Meta.srvIndex; }
    uint32_t          GetResponseSRVIndex() const { return (uint32_t)m_Response[m_FinalResponse].srvIndex; }
    ID3D12PipelineState* GetDebugPSO() const { return m_DebugPSO.Get(); }

    // Post-bake stats logging. Scene bounds / grid dims are logged inside
    // RecordBake; these two cover the GPU-side probe validity:
    //   1. RecordMetaReadback — appends a meta -> readback copy to the bake
    //      command list (call after RecordBake, before executing).
    //   2. LogBakeStats — maps the readback (GPU idle) and logs the valid /
    //      invalid probe coverage.
    bool RecordMetaReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void LogBakeStats() const;

private:
    // Ping-pong transport table: probeCount x kDirections x kResponseFloat4s float4s.
    GPUBuffer m_Response[2];
    GPUBuffer m_LitProbes; // per-frame lit SH9 RGB: probeCount x kLitFloat4s float4s
    GPUBuffer m_Meta;      // per-probe validity (bit 0)

    uint32_t  m_FinalResponse = 0; // response buffer holding the completed bake
    bool      m_Valid = false;
    float     m_Spacing = 2.0f;
    uint32_t  m_DimX = 0, m_DimY = 0, m_DimZ = 0, m_ProbeCount = 0;
    DirectX::XMFLOAT3 m_GridMin = { 0.0f, 0.0f, 0.0f };

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_BakePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_UpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DebugPSO; // RTV R8G8B8A8 (post-composite overlay)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_MetaReadback; // per-bake validity stats readback
};
