#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;

// =============================================================================
// DDGI — dynamic diffuse GI: a runtime octahedral irradiance probe volume.
//
// Indirect method 3. The two other methods are BakedGI (method 1) and ReSTIR
// GI (method 2). The three are exclusive; the user interface arbitrates.
// See docs/task023-ddgi-implementation-plan.md (the design) and
// docs/task025-ddgi-implementation-record.md (the as-built record).
//
// Every frame, for every probe:
//   1. TraceRaysCS casts 64 rays (32 stable + 32 rotating) against the TLAS.
//      A frontface hit evaluates the sun AT THE HIT with a shadow ray, then
//      adds the previous frame's atlas as feedback — the multi-bounce loop.
//      A miss stores the sky cubemap.
//   2. UpdateIrradianceCS folds the rotating rays into an octahedral
//      irradiance atlas (6x6 texels for each probe, plus a border ring).
//   3. UpdateDepthCS folds them into a depth-moment atlas (14x14 texels).
//   4. UpdateProbeStatesCS relocates the probes out of geometry and
//      classifies them.
//   5. The deferred pass samples the atlas at the receiver.
//
// There is no dirty flag and no sun gating: the method re-traces every frame.
// The temporal blend (0.98 history) carries the cost of the low ray count.
//
// LIFETIME. RecordSetup runs on a SYNCHRONOUS command list (the
// Renderer::BakeGIProbes pattern), so releasing and recreating the resources is
// safe. RecordFrame only records dispatches onto the live frame list.
// =============================================================================
class DDGI
{
public:
    // Must match Sources/Shaders/DDGI.hlsli.
    static constexpr uint32_t kIrradianceTexels = 6;
    static constexpr uint32_t kDepthTexels      = 14;
    static constexpr uint32_t kStableRays       = 32;
    static constexpr uint32_t kRaysPerProbe     = 64;      // 32 stable + 32 rotating
    static constexpr uint32_t kMaxProbeCount    = 128 * 1024;
    static constexpr uint32_t kMaxDim           = 256;     // per-axis grid cap
    static constexpr uint32_t kMaxAtlasDim      = 16384;   // per-side atlas cap
    static constexpr float    kHistoryBlendWeight = 0.98f; // weight of the history

    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Computes the grid from the scene world bounds, (re)creates the two atlas
    // ping-pong pairs, the ray hit buffer and the probe buffers, then clears
    // them. Call on a synced command list.
    bool RecordSetup(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                     ID3D12RootSignature* rootSignature, Model* model, float spacing,
                     D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress);

    // Records the trace pass and the three resolve passes, then leaves the
    // current atlas pair in a pixel-read state for the deferred consumer.
    void RecordFrame(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                     D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                     D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                     D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                     D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                     D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                     D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                     const DirectX::XMFLOAT3& sunIrradiance,
                     const DirectX::XMFLOAT3& sunToDir);

    bool     IsValid() const { return m_Valid; }
    float    GetSpacing() const { return m_Spacing; }
    uint32_t GetProbeCount() const { return m_ProbeCount; }
    DirectX::XMFLOAT3 GetGridMin() const { return m_GridMin; }
    uint32_t GetDimX() const { return m_DimX; }
    uint32_t GetDimY() const { return m_DimY; }
    uint32_t GetDimZ() const { return m_DimZ; }
    uint32_t GetProbesPerRow() const { return m_ProbesPerRow; }

    // The atlas pair the CONSUMER must read: the parity target of the frame
    // that RecordFrame records next. The getters and RecordFrame agree because
    // both derive the parity from m_FrameSerial, which only RecordFrame
    // advances.
    uint32_t GetIrradianceSRVIndex() const { return (uint32_t)m_Irradiance[m_FrameSerial & 1u].srvIndex; }
    uint32_t GetDepthSRVIndex() const      { return (uint32_t)m_Depth[m_FrameSerial & 1u].srvIndex; }
    uint32_t GetProbeStateSRVIndex() const  { return (uint32_t)m_ProbeStates.srvIndex; }
    uint32_t GetProbeOffsetSRVIndex() const { return (uint32_t)m_ProbeOffsets.srvIndex; }

    uint64_t GetResourceBytes() const;
    ID3D12PipelineState* GetDebugPSO() const { return m_DebugPSO.Get(); }

private:
    bool CreateResources();
    // Dispatches the three setup clears and leaves every atlas in a
    // compute-read state.
    void RecordClears(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                      D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress);

    // Two atlas pairs, ping-ponged by frame parity. The resolve passes write the
    // parity target; the trace pass and the resolves read the other copy.
    GPUTexture m_Irradiance[2];
    GPUTexture m_Depth[2];
    GPUBuffer  m_RayHits;      // float4(radiance.rgb, signedDepth), 64 for each probe
    GPUBuffer  m_ProbeOffsets; // float4 relocation offset
    GPUBuffer  m_ProbeStates;  // uint 0 = active, 1 = inactive

    uint32_t m_FrameSerial = 0;    // frames recorded since setup; parity picks the target
    bool     m_Valid = false;
    float    m_Spacing = 1.0f;
    uint32_t m_DimX = 0, m_DimY = 0, m_DimZ = 0, m_ProbeCount = 0;
    uint32_t m_ProbesPerRow = 1;
    uint32_t m_IrradianceWidth = 0, m_IrradianceHeight = 0;
    uint32_t m_DepthWidth = 0, m_DepthHeight = 0;
    DirectX::XMFLOAT3 m_GridMin = { 0.0f, 0.0f, 0.0f };

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_TracePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrradiancePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DepthPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_StatePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ClearIrradiancePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ClearDepthPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ClearBuffersPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DebugPSO;
};
