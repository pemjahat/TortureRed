#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;

// -----------------------------------------------------------------------------
// GPUCulling
//
// Extracted from MeshletPass — owns the Hierarchical Z-Buffer, two-pass
// occlusion culling (CullInstancesCS → CullMeshletsCS), and all cull-side
// debug overlays (HZB mip viewer, occluded-rect overlay, depth readout).
//
// MeshletPass remains responsible for binning, mesh-shader rasterize,
// and the visibility-buffer debug overlay (consuming GPUCulling's output).
// -----------------------------------------------------------------------------
class GPUCulling
{
public:
    // Creates all culling-side resources: candidate/occluded/visible meshlet
    // buffers, two-pass cull constants, debug-recording buffers, mip-tint
    // sideband, and HZB texture + SPD support buffers.
    void CreateResources(uint32_t internalWidth, uint32_t internalHeight);
    // Recreates only the resolution-dependent HZB texture (called on window
    // resize alongside MeshletPass::RecreateVisibilityBuffer).
    void RecreateHZB(uint32_t internalWidth, uint32_t internalHeight);

    // Compiles two-pass cull CS PSOs and HZB SPD PSOs. Returns the unified
    // meshlet-cull root signature (15 params) via outRootSig — MeshletPass
    // reuses it for its binning PSOs.
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* mainRootSignature);

    // ----- Two-pass occlusion culling -----
    // occlusionEnabled=1: two-phase (Phase 0 vs prev-HZB, Phase 1 vs fresh HZB)
    // occlusionEnabled=0: frustum-only single-phase hierarchical cull
    // mainRootSignature: needed by internal CopyCullStatsCS dispatch (shared main root sig param 13).
    void CullTwoPass(ID3D12GraphicsCommandList* cmdList, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                     Model* model, bool occlusionEnabled, int phase,
                     ID3D12RootSignature* mainRootSignature);

    // ----- HZB (Hierarchical Z-Buffer) -----
    void BuildHZB(ID3D12GraphicsCommandList* cmdList, GPUTexture& depthBuffer);
    GPUTexture& GetHZB() { return m_HZB; }
    uint32_t GetHZBMips() const { return m_HZBMips; }
    void DebugViewHZB(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                      uint32_t outputUAVIdx, uint32_t outputWidth, uint32_t outputHeight, int mipLevel);

    // ----- Occluded-rect debug -----
    void SetDebugRecordOccluded(bool enabled) { m_DebugRecordOccluded = enabled; }
    void DrawOccludedRects(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                           D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, GPUTexture& output,
                           uint32_t outputWidth, uint32_t outputHeight);
    void EmitDepthReadout(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                          uint32_t dataUAVIdx, uint32_t glyphSRVIdx, float fontSize,
                          uint32_t backbufferWidth, uint32_t backbufferHeight);
    // Controls the mip-tint sideband write during culling.
    void SetDebugRecordMip(bool enabled) { m_DebugRecordMipEnabled = enabled; }

    // ----- Cull stats overlay (on-screen text table via GPU debug text system) -----
    void EmitCullStats(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                       D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                       uint32_t dataUAVIdx, uint32_t glyphSRVIdx, float fontSize,
                       uint32_t backbufferWidth, uint32_t backbufferHeight,
                       uint32_t totalInstances, uint32_t totalMeshlets);
    void SetShowCullStats(bool show) { m_ShowCullStats = show; }
    bool GetShowCullStats() const { return m_ShowCullStats; }

    // ----- Culling output (consumed by MeshletPass::Binning / Rasterize / debug overlay) -----
    int GetVisibleMeshletsSRVIndex() const { return m_VisibleMeshlets.srvIndex; }
    int GetVisibleMeshletMipsSRVIndex() const { return m_VisibleMeshletMips.srvIndex; }
    GPUBuffer& GetVisibleMeshletMipsBuffer() { return m_VisibleMeshletMips; }
    int GetVisibleMeshletsCounterSRVIndex() const { return m_VisibleMeshletsCounter.srvIndex; }
    int GetVisibleMeshletsCounterUAVIndex() const { return m_VisibleMeshletsCounter.uavIndex; }
    int GetCandidateMeshletsCounterSRVIndex() const { return m_CandidateMeshletsCounter.srvIndex; }
    int GetOccludedInstancesCounterSRVIndex() const { return m_OccludedInstancesCounter.srvIndex; }
    // CullStatsBuffer UAV/SRV indices — written by CopyCullStatsCS, read by CullStatsCS.
    int GetCullStatsBufferSRVIndex() const { return m_CullStatsBuffer.srvIndex; }
    int GetCullStatsBufferUAVIndex() const { return m_CullStatsBuffer.uavIndex; }

private:
    static constexpr uint32_t SPD_MAX_MIPS = 12; // AMD FidelityFX SPD hard limit

    void CreateHZBResources(uint32_t internalWidth, uint32_t internalHeight);

    // ----- HZB resources -----
    GPUTexture m_HZB;                                // R32_FLOAT, multi-mip, reverse-Z nearest-depth (min-reduce)
    int        m_HZBMipUAVIndices[SPD_MAX_MIPS] = {}; // Bindless UAV index per mip
    GPUBuffer  m_HZBMipIndicesBuffer;                 // StructuredBuffer<uint>[12] SRV — uploaded copy for HZB.hlsl
    GPUBuffer  m_HZBSpdCounter;                       // RWStructuredBuffer<uint>[1] — SPD global atomic counter
    GPUBuffer  m_HZBConstantsBuffer;                  // Upload-heap CBV for HZBConstants (b0 of m_HZBRootSignature)
    uint32_t   m_HZBWidth = 0, m_HZBHeight = 0, m_HZBMips = 0;

    // ----- Two-pass occlusion culling resources -----
    GPUBuffer m_VisibleMeshlets;            // RWStructuredBuffer<MeshletCandidate> — output of CullMeshletsCS
    GPUBuffer m_VisibleMeshletsCounter;     // RWStructuredBuffer<uint>[1]
    GPUBuffer m_CandidateMeshlets;          // RWStructuredBuffer<MeshletCandidate> — output of CullInstancesCS
    GPUBuffer m_CandidateMeshletsCounter;   // RWStructuredBuffer<uint>[1]
    GPUBuffer m_OccludedInstances;          // RWStructuredBuffer<uint> — deferred instance indices for Phase 2
    GPUBuffer m_OccludedInstancesCounter;   // RWStructuredBuffer<uint>[1]
    GPUBuffer m_MeshletCullArgs;            // Indirect dispatch args for CullMeshletsCS (uint3)
    GPUBuffer m_InstanceCullArgs;           // Indirect dispatch args for Phase 2 CullInstancesCS (uint3)
    GPUBuffer m_TwoPassCullConstantsBuffer[2]; // Upload-heap CBV, double-buffered: [0]=Phase1, [1]=Phase2

    // ----- Cull stats debug -----
    GPUBuffer m_CullStatsBuffer;            // RWStructuredBuffer<uint>[CULL_STATS_COUNT] — written by CopyCullStatsCS
    bool      m_ShowCullStats = false;      // Toggle for on-screen stats overlay

    // ----- Occluded-rect debug -----
    GPUBuffer m_OccludedRects;          // RWStructuredBuffer<OccludedRectDebug>
    GPUBuffer m_OccludedRectsCounter;   // RWStructuredBuffer<uint>[1] — append counter
    bool      m_DebugRecordOccluded = false;

    // ----- Mip-selection tint sideband -----
    GPUBuffer m_VisibleMeshletMips;     // RWStructuredBuffer<uint> — one mip per visible-meshlet slot

    // ----- PSOs -----
    // Two-pass cull
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_CullInstancesPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_CullMeshletsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_BuildMeshletCullIndirectArgsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_BuildInstanceCullIndirectArgsPSO;
    // HZB
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_HZBInitPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_HZBCreatePSO;
    // Debug overlays
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_HZBDebugViewPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_OccludedRectBackgroundPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_OccludedRectsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DepthReadoutPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_CopyCullStatsPSO;   // 1-thread CS — copies per-phase counters to stats buffer
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_CullStatsPSO;       // 1-thread CS — debug-text renderer, reads stats buffer

    // ----- Root signatures -----
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_CullRootSignature; // 15-param unified cull root sig
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_HZBRootSignature;     // 1-CBV HZB root sig
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchCommandSignatureCS; // Indirect Dispatch (CullMeshletsCS ExecuteIndirect)

    bool m_DebugRecordMipEnabled = false; // Mip-tint sideband write toggle
};
