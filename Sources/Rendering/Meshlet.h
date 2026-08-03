#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;
struct FrameConstants;

// -----------------------------------------------------------------------------
// MeshletPass
//
// GPU-driven meshlet pipeline: cull -> bin (4-pass GPU sort) -> mesh-shader
// rasterize. Also owns the CPU-driven per-meshlet debug dispatch path and the
// visibility-buffer debug overlay PSO (plan001-meshletdebug).
//
// Named "MeshletPass" (rather than "Meshlet") to avoid colliding with the
// shared per-mesh `struct Meshlet` GPU data layout defined in Shared/SharedTypes.h.
//
// Resource/PSO creation happens once via CreateResources()/CreatePipelines();
// per-frame work happens via Cull()/Binning()/Rasterize()/RasterizeDebug().
// -----------------------------------------------------------------------------
class MeshletPass
{
public:
    void CreateResources(uint32_t internalWidth, uint32_t internalHeight);
    // Recreates only the resolution-dependent visibility buffer (called from
    // Renderer::CreateInternalResolutionResources on resize).
    void RecreateVisibilityBuffer(uint32_t internalWidth, uint32_t internalHeight);
    void CreatePipelines(ID3D12Device* device, ID3D12Device2* device2, ID3D12RootSignature* mainRootSignature, bool meshShaderSupported);

    // Hierarchical two-stage culling: CullInstancesCS → CullMeshletsCS.
    // occlusionEnabled=1: two-phase occlusion culling (Phase 1 vs prev-HZB, Phase 2 vs fresh HZB).
    // occlusionEnabled=0 (phase must be 0): frustum-only single-phase cull.
    void CullTwoPass(ID3D12GraphicsCommandList* cmdList, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model,
                     bool occlusionEnabled, int phase); // phase: 0=Phase1 (vs prev HZB), 1=Phase2 (vs fresh HZB)
    void Binning(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress);                    // 4-pass GPU sort: PrepareArgs -> Classify -> AllocateBinRanges -> WriteBins
    void Rasterize(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model);      // Mesh Shader rasterize per bin (GPU-driven)
    void RasterizeDebug(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model); // CPU-driven per-meshlet debug dispatch (no culling/binning)

    // ----- HZB (Hierarchical Z-Buffer) -----
    // Builds/rebuilds the HZB mip chain from `depthBuffer` via AMD FidelityFX SPD.
    void BuildHZB(ID3D12GraphicsCommandList* cmdList, GPUTexture& depthBuffer);
    GPUTexture& GetHZB() { return m_HZB; }
    uint32_t GetHZBMips() const { return m_HZBMips; }

    // Visibility buffer for meshlet debug overlay (plan001)
    GPUTexture& GetVisibilityBuffer() { return m_VisibilityBuffer; }
    int GetVisibleMeshletsSRVIndex() const { return m_VisibleMeshlets.srvIndex; }
    ID3D12PipelineState* GetDebugViewPSO() const { return m_MeshletDebugViewPSO.Get(); }
    int GetDebugMode() const { return m_MeshletDebugMode; }
    void SetDebugMode(int mode) { m_MeshletDebugMode = mode; }

private:
    static constexpr uint32_t NUM_RASTER_BINS = 2;
    static constexpr uint32_t SPD_MAX_MIPS = 12; // AMD FidelityFX SPD hard limit — ffx_spd.h (ThirdParty/FidelityFX-SPD)

    // Creates/recreates the resolution-dependent HZB texture + support buffers.
    // Called from CreateResources() and from the resize path alongside RecreateVisibilityBuffer().
    void CreateHZBResources(uint32_t internalWidth, uint32_t internalHeight);
    void CreateHZBPipelines(ID3D12Device* device);

    // ----- Resources -----
    GPUBuffer m_VisibleMeshlets;            // RWStructuredBuffer<MeshletCandidate>
    GPUBuffer m_VisibleMeshletsCounter;     // RWBuffer<uint>

    GPUBuffer m_MeshletCounts;              // RWStructuredBuffer<uint>[NUM_RASTER_BINS]
    GPUBuffer m_MeshletOffsetAndCounts;     // RWStructuredBuffer<uint4>[NUM_RASTER_BINS] — (count,1,1,offset) — SRV only
    GPUBuffer m_DispatchMeshArgs;           // RWStructuredBuffer<uint>[NUM_RASTER_BINS*3] — INDIRECT_ARGUMENT only
    GPUBuffer m_BinnedMeshlets;             // RWStructuredBuffer<uint>[MAX_VISIBLE_MESHLETS] — sorted indirection
    GPUBuffer m_GlobalMeshletCounter;       // RWStructuredBuffer<uint>[1] — prefix-sum scratch
    GPUBuffer m_ClassifyDispatchArgs;       // Indirect dispatch args for Classify/Write passes

    // Debug: visibility buffer for meshlet debug overlay (plan001)
    GPUTexture m_VisibilityBuffer;          // R32_UINT — packed (candidateIndex, primitiveID)
    int m_MeshletDebugMode = 0;             // 0=Off, 1=Instance, 2=Meshlet, 3=Primitive

    // ----- HZB (Hierarchical Z-Buffer) resources -----
    GPUTexture m_HZB;                                // R32_FLOAT, multi-mip. Reverse-Z: stores NEAREST (closest) depth per texel (min-reduce).
    int        m_HZBMipUAVIndices[SPD_MAX_MIPS] = {}; // Bindless UAV index per mip (CPU-side cache; -1 = not yet allocated)
    GPUBuffer  m_HZBMipIndicesBuffer;                 // StructuredBuffer<uint>[12] SRV — uploaded copy of m_HZBMipUAVIndices, read by HZB.hlsl
    GPUBuffer  m_HZBSpdCounter;                       // RWStructuredBuffer<uint>[1] — SPD's global atomic counter
    GPUBuffer  m_HZBConstantsBuffer;                  // Upload-heap CBV for HZBConstants (root param b0 of m_HZBRootSignature)
    uint32_t   m_HZBWidth = 0, m_HZBHeight = 0, m_HZBMips = 0;

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletBinPrepareArgsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletClassifyPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletAllocateBinRangesPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletWriteBinsPSO;

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterPSO[NUM_RASTER_BINS];      // Normal render
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterDebugPSO[NUM_RASTER_BINS]; // Debug mode (writes vis buffer)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterDirectPSO;                 // CPU-driven debug: one DispatchMesh per meshlet

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletDebugViewPSO;

    // HZB PSOs (main root signature — bindless heap access via ResourceDescriptorHeap[])
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_HZBInitPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_HZBCreatePSO;

    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchCommandSignatureCS;  // Indirect Dispatch (for binning)
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchMeshSignature;       // Indirect DispatchMesh (for rasterize)

    // Unified root signature for all meshlet cull passes (old frustum-only + two-pass occlusion).
    // Superset of both: 15 root params (2 CBV + 5 SRV + 8 UAV) + static point-clamp sampler +
    // CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED flag (required for HZB bindless access).
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_MeshletRootSignature;

    // Dedicated root signature for HZB Init/Create passes: single root CBV (b0, HZBConstants) +
    // CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED flag for ResourceDescriptorHeap[] bindless access.
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_HZBRootSignature;

    // ----- Two-pass occlusion culling resources -----
    GPUBuffer m_CandidateMeshlets;              // RWStructuredBuffer<MeshletCandidate> — enum'd by CullInstancesCS
    GPUBuffer m_CandidateMeshletsCounter;       // RWStructuredBuffer<uint>[1] — atomic counter
    GPUBuffer m_OccludedInstances;              // RWStructuredBuffer<uint> — deferred instance indices for Phase 2
    GPUBuffer m_OccludedInstancesCounter;       // RWStructuredBuffer<uint>[1]
    GPUBuffer m_MeshletCullArgs;                // Indirect dispatch args for CullMeshletsCS (uint3)
    GPUBuffer m_InstanceCullArgs;               // Indirect dispatch args for Phase 2 CullInstancesCS (uint3)
    GPUBuffer m_TwoPassCullConstantsBuffer[2];  // Upload-heap CBV, double-buffered: [0]=Phase1, [1]=Phase2

    // Two-pass cull PSOs
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_CullInstancesPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_CullMeshletsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_BuildMeshletCullIndirectArgsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_BuildInstanceCullIndirectArgsPSO;

    // Creates the TwoPassCull root signature and compiles all two-pass cull CS PSOs.
    void CreateTwoPassCullPipelines(ID3D12Device* device);

    bool m_MeshShaderSupported = false;
};
