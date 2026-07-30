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

    void Cull(ID3D12GraphicsCommandList* cmdList, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model);
    void Binning(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress);                    // 4-pass GPU sort: PrepareArgs -> Classify -> AllocateBinRanges -> WriteBins
    void Rasterize(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model);      // Mesh Shader rasterize per bin (GPU-driven)
    void RasterizeDebug(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model); // CPU-driven per-meshlet debug dispatch (no culling/binning)

    // Visibility buffer for meshlet debug overlay (plan001)
    GPUTexture& GetVisibilityBuffer() { return m_VisibilityBuffer; }
    int GetVisibleMeshletsSRVIndex() const { return m_VisibleMeshlets.srvIndex; }
    ID3D12PipelineState* GetDebugViewPSO() const { return m_MeshletDebugViewPSO.Get(); }
    int GetDebugMode() const { return m_MeshletDebugMode; }
    void SetDebugMode(int mode) { m_MeshletDebugMode = mode; }

private:
    static constexpr uint32_t NUM_RASTER_BINS = 2;

    // ----- Resources -----
    GPUBuffer m_VisibleMeshlets;            // RWStructuredBuffer<MeshletCandidate>
    GPUBuffer m_VisibleMeshletsCounter;     // RWBuffer<uint>
    GPUBuffer m_CullDispatchArgs;           // Indirect dispatch args for cull
    GPUBuffer m_CullConstantsBuffer;        // Cull constants (total meshlets)
    GPUBuffer m_VisibleMeshletsDebug;       // DEBUG: per-visible-meshlet VertexCount/TriangleCount

    GPUBuffer m_MeshletCounts;              // RWStructuredBuffer<uint>[NUM_RASTER_BINS]
    GPUBuffer m_MeshletOffsetAndCounts;     // RWStructuredBuffer<uint4>[NUM_RASTER_BINS] — (count,1,1,offset) — SRV only
    GPUBuffer m_DispatchMeshArgs;           // RWStructuredBuffer<uint>[NUM_RASTER_BINS*3] — INDIRECT_ARGUMENT only
    GPUBuffer m_BinnedMeshlets;             // RWStructuredBuffer<uint>[MAX_VISIBLE_MESHLETS] — sorted indirection
    GPUBuffer m_GlobalMeshletCounter;       // RWStructuredBuffer<uint>[1] — prefix-sum scratch
    GPUBuffer m_ClassifyDispatchArgs;       // Indirect dispatch args for Classify/Write passes

    // Debug: visibility buffer for meshlet debug overlay (plan001)
    GPUTexture m_VisibilityBuffer;          // R32_UINT — packed (candidateIndex, primitiveID)
    int m_MeshletDebugMode = 0;             // 0=Off, 1=Instance, 2=Meshlet, 3=Primitive

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletCullPSO;

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletBinPrepareArgsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletClassifyPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletAllocateBinRangesPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletWriteBinsPSO;

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterPSO[NUM_RASTER_BINS];      // Normal render
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterDebugPSO[NUM_RASTER_BINS]; // Debug mode (writes vis buffer)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterDirectPSO;                 // CPU-driven debug: one DispatchMesh per meshlet

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletDebugViewPSO;

    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchCommandSignatureCS;  // Indirect Dispatch (for binning)
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchMeshSignature;       // Indirect DispatchMesh (for rasterize)

    // Root signature for meshlet cull pass (separate from main RS)
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_MeshletRootSignature;

    bool m_MeshShaderSupported = false;
};
