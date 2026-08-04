#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;

// -----------------------------------------------------------------------------
// MeshletPass
//
// GPU-driven meshlet pipeline: bin -> mesh-shader rasterize. Culling and HZB
// have been extracted to GPUCulling (see Sources/Rendering/GPUCulling.h).
//
// Named "MeshletPass" (rather than "Meshlet") to avoid colliding with the
// shared per-mesh `struct Meshlet` GPU data layout defined in Shared/SharedTypes.h.
// -----------------------------------------------------------------------------
class MeshletPass
{
public:
    // Creates binning buffers, class flags, and the visibility-buffer.
    // Culling-side resources are created by GPUCulling::CreateResources.
    void CreateResources(uint32_t internalWidth, uint32_t internalHeight);
    void RecreateVisibilityBuffer(uint32_t internalWidth, uint32_t internalHeight);

    // Compiles binning CS PSOs, mesh-shader raster PSOs, command signatures,
    // and the visibility-buffer debug-view PSO. Takes the shared cull root
    // signature from GPUCulling for binning PSOs (same 15-param layout).
    void CreatePipelines(ID3D12Device* device, ID3D12Device2* device2,
                          ID3D12RootSignature* mainRootSignature,
                          bool meshShaderSupported);

    // 4-pass GPU sort: PrepareArgs -> Classify -> AllocateBinRanges -> WriteBins.
    // Input SRV indices come from GPUCulling's output (VisibleMeshlets / Counter).
    void Binning(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                  D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                  int visibleMeshletsSRVIdx, int visibleMeshletsCounterSRVIdx,
                  int visibleMeshletsCounterUAVIdx);

    // Mesh Shader rasterize per bin (GPU-driven via ExecuteIndirect).
    void Rasterize(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                    D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model,
                    int visibleMeshletsSRVIdx);

    // ----- Visibility buffer for meshlet debug overlay -----
    GPUTexture& GetVisibilityBuffer() { return m_VisibilityBuffer; }
    ID3D12PipelineState* GetDebugViewPSO() const { return m_MeshletDebugViewPSO.Get(); }
    int GetDebugMode() const { return m_MeshletDebugMode; }
    void SetDebugMode(int mode) { m_MeshletDebugMode = mode; }

private:
    static constexpr uint32_t NUM_RASTER_BINS = 2;

    // ----- Binning resources -----
    GPUBuffer m_MeshletCounts;              // RWStructuredBuffer<uint>[NUM_RASTER_BINS]
    GPUBuffer m_MeshletOffsetAndCounts;     // RWStructuredBuffer<uint4>[NUM_RASTER_BINS]
    GPUBuffer m_DispatchMeshArgs;           // RWStructuredBuffer<uint>[NUM_RASTER_BINS*3]
    GPUBuffer m_BinnedMeshlets;             // RWStructuredBuffer<uint>[MAX_VISIBLE_MESHLETS]
    GPUBuffer m_GlobalMeshletCounter;       // RWStructuredBuffer<uint>[1]
    GPUBuffer m_ClassifyDispatchArgs;       // Indirect dispatch args for Classify/Write passes

    // ----- Debug -----
    GPUTexture m_VisibilityBuffer;          // R32_UINT visibility buffer (debug overlay)
    int        m_MeshletDebugMode = 0;      // 0=Off, 1=Instance, 2=Meshlet, 3=Primitive RasterID / 4=MipTint

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletBinPrepareArgsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletClassifyPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletAllocateBinRangesPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletWriteBinsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterPSO[NUM_RASTER_BINS];
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletDebugViewPSO;

    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchCommandSignatureCS;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchMeshSignature;

    bool m_MeshShaderSupported = false;
};
