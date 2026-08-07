#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;

// -----------------------------------------------------------------------------
// MeshletPass
//
// GPU-driven meshlet pipeline: build dispatch args -> mesh-shader rasterize.
// Culling and HZB have been extracted to GPUCulling (see Sources/Rendering/GPUCulling.h).
//
// Named "MeshletPass" (rather than "Meshlet") to avoid colliding with the
// shared per-mesh `struct Meshlet` GPU data layout defined in Shared/SharedTypes.h.
//
// No binning — single combined PSO handles opaque + alpha-masked in one pass
// (alpha discard runs unconditionally in the pixel shader). Alpha-blended
// meshes are rejected in the culling stage and never reach the meshlet pipeline.
// -----------------------------------------------------------------------------
class MeshletPass
{
public:
    // Creates dispatch args buffer, class flags, and the visibility-buffer.
    // Culling-side resources are created by GPUCulling::CreateResources.
    void CreateResources(uint32_t internalWidth, uint32_t internalHeight);
    void RecreateVisibilityBuffer(uint32_t internalWidth, uint32_t internalHeight);

    // Compiles BuildDispatchMeshArgs CS PSO, mesh-shader raster PSOs, command
    // signatures, and the visibility-buffer debug-view PSO.
    void CreatePipelines(ID3D12Device* device, ID3D12Device2* device2,
                          ID3D12RootSignature* mainRootSignature,
                          bool meshShaderSupported);

    // Builds indirect DispatchMesh arguments from the VisibleMeshletsCounter.
    // Must be called after culling and before Rasterize().
    // phase: TWO_PASS_PHASE_FIRST/SECOND — selects which VisibleMeshletsCounter slot to
    // read (that phase's OWN new-meshlet count, not a running total across phases).
    void BuildDispatchMeshArgs(ID3D12GraphicsCommandList* cmdList,
                                ID3D12RootSignature* mainRootSignature,
                                D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                                int visibleMeshletsCounterSRVIdx, uint32_t phase);

    // Mesh Shader rasterize — single ExecuteIndirect DispatchMesh.
    // useVisibilityBuffer=true  -> MeshletRasterizeMS.hlsl, writes ONLY the visibility
    //                              token (R32_UINT) + depth. Caller binds 1 RTV.
    // useVisibilityBuffer=false -> MeshletRasterizeGBufferMS.hlsl, writes GBuffer
    //                              (albedo/normal/material) + visibility token directly.
    //                              Caller binds 4 RTVs.
    // phase: TWO_PASS_PHASE_FIRST/SECOND — Phase 2's mesh shader must offset its GroupID
    // into VisibleMeshlets[] by Phase 1's final count (visibleMeshletsCounterSRVIdx),
    // since both phases' candidates now coexist in the same buffer (see
    // docs/bug_flyingworld_meshlet_flicker.md).
    void Rasterize(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                    D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model,
                    int visibleMeshletsSRVIdx, int visibleMeshletsCounterSRVIdx,
                    bool useVisibilityBuffer, uint32_t phase);

    // Full-screen Visibility Buffer resolve: reconstructs GBuffer (albedo/normal/material)
    // from the visibility token written by Rasterize(). Run once per frame, after both
    // rasterize phases (Phase 1 / Phase 2) complete. Caller must have already bound the
    // GBuffer albedo/normal/material RTVs (no depth) and transitioned the visibility
    // buffer to a shader-readable state.
    void ResolveVisibilityGBuffer(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                                   D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model,
                                   int visibleMeshletsSRVIdx);

    // ----- Visibility buffer for meshlet debug overlay -----
    GPUTexture& GetVisibilityBuffer() { return m_VisibilityBuffer; }
    ID3D12PipelineState* GetDebugViewPSO() const { return m_MeshletDebugViewPSO.Get(); }
    int GetDebugMode() const { return m_MeshletDebugMode; }
    void SetDebugMode(int mode) { m_MeshletDebugMode = mode; }

    int GetDispatchMeshArgsSRVIndex() const { return m_DispatchMeshArgs.srvIndex; }
    int GetDispatchMeshArgsUAVIndex() const { return m_DispatchMeshArgs.uavIndex; }

private:
    // ----- Dispatch args (indirect DispatchMesh arguments, 1 entry) -----
    GPUBuffer m_DispatchMeshArgs;           // RWStructuredBuffer<uint>[3] — single DispatchMesh indirect arg

    // ----- Debug -----
    GPUTexture m_VisibilityBuffer;          // R32_UINT visibility buffer (debug overlay)
    int        m_MeshletDebugMode = 0;      // 0=Off, 1=Instance, 2=Meshlet, 3=Primitive RasterID / 4=MipTint

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_BuildDispatchMeshArgsPSO; // 1-thread CS: builds DispatchMesh indirect args
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterPSO;         // Visibility-only (1 RTV)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletRasterGBufferPSO;  // Direct-to-GBuffer (4 RTVs)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MeshletDebugViewPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_VisibilityGBufferPSO; // Full-screen VisibilityGBuffer.hlsl resolve

    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchMeshSignature;

    bool m_MeshShaderSupported = false;
};
