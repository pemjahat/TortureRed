#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;
struct FrameConstants;
struct LightConstants;

// -----------------------------------------------------------------------------
// PathTracing
//
// Owns the full ray-traced path tracer pipeline: the "old" path tracer CS, the
// manual multi-pass ReSTIR GI implementation, and the RTXDI-SDK ReSTIR GI
// implementation (both share the same ray-gen dispatch entry point, so they are
// kept together here per the migration plan rather than split into a separate
// RestirGI/RestirDI-only file). Also owns the debug path-visualization line
// draw call, which reads back GPU-written ray paths from the same dispatch.
//
// PSO/resource creation happens once via CreateResources()/CreatePipelines();
// per-frame work happens via DispatchRays()/DrawPathVizLines().
// -----------------------------------------------------------------------------
class PathTracing
{
public:
    // Creates fixed WINDOW_WIDTH x WINDOW_HEIGHT textures/buffers (called once at
    // Renderer::Initialize time, before internal-resolution textures exist).
    bool CreateResources(ID3D12Device* device, bool rayTracingSupported, uint32_t internalWidth, uint32_t internalHeight);
    // Recreates the resolution-dependent textures/buffers (called from
    // Renderer::CreateInternalResolutionResources on resize).
    void OnResolutionChanged(uint32_t w, uint32_t h);

    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    void DispatchRays(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                       Model* model, const FrameConstants& frame,
                       D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, D3D12_GPU_VIRTUAL_ADDRESS tlasGPUAddress,
                       D3D12_GPU_VIRTUAL_ADDRESS lightsBufferAddress, D3D12_GPU_VIRTUAL_ADDRESS lightLUTBufferAddress,
                       uint32_t internalWidth, uint32_t internalHeight);

    void DrawPathVizLines(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                          D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                          D3D12_CPU_DESCRIPTOR_HANDLE backBufferRTV);

    // Output getters (mirrors former Renderer::Get* accessors)
    GPUTexture& GetOutput() { return m_PathTracerPresentOutput; }
    GPUTexture& GetHdrOutput() { return m_PathTracerOutput; }
    GPUTexture& GetRestirDebugHeatmap() { return m_RestirDebugHeatmap; }

    // NOTE: m_CurrentReservoirIndex is also shared/used by Renderer::DispatchRestirGI()
    // (the raster-indirect-GI path, not yet extracted) to keep its own ping-pong
    // reservoirs in phase with the ray-traced path. Exposed here until that path
    // is split into its own RestirGI class.
    int GetCurrentReservoirIndex() const { return m_CurrentReservoirIndex; }
    void SetCurrentReservoirIndex(int index) { m_CurrentReservoirIndex = index; }

private:
    // ----- Ray Tracing PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PathTracerPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PathTracerPresentPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirReservoirDebugPSO;

    // RTXDI SDK Pipeline States
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirReservoirDebugPSO;

    // Path Visualization Lines
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PathVizLinePSO;
    GPUBuffer m_PathVizLineBuffer;

    // ----- Resources -----
    GPUTexture m_PathTracerOutput;         // HDR (R16G16B16A16_FLOAT)
    GPUTexture m_PathTracerPresentOutput;  // LDR present (R8G8B8A8_UNORM)
    GPUTexture m_AccumulationBuffer;
    GPUTexture m_RestirDebugHeatmap;
    GPUBuffer m_ReservoirBuffer[2];         // Manual ReSTIR Reservoirs (Current and Previous)
    GPUBuffer m_ReservoirIntermediate;

    // RTXDI Reservoir Buffer (RTXDI_PackedGIReservoir)
    GPUBuffer m_RtxdiReservoirBuffer[2];
    GPUBuffer m_RtxdiReservoirIntermediate;
    GPUBuffer m_RtxdiNeighborOffsetsBuffer;

    int m_CurrentReservoirIndex = 0;

    bool m_RayTracingSupported = false;
};
