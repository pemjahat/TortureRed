#pragma once

#include "Graphics/GraphicsTypes.h"

struct FrameConstants;
struct LightConstants;

// -----------------------------------------------------------------------------
// Sky
//
// Owns the Hosek-Wilkie analytic sky model: CPU-side coefficient solve, baked
// radiance cubemap, and GPU-side SH9 irradiance projection. Two-tier consumption:
//
//   Tier 1: SampleSky(dir) cubemap accessor used by ray-traced GI ray-miss
//           paths (PathTracer, SHaRC, ReSTIR GI).
//   Tier 2: EvalSH9Irradiance(N, SkySH9) ambient term in the deferred lighting
//           pass (Lighting.hlsl).
//
// Bake is CPU-side (matching PathTracer's reference implementation), triggered
// by a dirty flag on sun-direction/turbidity/albedo change. Cubemap data is
// uploaded via a staging buffer; SH9 projection runs on GPU over the baked
// cubemap.
//
// Turbidity, ground albedo, and the sky enable toggle use fixed code defaults
// (no runtime ImGui panel in this pass). Sun direction comes from lights[0].
// -----------------------------------------------------------------------------
class Sky
{
public:
    void CreateResources(ID3D12Device* device, uint32_t internalWidth, uint32_t internalHeight);
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Called once per frame, before the G-Buffer pass. Checks the dirty flag
    // (sun direction / turbidity / albedo changed) and re-bakes + re-projects
    // if needed. Sky is always enabled; no toggle required.
    void Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                 D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                 const LightConstants& sunLight,
                 float turbidity, float groundAlbedo);

    // Bindless indices for the sky cubemap and SH9 buffer, to be written
    // into FrameConstants each frame.
    uint32_t GetSkyCubemapSRVIndex()  const { return (uint32_t)m_SkyCubemap.srvIndex; }
    uint32_t GetSkySH9BufferSRVIndex() const { return (uint32_t)m_SkySH9Buffer.srvIndex; }

    // Prompt a re-bake on the next Execute() call (e.g. after sun direction changes).
    void MarkDirty() { m_Dirty = true; }

    // Per-frame before/after transition helpers for Renderer.cpp.
    // Sky cubemap stays in SRV after bake; unrelated passes must not write to it.
    void TransitionCubemapToSRV(ID3D12GraphicsCommandList* cmdList);
    void TransitionSH9ToUAV(ID3D12GraphicsCommandList* cmdList);
    void TransitionSH9ToSRV(ID3D12GraphicsCommandList* cmdList);

private:
    // Bakes the Hosek-Wilkie cubemap on CPU and uploads via staging buffer.
    bool BakeCubemap(ID3D12GraphicsCommandList* cmdList,
                     const LightConstants& sunLight, float turbidity, float groundAlbedo);

    // Dispatches the SH9 projection compute shader over the baked cubemap.
    void DispatchSH9Projection(ID3D12GraphicsCommandList* cmdList,
                               ID3D12RootSignature* rootSignature,
                               D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress);

    static constexpr UINT kCubemapSize   = 128;  // 128x128 per face, 6 faces
    static constexpr UINT kCubemapFaces  = 6;
    static constexpr DXGI_FORMAT kCubemapFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

    GPUTexture m_SkyCubemap;          // TextureCube, 128x128x6 R16G16B16A16_FLOAT
    GPUBuffer  m_SkySH9Buffer;        // RWStructuredBuffer<float4>, 9 elements
    GPUBuffer  m_StagingBuffer;       // UPLOAD buffer for cubemap data transfer

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ProjectSH9PSO;

    // Dirty tracking
    float m_LastTurbidity    = -1.0f;
    float m_LastGroundAlbedo = -1.0f;
    float m_LastSunElevation = -999.0f;
    bool  m_Dirty            = true;
    bool  m_ResourcesCreated = false;
};
