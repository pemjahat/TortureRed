#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;
struct FrameConstants;
struct GBuffer; // Graphics/GBuffer resource struct (albedo/normal/material/depth), still owned by Renderer

// -----------------------------------------------------------------------------
// RestirGI
//
// Owns the raster-driven indirect-GI pipeline: SHaRC (Spatial Hash Radiance
// Cache) secondary-ray sampling used as the RTDGI/RTR temporal history source,
// plus the split diffuse/specular ReSTIR reservoirs (temporal -> spatial ->
// resolve-to-intermediates). Writes its resolved result into the shared
// FinalDiffuse/FinalSpecular interchange textures via the generic
// NrdStoreShadingOutput bridge PSO (shared with RestirDI, stays in Renderer).
//
// NRD triggering, the "swap ping-pong reservoir index" shared with PathTracing,
// and the FullScreenDebug SRV-transition tail are cross-cutting concerns, so
// Execute() takes/returns what it needs via parameters/return value and
// Renderer::DispatchRestirGI() still orchestrates the NRD/reservoir-index tail.
// -----------------------------------------------------------------------------
class RestirGI
{
public:
    void CreateResources(uint32_t internalWidth, uint32_t internalHeight);
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Returns true if the SHaRC debug overlay was rendered instead of the normal
    // GI passes (caller should skip NRD and treat this frame as "GI did not run").
    bool Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                 Model* model, const FrameConstants& frame,
                 D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, D3D12_GPU_VIRTUAL_ADDRESS tlasGPUAddress,
                 D3D12_GPU_VIRTUAL_ADDRESS lightsBufferAddress, D3D12_GPU_VIRTUAL_ADDRESS lightLUTBufferAddress,
                 const GBuffer& gbuffer, GPUTexture& fullScreenDebugTex,
                 GPUTexture& finalDiffuseTex, GPUTexture& finalSpecularTex,
                 ID3D12PipelineState* nrdStoreShadingOutputPSO,
                 int currentReservoir, int previousReservoir,
                 uint32_t internalWidth, uint32_t internalHeight);

    GPUTexture& GetGIDiffuseIntermediate()  { return m_GIDiffuseIntermediate; }
    GPUTexture& GetGISpecularIntermediate() { return m_GISpecularIntermediate; }

private:
    static constexpr UINT SHARC_HASH_ENTRIES_NUM = 4 * 1024 * 1024;

    // ----- SHaRC (Spatial Hash Radiance Cache) resources -----
    GPUBuffer m_SharcHashEntriesBuf;    // uint64_t x SHARC_HASH_ENTRIES_NUM = 32 MB
    GPUBuffer m_SharcAccumulationBuf;   // SharcAccumulationData (uint4) x SHARC_HASH_ENTRIES_NUM = 64 MB
    GPUBuffer m_SharcResolvedBuf;       // SharcPackedData (float16_t4+2xuint) x SHARC_HASH_ENTRIES_NUM = 64 MB
    SharcBindlessIndices m_SharcIndices = {};

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcUpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcDebugPSO;

    // ----- Split Diffuse / Specular ReSTIR resources -----
    GPUBuffer m_DiffuseReservoirBuffer[2];       // Ping-pong diffuse reservoirs
    GPUBuffer m_SpecularReservoirBuffer[2];      // Ping-pong specular reservoirs
    GPUBuffer m_DiffuseReservoirIntermediate;    // Post-spatial diffuse
    GPUBuffer m_SpecularReservoirIntermediate;   // Post-spatial specular
    GPUBuffer m_DiffuseCandidateBuffer;          // DiffuseCandidate per pixel (RTDGI -> RTR)
    GPUTexture m_GIDiffuseIntermediate;          // GI resolved diffuse (raw float4: radiance, hitT)
    GPUTexture m_GISpecularIntermediate;         // GI resolved specular (raw float4: radiance, hitT)

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DiffuseTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SpecularTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DiffuseSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SpecularSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_GIResolveIntermediatesPSO;   // GI reservoir -> float4 intermediates
};
