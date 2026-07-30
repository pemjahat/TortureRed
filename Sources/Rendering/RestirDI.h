#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;
struct FrameConstants;

// -----------------------------------------------------------------------------
// RestirDI
//
// Owns the ReSTIR direct-lighting (DI) reservoirs/PSOs: combined initial
// sampling + temporal resampling, spatial resampling, and split diffuse/
// specular shading. Writes its split-shaded result into the shared
// FinalDiffuse/FinalSpecular interchange textures (owned by Renderer) via the
// generic NrdStoreShadingOutput bridge PSO (also shared with RestirGI, stays
// in Renderer for now).
//
// NRD triggering and the FullScreenDebug SRV-transition tail are cross-cutting
// concerns shared with RestirGI/Denoise, so they stay orchestrated in
// Renderer::DispatchRestirDI immediately after calling Execute().
// -----------------------------------------------------------------------------
class RestirDI
{
public:
    void CreateResources(uint32_t internalWidth, uint32_t internalHeight);
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Runs the DI temporal -> spatial -> split-shade -> store-shading-output passes.
    // finalDiffuseTex/finalSpecularTex are overwritten (isFirstPass=1) with the DI result.
    void Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                 Model* model, const FrameConstants& frame,
                 D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, D3D12_GPU_VIRTUAL_ADDRESS tlasGPUAddress,
                 D3D12_GPU_VIRTUAL_ADDRESS lightsBufferAddress, D3D12_GPU_VIRTUAL_ADDRESS lightLUTBufferAddress,
                 GPUTexture& fullScreenDebugTex, GPUTexture& finalDiffuseTex, GPUTexture& finalSpecularTex,
                 ID3D12PipelineState* nrdStoreShadingOutputPSO,
                 uint32_t internalWidth, uint32_t internalHeight);

    GPUTexture& GetDIDiffuseIntermediate()  { return m_DIDiffuseIntermediate; }
    GPUTexture& GetDISpecularIntermediate() { return m_DISpecularIntermediate; }

private:
    // ----- Resources -----
    GPUBuffer  m_DIReservoirBuffer[2];       // Ping-pong DI reservoirs (DIRreservoir per pixel)
    GPUBuffer  m_DIReservoirIntermediate;    // Post-spatial DI reservoirs
    GPUTexture m_DIDiffuseIntermediate;      // Split DI diffuse (NRD-normalized float4)
    GPUTexture m_DISpecularIntermediate;     // Split DI specular (NRD-normalized float4)
    int        m_CurrentDIReservoirIndex = 0;

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirDITemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirDISpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirDISplitShadePSO;  // Split diffuse/specular shade
};
