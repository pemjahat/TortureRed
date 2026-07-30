#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;
class Renderer;

// -----------------------------------------------------------------------------
// DeferredLighting
//
// Owns:
//  - Lights structured buffer + Light LUT buffer (O(1) importance sampling)
//    for local-light selection.
//  - The Lighting / FullScreenDebug fullscreen-triangle pass Execute logic
//    (moved out of Application::Render()). Reads GBuffer SRVs plus
//    FinalDiffuse/FinalSpecular (written by RestirDI/RestirGI) and writes
//    either to the internal-res HDR target (TAA path) or directly to the
//    back buffer (non-TAA path).
//
// PSO creation stays centralized in Renderer's shared CreatePipelineState()
// per the migration plan; Execute() receives the PSOs/textures it needs via
// the Renderer pointer (same pattern as GBufferPass::Execute).
// -----------------------------------------------------------------------------
class DeferredLighting
{
public:
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // ---- Lights buffer ----
    void CreateLightsBuffer();
    void UpdateLightsBuffer(const std::vector<LightConstants>& lights);
    D3D12_GPU_VIRTUAL_ADDRESS GetLightsBufferGPUAddress() const { return m_LightsBuffer.gpuAddress; }
    UINT GetLightsDescriptorIndex() const { return (UINT)m_LightsBuffer.srvIndex; }

    // ---- Light LUT buffer for O(1) importance sampling ----
    void CreateLightLUTBuffer();
    void UpdateLightLUTBuffer(const std::vector<LightConstants>& lights);
    UINT GetLightLUTDescriptorIndex() const { return (UINT)m_LightLUTBuffer.srvIndex; }
    D3D12_GPU_VIRTUAL_ADDRESS GetLightLUTBufferGPUAddress() const { return m_LightLUTBuffer.gpuAddress; }

    // Lighting Pass (or FullScreenDebug Pass when debugActive is true).
    void Execute(ID3D12GraphicsCommandList* cmdList, Renderer* renderer, Model* model,
                 const FrameConstants& frame, bool rasterTaaActive, bool debugActive,
                 bool debugShadowMap, uint32_t outputWidth, uint32_t outputHeight);

private:
    // ----- Resources -----
    GPUBuffer m_LightsBuffer;
    GPUBuffer m_LightLUTBuffer; // LUT for O(1) importance sampling

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_LightingPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_LightingHdrPSO; // Lighting PSO targeting R16G16B16A16_FLOAT (HDR, no tonemapping)

    static constexpr UINT LIGHT_LUT_RESOLUTION = 256;
    UINT m_MaxLights = 256;
};
