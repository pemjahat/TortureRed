#pragma once

#include <memory>
#include "Graphics/GraphicsTypes.h"

namespace nrd { struct Integration; }
struct FrameConstants;

// -----------------------------------------------------------------------------
// Denoise
//
// Owns the NRD (NVIDIA Real-time Denoisers) RELAX integration used to denoise
// the split diffuse/specular lighting produced by RestirDI/RestirGI (raster
// indirect). Those callers still live in Renderer.cpp; they write their result
// into the shared Final*Tex interchange textures (owned by Renderer) via the
// NrdStoreShadingOutput PSO (also still in Renderer, since it's a generic
// 2-input/2-output bridge shared by both DI and GI paths), then call
// Denoise::Execute() to run NRD in-place on those textures.
//
// Resource/PSO creation happens once via CreateResources()/CreatePipelines();
// per-frame work happens via Execute().
// -----------------------------------------------------------------------------
class Denoise
{
public:
    // Declared here, defined in Denoise.cpp where nrd::Integration is a complete
    // type — required because std::unique_ptr<nrd::Integration>'s destructor needs
    // the full type, and Denoise is used by-value inside Renderer (Renderer.h only
    // has a forward declaration of nrd::Integration).
    Denoise();
    ~Denoise();

    bool CreateResources(uint32_t internalWidth, uint32_t internalHeight);
    void OnResolutionChanged(uint32_t w, uint32_t h);
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Lazily creates the NRD integration instance (safe to call every frame).
    bool Initialize(ID3D12Device* device, ID3D12CommandQueue* commandQueue, uint32_t internalWidth, uint32_t internalHeight);
    void Shutdown();

    // Runs NRD RELAX diffuse+specular denoising in-place on finalDiffuseTex/finalSpecularTex
    // (read for NrdPackNoise input, then overwritten with the denoised result).
    // Returns false if NRD/PSOs are not ready or a step failed (caller should treat as "did not run").
    bool Execute(ID3D12GraphicsCommandList* cmdList, ID3D12CommandAllocator* cmdAllocator,
                 ID3D12RootSignature* rootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                 const FrameConstants& frame, GPUTexture& finalDiffuseTex, GPUTexture& finalSpecularTex,
                 uint32_t internalWidth, uint32_t internalHeight);

    // Motion vectors are written by Renderer::GenerateMotionVectors() (TAA module) and
    // consumed both by NRD (IN_MV) and by the naive TSR reprojection pass, so the
    // texture is exposed here rather than duplicated.
    GPUTexture& GetMotionVectorsTex() { return m_NrdMotionVectorsTex; }
    GPUTexture& GetDenoisedDiffuseTex()  { return m_NrdDenoisedDiffuseTex; }
    GPUTexture& GetDenoisedSpecularTex() { return m_NrdDenoisedSpecularTex; }

    bool WasActiveLastFrame() const { return m_NrdWasActiveLastFrame; }
    void SetWasActiveLastFrame(bool v) { m_NrdWasActiveLastFrame = v; }
    bool IsInitialized() const { return m_NrdInitialized; }

private:
    // ----- NRD Textures -----
    GPUTexture m_NrdMotionVectorsTex;
    GPUTexture m_NrdNormalRoughnessTex;
    GPUTexture m_NrdViewZTex;
    GPUTexture m_NrdRelaxDiffuseTex;       // RELAX-packed input for NRD denoiser
    GPUTexture m_NrdRelaxSpecularTex;      // RELAX-packed input for NRD denoiser
    GPUTexture m_NrdDenoisedDiffuseTex;
    GPUTexture m_NrdDenoisedSpecularTex;
    GPUTexture m_NrdValidationTex;

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdPrepareGuidesPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdCompositePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdPackNoisePSO;

    // ----- NRD Integration lifetime -----
    std::unique_ptr<nrd::Integration> m_NrdIntegration;
    bool m_NrdInitialized = false;
    bool m_NrdWasActiveLastFrame = false; // Tracks whether NRD ran last frame; used to force RESTART on re-enable
};
