#pragma once

#include "Graphics/GraphicsTypes.h"

struct FrameConstants;
struct GBuffer;

// -----------------------------------------------------------------------------
// TAA
//
// Owns the naive TSR (Temporal Super Resolution) reproject+resolve pipeline
// and the motion-vector generation pass that feeds both TAA and NRD (via the
// Denoise-owned motion vectors texture, passed in by reference so both
// consumers write/read the same resource without duplicating it).
//
// Resource/PSO creation happens once via CreateResources()/CreatePipelines();
// per-frame work happens via GenerateMotionVectors()/Execute().
// -----------------------------------------------------------------------------
class TAA
{
public:
    void CreateResources(uint32_t outputW, uint32_t outputH, uint32_t internalW, uint32_t internalH);
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Writes into motionVectorsTex (owned by Denoise, shared with NRD's IN_MV input).
    void GenerateMotionVectors(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                                D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, GBuffer& gbuffer,
                                GPUTexture& motionVectorsTex, uint32_t internalWidth, uint32_t internalHeight);

    // Reproject + resolve. Reads motionVectorsTex (written by GenerateMotionVectors).
    void Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                 D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, const FrameConstants& frame,
                 const GPUTexture& inputColor, GBuffer& gbuffer, GPUTexture& motionVectorsTex);

    GPUTexture& GetOutputTex() { return m_TaaOutputTex; }
    bool IsEnabled() const { return m_TaaEnabled; }

private:
    bool m_TaaEnabled = false;
    int  m_TaaHistoryIndex = 0; // Ping-pong index (0 or 1)
    GPUTexture m_TaaHistoryTex[2];           // Output-res: rgb + coverage
    GPUTexture m_TaaReprojectedHistoryTex;   // Output-res: reprojected history
    GPUTexture m_TaaClosestVelocityTex;      // Output-res: dilated closest velocity
    GPUTexture m_TaaOutputTex;               // Output-res: final TAA output

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NaiveTsrReprojectPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NaiveTsrResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MotionVectorsPSO;
};
