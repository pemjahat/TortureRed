#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;
class Renderer;

// -----------------------------------------------------------------------------
// GBufferPass
//
// Owns the G-Buffer render targets (albedo/normal/material/depth) and the
// depth pre-pass + G-buffer raster pass Execute logic (moved out of
// Application::Render()). PSO creation stays centralized in Renderer's shared
// CreatePipelineState() per the migration plan; Execute() receives the PSOs
// it needs as parameters.
//
// Named "GBufferPass" (rather than "GBuffer") to avoid colliding with the
// shared `struct GBuffer` resource-bundle type defined in Graphics/GraphicsTypes.h.
// -----------------------------------------------------------------------------
class GBufferPass
{
public:
    void CreateResources(uint32_t w, uint32_t h);

    GBuffer& GetGBuffer() { return m_GBuffer; }

    // Depth pre-pass (optional) + G-buffer raster pass. Model handles its own
    // frustum-culled ExecuteIndirect draw via Model::Render().
    void Execute(ID3D12GraphicsCommandList* cmdList, Model* model, Renderer* renderer,
                 const DirectX::BoundingFrustum& frustum, bool enableDepthPrePass,
                 ID3D12PipelineState* depthPrePassPSO, ID3D12PipelineState* gbufferPSO,
                 ID3D12PipelineState* gbufferWritePSO);

private:
    GBuffer m_GBuffer; // Graphics/GraphicsTypes.h struct: albedo/normal/material/depth
};
