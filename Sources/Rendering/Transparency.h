#pragma once

#include "Graphics/GraphicsTypes.h"

class Model;
class Renderer;

// -----------------------------------------------------------------------------
// Transparency
//
// Forward transparency pass (moved out of Application::Render()). Runs after
// the Lighting pass and before TAA so that transparent geometry is temporally
// accumulated. In the TAA path it renders into the internal-res HDR
// intermediate texture (depth-tested against the G-buffer depth, also at
// internal resolution); in the non-TAA path it renders directly to the back
// buffer at output resolution.
//
// This pass owns no persistent GPU resources of its own (it reads/writes
// resources owned by GBufferPass/Renderer); PSO creation for the transparent
// PSOs (LDR/HDR) stays centralized in Renderer's shared CreatePipelineState()
// per the migration plan.
// -----------------------------------------------------------------------------
class Transparency
{
public:
    void Execute(ID3D12GraphicsCommandList* cmdList, Model* model, Renderer* renderer,
                 const DirectX::BoundingFrustum& frustum, bool rasterTaaActive,
                 uint32_t outputWidth, uint32_t outputHeight,
                 ID3D12PipelineState* transparentPSO, ID3D12PipelineState* transparentHdrPSO);
};
