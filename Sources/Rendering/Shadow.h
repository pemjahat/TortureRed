#pragma once

#include "Graphics/GraphicsTypes.h"

// -----------------------------------------------------------------------------
// Shadow
//
// Owns the shadow-map depth texture resource. Note: PSO creation for the
// shadow depth pass (m_ShadowPSO) stays centralized in Renderer's shared
// CreatePipelineState() function per the migration plan (it shares a root
// signature and PSO-desc pattern with DepthPrePass/GBuffer/Lighting/Debug/
// ProbeSphereDebug), so only the resource itself lives here.
//
// NOTE: The current renderer uses ray-traced shadows (TraceRayInline in
// Lighting.hlsl/Forward.hlsl/DebugShadow.hlsl) as the primary shadowing
// technique; this shadow-map texture/PSO exist but no render pass currently
// writes to or samples this texture. Kept as-is (no behavior change) during
// this migration pass.
// -----------------------------------------------------------------------------
class Shadow
{
public:
    bool CreateResources();
    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);
    
    GPUTexture& GetShadowMap() { return m_ShadowMap; }

private:
    static constexpr UINT SHADOW_MAP_SIZE = 2048;

    // ----- Resources -----
    GPUTexture m_ShadowMap;

    // ----- PSOs -----
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ShadowPSO;
};
