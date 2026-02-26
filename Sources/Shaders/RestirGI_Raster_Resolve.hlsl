#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// Outputs
RWTexture2D<float4> g_IndirectLightingTex : register(u1);
RWStructuredBuffer<Reservoir> g_ReservoirFinal : register(u3);  // Spatial output (final reservoir)

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims;
    g_IndirectLightingTex.GetDimensions(launchDims.x, launchDims.y);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    RNG rng;
    seed_rng(rng, screenPos, FrameCB.frameIndex);

    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, FrameCB, rng, surface, primaryRayT);
    if (!hasPrimaryHit) {
        g_IndirectLightingTex[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    Reservoir r = g_ReservoirFinal[pixelIndex];
    
    float3 indirectLighting = 0.0f;

    if (r.W > 0.0f) {
        float3 L = normalize(r.hitPos - surface.worldPos);
        float NdotL = max(0.0f, dot(surface.normal, L));
        
        if (NdotL > 0.0f) {
            float3 diffBRDF, specBRDF;
            EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffBRDF, specBRDF);
            
            if (!FrameCB.enableIndirectSpecular) {
                specBRDF = 0.0f;
            }
            
            indirectLighting = (diffBRDF + specBRDF) * r.radiance * r.W * NdotL;
        }
    }

    g_IndirectLightingTex[screenPos] = float4(indirectLighting, 1.0f);
}
