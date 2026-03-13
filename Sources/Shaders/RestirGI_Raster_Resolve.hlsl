#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  FrameCB   : register(b0);
ConstantBuffer<BindlessIndices>  g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    RNG rng;
    seed_rng(rng, screenPos, FrameCB.frameIndex);

    // Accessing texture bindless
    StructuredBuffer<Reservoir> tempReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWTexture2D<float4> indirectLightingTex = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    
    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, FrameCB, rng, surface, primaryRayT);
    if (!hasPrimaryHit) {
        indirectLightingTex[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    Reservoir r = tempReservoirs[pixelIndex];

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

            // The reservoir stores continuation radiance from the sampled
            // first-bounce candidate; resolve applies the primary BRDF here.
            indirectLighting = (diffBRDF + specBRDF) * r.radiance * r.W * NdotL;
        }
    }

    indirectLightingTex[screenPos] = float4(indirectLighting, 1.0f);
}
