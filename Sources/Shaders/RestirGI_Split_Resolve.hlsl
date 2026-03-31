// RestirGI_Split_Resolve.hlsl — Split resolve pass
// Reads both diffuse and specular reservoir streams independently,
// applies the correct BRDF lobe to each, and composites into a single output.

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

    // InputIdx0 = diffuse reservoir (post-spatial)
    // InputIdx1 = specular reservoir (post-spatial)
    // OutputIdx0 = indirect lighting texture
    StructuredBuffer<Reservoir> diffuseReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    StructuredBuffer<Reservoir> specularReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx1];
    RWTexture2D<float4> indirectLightingTex = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, FrameCB, rng, surface, primaryRayT);
    if (!hasPrimaryHit) {
        indirectLightingTex[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float3 indirectLighting = 0.0f;

    // Evaluate diffuse reservoir with diffuse BRDF only
    Reservoir rDiffuse = diffuseReservoirs[pixelIndex];
    if (rDiffuse.W > 0.0f) {
        float3 L = normalize(rDiffuse.hitPos - surface.worldPos);
        float NdotL = max(0.0f, dot(surface.normal, L));
        if (NdotL > 0.0f) {
            float3 diffBRDF, specBRDF;
            EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffBRDF, specBRDF);
            indirectLighting += diffBRDF * rDiffuse.radiance * rDiffuse.W * NdotL;
        }
    }

    // Evaluate specular reservoir with specular BRDF only
    if (FrameCB.enableIndirectSpecular) {
        Reservoir rSpecular = specularReservoirs[pixelIndex];
        if (rSpecular.W > 0.0f) {
            float3 L = normalize(rSpecular.hitPos - surface.worldPos);
            float NdotL = max(0.0f, dot(surface.normal, L));
            if (NdotL > 0.0f) {
                float3 diffBRDF, specBRDF;
                EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffBRDF, specBRDF);
                indirectLighting += specBRDF * rSpecular.radiance * rSpecular.W * NdotL;
            }
        }
    }

    indirectLightingTex[screenPos] = float4(min(indirectLighting, 10.0f), 1.0f);
}
