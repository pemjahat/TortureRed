#define SHARC_ENABLE_DEBUG 1
#include "sharc/SharcCommon.h"
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>       FrameCB   : register(b0);
ConstantBuffer<BindlessIndices>       g_Indices : register(b1);
ConstantBuffer<SharcBindlessIndices>  g_Sharc   : register(b2);

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
    RWTexture2D<float4> indirectIrradiance = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    
    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, FrameCB, rng, surface, primaryRayT);
    if (!hasPrimaryHit) {
        indirectIrradiance[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    Reservoir r = tempReservoirs[pixelIndex];

#if SHARC_ENABLE_DEBUG
    if (FrameCB.sharcDebug)
    {
        // Visualize the SHaRC cache at the secondary hit point stored in the
        // reservoir (matching RTXGI Pathtracer debug pattern)
        SharcParameters sharcParams;
        sharcParams.gridParameters.cameraPosition  = FrameCB.irCacheCameraPosition.xyz;
        sharcParams.gridParameters.logarithmBase   = SHARC_GRID_LOGARITHM_BASE;
        sharcParams.gridParameters.sceneScale      = FrameCB.sharcSceneScale;
        sharcParams.gridParameters.levelBias       = 0.0f;
        sharcParams.hashMapData.capacity           = SHARC_HASH_ENTRIES_NUM;
        sharcParams.hashMapData.hashEntriesBuffer  = ResourceDescriptorHeap[g_Sharc.HashEntriesBufIdx];
        sharcParams.accumulationBuffer             = ResourceDescriptorHeap[g_Sharc.AccumulationBufIdx];
        sharcParams.resolvedBuffer                 = ResourceDescriptorHeap[g_Sharc.ResolvedBufIdx];
        sharcParams.radianceScale                  = 1e3f;
        sharcParams.enableAntiFireflyFilter        = false;

        SharcHitData sharcQuery;
        // Use secondary hit when reservoir is valid, else fall back to primary surface
        if (r.W > 0.0f)
        {
            sharcQuery.positionWorld = r.hitPos;
            sharcQuery.normalWorld   = r.hitNormal;
        }
        else
        {
            sharcQuery.positionWorld = surface.worldPos;
            sharcQuery.normalWorld   = surface.normal;
        }

        float3 debugColor = float3(0.0f, 0.0f, 0.0f);
        SharcGetCachedRadiance(sharcParams, sharcQuery, debugColor, true);
        indirectIrradiance[screenPos] = float4(debugColor, 1.0f);
        return;
    }
#endif // SHARC_ENABLE_DEBUG

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

    indirectIrradiance[screenPos] = float4(indirectLighting, 1.0f);
}
