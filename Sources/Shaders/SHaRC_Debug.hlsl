// SHaRC_Debug.hlsl
// Shared debug visualization pass — two modes selected by g_Frame.sharcDebug:
//
//   1 — SHaRC Output : queries SharcGetCachedRadiance at the primary hit,
//       producing the same blocky voxel-boundary visualization as RTXGI's
//       "Enable Debug" checkbox.
//
//   2 — Bounce Heatmap : traces secondary bounces from the primary hit and
//       maps the geometric bounce count to a cyan→red color ramp so you can
//       see path depth distribution across the screen.
//
// Dispatched at full resolution ONLY when sharcDebug != 0, after SHaRC
// Update+Resolve but before ReSTIR temporal/spatial passes.
//
// Writes debugColor to OutputIdx0 (FullScreenDebugTex, R16G16B16A16_FLOAT).
// FullScreenDebug.hlsl reads this texture as a single InputIdx0 and outputs
// directly to the screen — no NRD material factor modulation, no BSDF
// evaluation, no shadow rays.

#define SHARC_ENABLE_DEBUG 1
#include "SharcCommon.h"
#include "CommonTracing.hlsl"
#include "SHaRC_Integration.hlsl"

ConstantBuffer<FrameConstants>       g_Frame   : register(b0);
ConstantBuffer<BindlessIndices>      g_Indices : register(b1);
ConstantBuffer<SharcBindlessIndices> g_Sharc   : register(b2);

// Maps a bounce count to RTXGI's BounceHeatmap palette (PathtracerUtils.hlsli):
//   0  → blue  (first secondary ray missed sky — no geometry bounces)
//   1  → green (one secondary geometry hit)
//   2+ → red   (two or more secondary geometry hits)
// Counting semantics are identical to RTXGI: bounce++ only fires after a geometry
// hit, break on sky miss does not increment — so value N means N geometry hits.
float3 BounceCountColor(int n)
{
    switch (n)
    {
    case 0:  return float3(0.0f, 0.0f, 1.0f); // blue
    case 1:  return float3(0.0f, 1.0f, 0.0f); // green
    default: return float3(1.0f, 0.0f, 0.0f); // red (2+)
    }
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos  = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    RWTexture2D<float4> outDebug = ResourceDescriptorHeap[g_Indices.OutputIdx0]; // FullScreenDebugTex

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex);

    // Init sharc parameter
    SharcParameters sharcParams;
    sharcParams.gridParameters.cameraPosition  = g_Frame.cameraPosition.xyz;
    sharcParams.gridParameters.logarithmBase   = SHARC_GRID_LOGARITHM_BASE;
    sharcParams.gridParameters.sceneScale      = g_Frame.sharcSceneScale;
    sharcParams.gridParameters.levelBias       = 0.0f;
    sharcParams.hashMapData.capacity           = SHARC_HASH_ENTRIES_NUM;
    sharcParams.hashMapData.hashEntriesBuffer  = ResourceDescriptorHeap[g_Sharc.HashEntriesBufIdx];
    sharcParams.accumulationBuffer             = ResourceDescriptorHeap[g_Sharc.AccumulationBufIdx];
    sharcParams.resolvedBuffer                 = ResourceDescriptorHeap[g_Sharc.ResolvedBufIdx];
    sharcParams.radianceScale                  = 1e3f;
    sharcParams.enableAntiFireflyFilter        = false;

    float3 throughput = 1;
    int bounceCount = 0;
    float3 debugColor = float3(0, 0, 0);

    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, g_Frame, rng, surface, primaryRayT);

    if (hasPrimaryHit) {
        bool isPathDiffuse = false;
        float cumulativeRoughness = surface.roughness;

        if (g_Frame.sharcDebug == 1)
        {
            SharcHitData sharcQuery;
            sharcQuery.positionWorld = surface.worldPos;
            sharcQuery.normalWorld   = surface.normal;

            SharcGetCachedRadiance(sharcParams, sharcQuery, debugColor, true);
        }

        for (bounceCount = 1; bounceCount < 4; bounceCount++) {
            if (all(throughput <= 0.0f)) break;

            // Sample bounce
            float3 nextDir, nextThroughput;
            float next_pdf;
            SampleIndirectRay(surface.normal, surface.viewDir, 
                surface.albedo, surface.metallic, surface.roughness, 
                rng, nextDir, nextThroughput, next_pdf, isPathDiffuse, 
                g_Frame.enableIndirectSpecular != 0);                          

            throughput *= nextThroughput;

            // Russian Roulette
            if (bounceCount > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (next_float(rng) > p) break;
                throughput /= p;
            }

            RayDesc ray;
            ray.Origin    = surface.worldPos + surface.normal * 0.001f;
            ray.Direction = nextDir;
            ray.TMin      = 0.01f;
            ray.TMax      = 1000.0f;

            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
            while (q.Proceed()) {
                PROCESS_ALPHA_MASK(q, rng);
            }

            if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            {
                // Hit sky
                break;
            }

            // --- Decode triangle geometry ---
            Surface hitSurf;
            ResolveHitSurface(ray, q.CommittedRayT(), q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), hitSurf);

            SharcHitData sharcQuery;
            sharcQuery.positionWorld = hitSurf.worldPos;
            sharcQuery.normalWorld   = hitSurf.normal;

            float3 sampleRadiance;
            float pathRoughness = isPathDiffuse ? 1.0f : cumulativeRoughness;
            if (IsSharcQueryValid(hitSurf.worldPos, q.CommittedRayT(), pathRoughness, sharcParams)
                && SharcGetCachedRadiance(sharcParams, sharcQuery, sampleRadiance, false))
                break;
            // HashGridParameters gridParameters;
            // gridParameters.cameraPosition = g_Frame.irCacheCameraPosition.xyz;
            // gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
            // gridParameters.sceneScale = g_Frame.sharcSceneScale;
            // gridParameters.levelBias = 0.0f;
            // debugColor = HashGridDebugColoredHash(hitPos, worldNormal, gridParameters);
            // break;

            // Advance primary surface to this hit for the next bounce
            surface = hitSurf;

            cumulativeRoughness += surface.roughness;
        }
    }

    if (g_Frame.sharcDebug == 1)
    {
        outDebug[screenPos] = float4(debugColor, 1.0f);
    }
    else
    {
        outDebug[screenPos] = float4(BounceCountColor(bounceCount), 1.0f);
    }
}
