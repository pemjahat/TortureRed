// SHaRC_Update.hlsl
// Compiled with -DSHARC_UPDATE=1 -DSHARC_PROPAGATION_DEPTH=4
//
// Dispatched on a coarse tile grid (8x8 threads per group). Each thread picks
// one full-resolution pixel inside its SHARC_UPDATE_DOWNSCALE x
// SHARC_UPDATE_DOWNSCALE tile, varying the sample position over time so the
// update pass covers the full screen across frames.
// SharcResolveEntry (in SHaRC_Resolve.hlsl) blends the accumulation into the
// resolved buffer with EMA and resets the accumulation for next frame.

#include "SharcCommon.h"
#include "CommonTracing.hlsl"

#ifndef SHARC_UPDATE_DOWNSCALE
#define SHARC_UPDATE_DOWNSCALE 5
#endif

ConstantBuffer<FrameConstants>       g_Frame   : register(b0);
ConstantBuffer<BindlessIndices>      g_Indices : register(b1);
ConstantBuffer<SharcBindlessIndices> g_Sharc   : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

SharcParameters BuildSharcParams()
{
    SharcParameters p;
    p.gridParameters.cameraPosition  = g_Frame.cameraPosition.xyz;
    p.gridParameters.logarithmBase   = SHARC_GRID_LOGARITHM_BASE;
    p.gridParameters.sceneScale      = g_Frame.sharcSceneScale;
    p.gridParameters.levelBias       = 0.0f;
    p.hashMapData.capacity           = SHARC_HASH_ENTRIES_NUM;
    p.hashMapData.hashEntriesBuffer  = ResourceDescriptorHeap[g_Sharc.HashEntriesBufIdx];
    p.accumulationBuffer             = ResourceDescriptorHeap[g_Sharc.AccumulationBufIdx];
    p.resolvedBuffer                 = ResourceDescriptorHeap[g_Sharc.ResolvedBufIdx];
    p.radianceScale                  = 1e3f;
    p.enableAntiFireflyFilter        = false;
    return p;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 tilePos = DTid.xy;
    uint tileSampleCount = SHARC_UPDATE_DOWNSCALE * SHARC_UPDATE_DOWNSCALE;
    uint tilePhase = pcg_hash(tilePos.x + tilePos.y * 65536u) % tileSampleCount;
    uint sampleIndex = (g_Frame.frameIndex + tilePhase) % tileSampleCount;
    uint2 screenPos = tilePos * SHARC_UPDATE_DOWNSCALE
        + uint2(sampleIndex % SHARC_UPDATE_DOWNSCALE, sampleIndex / SHARC_UPDATE_DOWNSCALE);
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex);

    // Trace primary surface from GBuffer (same ray as the camera)
    Surface surface;
    float primaryRayT;
    if (!TracePrimarySurface(screenPos, launchDims, g_Frame, rng, surface, primaryRayT)) return;

    SharcParameters sharcParams = BuildSharcParams();

    SharcState sharcState;
    SharcInit(sharcState);

    float3 throughput = 1;

    [loop]
    for (int bounce = 1; bounce < SHARC_PROPAGATION_DEPTH; ++bounce)
    {
        // Sample next bounce direction + per-step throughput weight
        float3 rayDir, segmentThroughput;
        float  pdf;
        bool   isDiffuse;
        SampleIndirectRay(surface.normal, surface.viewDir,
                          surface.albedo, surface.metallic, surface.roughness,
                          rng, rayDir, segmentThroughput, pdf, isDiffuse,
                          g_Frame.enableIndirectSpecular != 0);

        if (all(segmentThroughput <= 0.0f)) break;

        throughput *= segmentThroughput;

        // Russian Roulette
        if (bounce > 2) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (next_float(rng) > p) break;
            throughput /= max(p, 1e-3f);
            segmentThroughput /= max(p, 1e-3f);
        }

        RayDesc ray;
        ray.Origin    = surface.worldPos + surface.normal * 0.001f;
        ray.Direction = rayDir;
        ray.TMin      = 0.01f;
        ray.TMax      = 1000.0f;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
        while (q.Proceed()) { PROCESS_ALPHA_MASK(q, rng); }

        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            // Sky hit — sample sky radiance and propagate back through stored vertices
            float3 skyRadiance = SampleSky(rayDir, g_Frame.skyCubemapIndex);
            SharcUpdateMiss(sharcParams, sharcState, skyRadiance);
            break;
        }

        // --- Decode triangle geometry ---
        Surface hitSurf;
        ResolveHitSurface(ray, q.CommittedRayT(), q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), hitSurf, 0.15f);

        // --- Direct lighting at bounce hit ---
        float3 directLighting = GetDirectLightingHybrid(
            hitSurf.worldPos, hitSurf.normal, hitSurf.viewDir,
            hitSurf.albedo, hitSurf.metallic, hitSurf.roughness,
            g_Scene, g_Lights, g_Frame.numLights, g_Frame, true, rng);

        // --- Deposit sample into SHaRC hash table ---
        SharcHitData sharcHit;
        sharcHit.positionWorld = hitSurf.worldPos;
        sharcHit.normalWorld   = hitSurf.normal;

        bool continueTracing = SharcUpdateHit(sharcParams, sharcState, sharcHit,
                                              directLighting, next_float(rng));

        // SharcUpdateHit returns false when cache resampling terminates the path
        if (!continueTracing) break;

        // Scale stored sample weights by this bounce's BSDF throughput
        SharcSetThroughput(sharcState, segmentThroughput);

        // Advance primary surface to this hit for the next bounce
        surface = hitSurf;
    }
}
