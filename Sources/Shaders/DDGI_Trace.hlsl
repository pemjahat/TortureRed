// =============================================================================
// DDGI_Trace.hlsl — TraceRaysCS: cast N rays for each probe, store the result.
//
// One thread for each (probe, ray) pair. The reference uses a ray-generation
// shader with a state object; TortureRed has neither, so this is a compute
// shader with an inline RayQuery — the pattern of BakedGI_Bake.hlsl and
// PathTracer.hlsl. The output buffer has the same layout as the reference's
// typed ray buffer: float4(radiance.rgb, signedDepth).
//
// PROBE RAY CONTENT (plan task023 decision 4, reading A):
//   miss        -> radiance = sky cubemap, depth = 2 x spacing (the sky clamp)
//   frontface   -> radiance = albedo/pi * (sun at the hit with a shadow ray)
//                            + albedo/pi * small-albedo * atlas feedback
//                  depth    = min(hitT, 2 x spacing)
//   backface    -> radiance = 0, depth = -hitT * 0.2
// The sun is evaluated AT THE HIT, not at the probe (the retrace-sun
// discipline). The probe therefore stores indirect light only, and the
// receiver's direct sun stays with the runtime-exact path.
//
// The atlas-feedback term closes the multi-bounce loop across frames: a hit
// reads the PREVIOUS frame's irradiance atlas. The trace therefore reads the
// history copy, never the target the resolve is about to write.
//
// REQUIRES in scope: FrameConstants (b0, for the sky cubemap index),
// DDGIParams (b2), and the tracing resources CommonTracing.hlsl declares.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<DDGIParams>     DDGI    : register(b2);

// DDGI.hlsli includes BakedGI.hlsli, which references FrameCB — include after
// the declarations above.
#include "DDGI.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint3 dims       = uint3(DDGI.dimX, DDGI.dimY, DDGI.dimZ);
    const uint  probeCount = dims.x * dims.y * dims.z;
    const uint  probeIdx   = dtid.x;
    const uint  rayIndex   = dtid.y;
    if (probeIdx >= probeCount || rayIndex >= DDGI.numRaysPerProbe)
        return;

    const uint3 cell = DDGIUnflatten3D(probeIdx, dims);
    const DDGIVolumeGpu vol = DDGIGetVolumeFromParams(
        float3(DDGI.gridMinX, DDGI.gridMinY, DDGI.gridMinZ), DDGI.spacing, dims,
        DDGI.numRaysPerProbe, DDGI.maxRaysPerProbe,
        DDGI.irradianceHistorySRVIdx, DDGI.depthHistorySRVIdx,
        DDGI.probeStateSRVIdx, DDGI.probeOffsetSRVIdx);

    // An inactive probe only traces the stable rays, which are the ones
    // UpdateProbeStatesCS uses to decide whether to re-activate it. The
    // remaining slots are never read for an inactive probe.
    if (!DDGIIsProbeActive(vol, probeIdx) && rayIndex >= DDGI_NUM_STABLE_RAYS)
        return;

    const float3 probePos = DDGIProbePosition(vol, cell);
    const float  maxDepth = DDGI.spacing * 2.0f; // the reference sky clamp
    const float3x3 randomRotation = DDGIAngleAxis3x3(DDGI.randomAngle,
                                     float3(DDGI.randomVectorX, DDGI.randomVectorY, DDGI.randomVectorZ));
    const float3 rayDir = DDGIGetRayDirection(rayIndex, DDGI.numRaysPerProbe, randomRotation);

    // Deterministic seed from (probe, ray) only: the stable rays must produce
    // the same depth every frame, or probe-state classification flickers on
    // alpha-masked geometry. The rotating rays vary by direction instead.
    RNG rng;
    rng.state = pcg_hash(probeIdx * 131u + rayIndex * 7919u + 0x9e3779b9u);
    rng.inc   = 1;

    RayDesc ray;
    ray.Origin    = probePos;
    ray.Direction = rayDir;
    ray.TMin      = 0.001f;
    ray.TMax      = 1e4f;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) { PROCESS_ALPHA_MASK(q, rng); }

    float  depth    = maxDepth;
    float3 radiance = float3(0.0f, 0.0f, 0.0f);

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        const float hitT = q.CommittedRayT();
        depth = min(hitT, maxDepth);

        Surface surf;
        ResolveHitSurface(ray, hitT, q.CommittedInstanceIndex(), q.CommittedPrimitiveIndex(),
                          q.CommittedTriangleBarycentrics(), surf);

        if (dot(surf.normal, rayDir) < 0.0f)
        {
            // --- Frontface: the sun at the hit, then the atlas feedback. ---
            const float3 sunE   = float3(DDGI.sunIrradianceX, DDGI.sunIrradianceY, DDGI.sunIrradianceZ);
            const float3 sunDir = normalize(float3(DDGI.sunDirX, DDGI.sunDirY, DDGI.sunDirZ));
            const float3 diffAlbedo = surf.albedo * (1.0f - surf.metallic);

            const float NdotL = max(0.0f, dot(surf.normal, sunDir));
            float Vhit = 0.0f;
            if (NdotL > 0.0f)
            {
                RayDesc shadowRay;
                shadowRay.Origin    = surf.worldPos + surf.normal * 0.001f;
                shadowRay.Direction = sunDir;
                shadowRay.TMin      = 0.001f;
                shadowRay.TMax      = 1e4f;
                RayQuery<RAY_FLAG_NONE> sq;
                sq.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
                while (sq.Proceed()) { PROCESS_ALPHA_MASK(sq, rng); }
                Vhit = (sq.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;
            }

            // The feedback albedo is clamped (the reference does the same): the
            // frame-to-frame loop must not diverge on a bright albedo.
            const float3 feedback = DDGIQueryVolume(vol, surf.worldPos, surf.normal, rayDir).rgb;

            radiance = diffAlbedo / PI * (NdotL * Vhit * sunE);
            radiance += min(diffAlbedo, 0.9f.xxx) / PI * feedback;
        }
        else
        {
            // Backface: negative depth so UpdateProbeStatesCS can push the probe
            // through the surface during relocation.
            depth *= DDGI_BACKFACE_DEPTH_MULTIPLIER;
        }
    }
    else
    {
        radiance = SampleSky(rayDir, FrameCB.skyCubemapIndex);
    }

    RWStructuredBuffer<float4> rayHits = ResourceDescriptorHeap[DDGI.rayHitUAVIdx];
    rayHits[probeIdx * DDGI.maxRaysPerProbe + rayIndex] = float4(radiance, depth);
}
