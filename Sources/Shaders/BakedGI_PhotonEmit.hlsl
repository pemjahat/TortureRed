// =============================================================================
// BakedGI_PhotonEmit.hlsl — first-bounce photon map for one sun hour state.
//
// DEBUG-FIRST. The transport bake does NOT read the photon grid yet. The map
// exists so the "Photon Map" view can show WHERE the sun's first-bounce flux
// lands, before any decision to replace the gather's direct term with it.
//
// One thread = one SUN RAY. Rays leave a quad placed outside the scene on the
// sun side, perpendicular to the sun direction, sized to the scene's projected
// bounds (the CPU builds the quad; see BakedGI::RecordPhotonEmit). Each ray is
// a stratified jittered sample of one quad cell, so the deposits are uniform
// over the scene's PROJECTED area and every ray carries the same flux share.
//
// The deposit rule. The FIRST frontface hit along -s is a sunlit surface by
// construction: it faces the sun (n.s > 0) and nothing closer to the sun
// blocked it, so no shadow ray is needed — visibility is 1 exactly. A backface
// hit is the far side of one-sided geometry and scatters nothing toward the
// sun (the bake's own backface rule, BakedGI_Bake.hlsl), so it deposits
// nothing. Sky misses deposit nothing.
//
// Power. Unit-delta convention: the sun irradiance is 1. Each ray's tube has
// cross-section quadArea / N, and the cosine between irradiance and surface
// area cancels, so the reflected flux one photon stands for is
//
//     Phi = diffuseAlbedo * (quadArea / N)          N = raysX * raysY
//
// A future probe gather multiplies Phi by the live sun irradiance, the same
// way the fold scales the unit-delta response tables. Multiplying Phi by
// N / quadArea (FrameConstants::bakedGIPhotonFluxScale) recovers the deposit's
// diffuse albedo, which is what the debug view colours by.
//
// Storage. One grid cell for each probe cell (BakedGIFlatten), capacity
// BAKED_GI_PHOTON_CAP deposits, three float4 each:
//   rec0 = (position.xyz, Phi.r)
//   rec1 = (Phi.g, Phi.b, n.s, sun-ray hit distance)
//   rec2 = (surface normal.xyz, 0)
// The append counter counts every attempt and rises past the cap, so the CPU
// readout can report the dropped count. counts[probeCount] (the extra slot)
// counts rays whose hit fell outside the probe grid.
//
// Determinism. The seed is a hash of (quad cell, hour state), so the same
// state and the same scene produce the same map — the replay discipline the
// bake and the ray-debug pass already follow.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>          FrameCB : register(b0);
ConstantBuffer<BakedGIPhotonEmitParams> Emit     : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint rayIndex = dtid.x;
    const uint rayCount = Emit.raysX * Emit.raysY;
    if (rayIndex >= rayCount)
        return;

    const uint ix = rayIndex % Emit.raysX;
    const uint iy = rayIndex / Emit.raysX;

    // One deterministic stream for each (quad cell, state). The jitter draws
    // first; alpha-mask traversal draws after, exactly like the bake's streams.
    RNG rng;
    rng.state = pcg_hash(ix * 131u + iy * 7919u
                         + Emit.sunState * 0x85ebca6bu + 0x9e3779b9u);
    rng.inc   = 1;

    // Stratified jitter inside the ray's own quad cell.
    const float u = ((float)ix + next_float(rng)) / (float)Emit.raysX;
    const float v = ((float)iy + next_float(rng)) / (float)Emit.raysY;

    const float3 quadOrigin = float3(Emit.quadOriginX, Emit.quadOriginY, Emit.quadOriginZ);
    const float3 quadRight  = float3(Emit.quadRightX,  Emit.quadRightY,  Emit.quadRightZ);
    const float3 quadUp     = float3(Emit.quadUpX,     Emit.quadUpY,     Emit.quadUpZ);
    const float3 sunDir     = float3(Emit.sunDirX,     Emit.sunDirY,     Emit.sunDirZ);

    RayDesc ray;
    ray.Origin    = quadOrigin + quadRight * u + quadUp * v;
    ray.Direction = -sunDir; // from the sun side, toward the scene
    ray.TMin      = 0.001f;
    ray.TMax      = Emit.rayTMax;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    while (q.Proceed()) { PROCESS_ALPHA_MASK(q, rng); }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return; // the ray left the scene: no flux landed

    Surface surf;
    ResolveHitSurface(ray, q.CommittedRayT(), q.CommittedInstanceIndex(),
                      q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), surf);

    // Backface: the far side of one-sided geometry. The sun does not light it,
    // and shading it would deposit the wall's lit far side (the thin-wall leak,
    // the same rule the transport gather applies).
    if (dot(surf.normal, ray.Direction) >= 0.0f)
        return;

    // Cell of the deposit. Outside the probe grid: count and drop, so the
    // readout can report coverage holes instead of hiding them.
    const uint3 dims    = uint3(Emit.dimX, Emit.dimY, Emit.dimZ);
    const float3 gridMin = float3(Emit.gridMinX, Emit.gridMinY, Emit.gridMinZ);
    const float3 rel     = surf.worldPos - gridMin;
    const int3   cellI   = int3(floor(rel / Emit.spacing));
    RWStructuredBuffer<uint> counts = ResourceDescriptorHeap[Emit.countUAVIdx];
    if (any(cellI < int3(0, 0, 0)) || any(cellI >= int3(dims)))
    {
        InterlockedAdd(counts[Emit.probeCount], 1u);
        return;
    }

    const uint cell = BakedGIFlatten((uint3)cellI, dims);
    uint slot;
    InterlockedAdd(counts[cell], 1u, slot);
    if (slot >= BAKED_GI_PHOTON_CAP)
        return; // the counter keeps rising; the readout reports the drop

    // Phi = diffuseAlbedo * (quadArea / N) under the unit-delta sun. The
    // cosine cancels between the irradiance and the area the ray's tube
    // spreads over, so no per-deposit orientation term belongs here — the
    // (n.w)/r^2 solid-angle pullback is the GATHER's job, not the deposit's.
    const float  quadArea = length(cross(quadRight, quadUp));
    const float3 diffAlb  = surf.albedo * (1.0f - surf.metallic);
    const float3 power    = diffAlb * (quadArea / (float)rayCount);

    RWStructuredBuffer<float4> photons = ResourceDescriptorHeap[Emit.photonUAVIdx];
    const uint base = (cell * BAKED_GI_PHOTON_CAP + slot) * BAKED_GI_PHOTON_F4;
    photons[base + 0u] = float4(surf.worldPos, power.r);
    photons[base + 1u] = float4(power.g, power.b,
                                dot(surf.normal, sunDir), q.CommittedRayT());
    photons[base + 2u] = float4(surf.normal, 0.0f);
}
