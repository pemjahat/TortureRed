// =============================================================================
// BakedGI_Bake.hlsl — step-1 transport bake .
//
// One thread per (probe, delta direction W_i):
//   - Validity: READ from the meta bits classified by the visibility pass
//     (DDGI criteria — >= 25% backfaces among 32 stable fibonacci rays;
//     see BakedGI_Visibility.hlsl). Dead probes' responses are zeroed.
//   - Direct-delta visibility V_i: one shadow ray toward W_i (stored in slot
//     [9].x; the update pass uses it for the occluded first-bounce sky term).
//   - INDIRECT-ONLY directional-irradiance SH9 RGB response R_i to a unit
//     delta light at W_i: uniform-sphere gather rays; each hit's outgoing
//     radiance = diffuseAlbedo/pi * ( exact direct delta at the hit  +
//     feedback from the previous series iteration — VISIBILITY-WEIGHTED:
//     each feedback cage probe is gated by backface + Chebyshev weights
//     from the baked depth moments, so series light cannot cross
//     walls/roofs ). The exact direct is
//     recomputed with a shadow ray at every hit — Sloan's "retrace sun"
//     discipline, so the coarse probe cache never pollutes the bounce source.
//     BACKFACE hits (surface normal pointing ALONG the gather ray — the far
//     side of single-sided geometry) scatter nothing and are skipped:
//     shading them would record the wall's LIT far side as visible from the
//     probe (e.g. a probe in an inter-wall cavity sees both walls' backfaces
//     but would carry the rooms' light on both sides) — the thin-wall leak,
//     which the series feedback would then smear across the wall in BOTH
//     directions. Treyarch's bake-time backface invalidation / the DDGI
//     frontface rule.
//
// The delta's DIRECT arrival at the probe itself is deliberately NOT stored
// in R_i: the receiver's direct sun stays with the runtime-exact shadowed
// path (no double counting), and the sky's occluded first bounce is
// reconstructed at runtime from V_i (see BakedGI_Update.hlsl).
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>      FrameCB    : register(b0);
ConstantBuffer<BakedGIBakeParams>   BakeParams : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

// Gather rays for each direction. 64 -> 256 on 2026-10-26.
//
// Every gather ray carries the same solid angle, 4*PI/GATHER_RAYS, because the
// sampler is uniform over the full sphere. So one ray onto a distant surface
// stands for that much solid angle regardless of how small the surface is. At
// 64 the weight is 0.196 sr, far more than a cloth across a hall subtends, and
// one such ray decided 74.7 percent of the sun red DC of probe 7669. At 256 the
// weight falls to 0.049 sr and the hit count for any source quadruples.
//
// Cost is bake time only: the payload size and the per-frame read are unchanged.
//
// MUST match BakedGI::kBakeGatherRays (BakedGI.h) and BAKE_RAYS
// (BakedGI_RayDebug.hlsl).
#define GATHER_RAYS 256

// Trilinear evaluation of a previous iteration's response table at worldPos,
// projected with the receiver normal — the series-expansion feedback term.
//
// VISIBILITY-WEIGHTED: each cage probe is gated by the same backface +
// Chebyshev weights the runtime fetch uses, read from the baked octahedral
// depth moments — feedback cannot cross walls/roofs. Plain trilinear
// feedback was the last leak channel (capture-verified, ProbeGI4.rdc): a
// cavity probe's roof-underside gather hit sits in a cell whose corners
// span the roof (dark cavity probes below, legitimately bright above-roof
// probes above); the plain blend injected their response through the slab
// at ~50% weight — cavity probes with V_i = 0 in ALL 16 directions still
// carried R_j ~ 0.04 (~500 sunDC at runtime). With the weights, the
// above-roof probes are Chebyshev-rejected (the slab is between) and the
// feedback is zero — the physically correct value for a surface that cannot
// see any lit region.
// stride: the records per probe. BAKED_GI_DIRECTIONS for the 16-bin table,
//         1 for the method-3 sun table (one record for each probe).
// idx:    the bin index, or 0 for the sun table.
// ONE function, two callers: the only difference between the binned feedback and
// the sun feedback is the stride and the index, so a second copy would be a
// second thing to get wrong.
float3 EvalResponseIndirect(StructuredBuffer<float4> resp, StructuredBuffer<float2> vis,
                            uint stride, uint idx,
                            float3 worldPos, float3 n, uint3 dims, float3 gridMin, float spacing)
{
    // Slide the read point slightly along n (toward the probe hemisphere) so
    // the hit's own recorded surface does not sit exactly on the Chebyshev
    // cliff (d == mu with near-zero variance flips per texel — the failure
    // the runtime fetch's self-shadow bias exists to avoid, in miniature).
    //
    // xq drives the trilinear CELL and the Chebyshev DISTANCE only. It must
    // NOT move the backface plane: that plane is the hit surface's own tangent
    // plane, a property of the surface, not of the pushed read point. It did
    // move it, and a 0.0601 m wall-to-probe-layer gap against a 0.1 m push
    // culled the entire adjacent layer — the probes holding ~94% of the
    // trilinear weight — in favour of the layer 1 m behind it, whose radiance
    // then arrived through the wall. Measured on Sponza: the wall read
    // r/g 3.16 through the far layer and r/g 1.04 through its own.
    float3 xq = worldPos + n * (0.1f * spacing);

    float3 t = clamp((xq - gridMin) / spacing,
                     float3(0.0f, 0.0f, 0.0f), dims - 1.0f);
    int3 base = int3(min(floor(t), max(dims - 2.0f, float3(0.0f, 0.0f, 0.0f))));
    float3 f  = t - base;

    float3 e[9] = (float3[9])0;
    float  wsum = 0.0f;
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        int3 off = int3(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        float3 w3 = lerp(float3(1.0f, 1.0f, 1.0f) - f, f, off);
        float w = w3.x * w3.y * w3.z;

        uint flat = BakedGIFlatten(base + off, dims);
        float3 p  = gridMin + float3(base + off) * spacing;

        // No meta read here: invalid probes carry zeroed responses AND
        // near-geometry moments that Chebyshev self-rejects — and skipping
        // meta avoids reading a buffer this same dispatch is writing.
        float3 toProbe = p - xq;
        float  dist    = length(toProbe);
        float  wFull;
        if (dist < 1e-4f)
        {
            wFull = w; // read point sits on the probe: trivially visible
        }
        else
        {
            float3 dirv = toProbe / dist;

            // Backface from the HIT, not from the pushed point (see above).
            float3 toHit = p - worldPos;
            float  dHit  = length(toHit);
            float  back  = (dHit < 1e-4f) ? 1.0f : saturate(dot(n, toHit / dHit));
            if (back <= 0.0f)
                continue; // behind the hit surface's tangent plane
            float2 m = BakedGISampleMoments(vis, flat, -dirv); // probe -> read point
            wFull = w * back * BakedGIChebyshev(m, dist);
        }

        uint slot = (flat * stride + idx) * BAKED_GI_RESPONSE_F4;
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
            e[lm] += resp[slot + lm].rgb * wFull;
        wsum += wFull;
    }
    if (wsum < 1e-6f)
        return float3(0.0f, 0.0f, 0.0f); // whole cage walled off: the surface IS dark

    float basis[9];
    EvalSH9Basis(n, basis);
    float3 E = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
        E += (e[lm2] / wsum) * basis[lm2];
    return max(E, float3(0.0f, 0.0f, 0.0f));
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint3 dims = uint3(BakeParams.dimX, BakeParams.dimY, BakeParams.dimZ);
    uint probeCount = dims.x * dims.y * dims.z;
    uint probe = dtid.x; // flattened probe cell
    uint dir   = dtid.y; // delta direction index
    // sunMode: the delta light IS the real sun (method 3). One dispatch covers it,
    // so only row 0 works. The record has no bin index — one per probe.
    const bool sunMode = BakeParams.sunMode != 0u;
    if (probe >= probeCount)
        return;
    if (sunMode)
    {
        if (dir != 0u)
            return;
    }
    else if (dir >= BAKED_GI_DIRECTIONS)
        return;

    uint3 cell = uint3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    float3 gridMin = float3(BakeParams.gridMinX, BakeParams.gridMinY, BakeParams.gridMinZ);
    float3 probePos = gridMin + float3(cell) * BakeParams.spacing;

    // Record addressing: 16 bins for the binned transport, ONE record for the sun.
    const uint kRespStride = sunMode ? 1u : BAKED_GI_DIRECTIONS;
    const uint kRespIdx    = sunMode ? 0u : dir;

    // The delta direction. sunMode uses the real sun, so the visibility test and
    // the cosine below finally ask the right question: this is the whole point of
    // the method, and it is why the fold no longer has an angular error.
    float3 sunDir = float3(BakeParams.sunDirX, BakeParams.sunDirY, BakeParams.sunDirZ);
    float3 wDelta;
    if (sunMode)
    {
        const float sunLen = length(sunDir);
        // A zero sun cannot be a delta light. Fall back to +Y rather than emit
        // NaN: a NaN would poison the stored table and the whole probe cell.
        wDelta = (sunLen > 1e-6f) ? (sunDir / sunLen) : float3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        wDelta = BakedGIDeltaDirection(dir);
    }

    RWStructuredBuffer<float4> response = ResourceDescriptorHeap[BakeParams.responseUAVIdx];
    RWStructuredBuffer<uint>    meta     = ResourceDescriptorHeap[BakeParams.metaUAVIdx];
    uint slot = (probe * kRespStride + kRespIdx) * BAKED_GI_RESPONSE_F4;

    // --- Validity: classified by the visibility pass (dispatched BEFORE the
    // transport iterations — its UAV barrier orders the write). DDGI
    // criteria: >= 25% backfaces among 32 stable fibonacci rays = dead; no
    // relocation (bad probes stay put — the experiment measures the criteria
    // swap alone). The old 1cm existential test runs alongside as a
    // diagnostic bit. Dead probes get zeroed responses; all direction
    // threads read the same verdict, so the old same-value race is gone. ---
    const bool valid = (meta[probe] & BAKED_GI_META_VALID) != 0u;

    if (!valid)
    {
        [unroll]
        for (uint z = 0; z < BAKED_GI_RESPONSE_F4; ++z)
            response[slot + z] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    RNG rng;
    rng.state = pcg_hash(probe * 131u + dir * 7919u + 0x9e3779b9u);
    rng.inc   = 1;

    // --- Direct-delta visibility from the probe (V_i) ---
    RayDesc visRay;
    visRay.Origin    = probePos + wDelta * 0.001f;
    visRay.Direction = wDelta;
    visRay.TMin      = 0.001f;
    visRay.TMax      = 1e4f;
    RayQuery<RAY_FLAG_NONE> visQuery;
    visQuery.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, visRay);
    while (visQuery.Proceed()) { PROCESS_ALPHA_MASK(visQuery, rng); }
    float V = (visQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;

    // --- Gather: indirect field around the probe under the delta light ---
    StructuredBuffer<float4> responsePrev = ResourceDescriptorHeap[BakeParams.responsePrevSRVIdx];
    StructuredBuffer<float2> visMoments   = ResourceDescriptorHeap[BakeParams.visSRVIdx];

    float3 acc[9] = (float3[9])0;
    // Dynamic loop on purpose: [unroll] here produces a kernel with 64
    // unrolled inline RayQuery objects and the inlined feedback evaluation,
    // which can make the driver's PSO compilation appear to hang. The small
    // [unroll]s below (9 lm, 8 corners) are fine; this one must stay rolled.
    [loop]
    for (uint r = 0; r < GATHER_RAYS; ++r)
    {
        // Uniform sphere sampling; the cosine lobe lives in the projection
        // basis (e_lm = A_l * integral L(w) Y_lm(w) dw — reciprocity identity).
        float u1 = next_float(rng);
        float u2 = next_float(rng);
        float z   = 1.0f - 2.0f * u1;
        float phi = 2.0f * PI * u2;
        float s   = sqrt(max(0.0f, 1.0f - z * z));
        float3 gdir = float3(s * cos(phi), s * sin(phi), z);

        RayDesc gRay;
        gRay.Origin    = probePos;
        gRay.Direction = gdir;
        gRay.TMin      = 0.001f;
        gRay.TMax      = 1e4f;
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, gRay);
        while (q.Proceed()) { PROCESS_ALPHA_MASK(q, rng); }
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            continue; // sky miss contributes nothing under a delta environment

        Surface surf;
        ResolveHitSurface(gRay, q.CommittedRayT(), q.CommittedInstanceIndex(),
                          q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), surf);

        // Backface rejection: normal pointing along the gather ray = the far
        // side of single-sided geometry (e.g. both walls around an inter-wall
        // cavity probe). A one-sided surface scatters nothing toward the
        // probe from its back — skip the hit entirely (also skips its
        // retrace-sun shadow ray and its feedback read, so the polluted
        // response cannot propagate through the series iterations either).
        // Mirrors the raster path's backface culling convention.
        if (dot(surf.normal, gdir) >= 0.0f)
            continue;

        // Exact direct delta at the hit (retrace-sun discipline).
        float NdotD = max(0.0f, dot(surf.normal, wDelta));
        float Vhit  = 0.0f;
        if (NdotD > 0.0f)
        {
            RayDesc sRay;
            sRay.Origin    = surf.worldPos + surf.normal * 0.001f;
            sRay.Direction = wDelta;
            sRay.TMin      = 0.001f;
            sRay.TMax      = 1e4f;
            RayQuery<RAY_FLAG_NONE> sq;
            sq.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sRay);
            while (sq.Proceed()) { PROCESS_ALPHA_MASK(sq, rng); }
            Vhit = (sq.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;
        }

        // Feedback: previous iteration's indirect response at the hit,
        // visibility-weighted so it cannot cross walls/roofs.
        float3 Efb = (BakeParams.iteration > 0u)
            ? EvalResponseIndirect(responsePrev, visMoments, kRespStride, kRespIdx, surf.worldPos, surf.normal, dims, gridMin, BakeParams.spacing)
            : float3(0.0f, 0.0f, 0.0f);

        // Lambertian outgoing radiance under the delta light.
        float3 diffAlbedo = surf.albedo * (1.0f - surf.metallic);
        float3 L = diffAlbedo / PI * (NdotD * Vhit + Efb);

        float basis[9];
        EvalSH9Basis(gdir, basis);
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
            acc[lm] += basis[lm] * L;
    }

    float norm = (4.0f * PI / (float)GATHER_RAYS);
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
        response[slot + lm2] = float4(acc[lm2] * norm * BakedGIA_l(lm2), 0.0f);
    response[slot + 9] = float4(V, 0.0f, 0.0f, 0.0f);
}
