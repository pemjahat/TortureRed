// =============================================================================
// BakedGI_Bake.hlsl — step-1 transport bake .
//
// One thread per (probe, delta direction W_i):
//   - Validity: READ from the meta bits classified by the visibility pass
//     (DDGI criteria — >= 25% backfaces among 32 stable fibonacci rays;
//     see BakedGI_Visibility.hlsl). Dead probes' responses are zeroed.
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
// reconstructed at runtime from the SH9 sky-visibility field (see
// BakedGI_Update.hlsl).
//
// GUIDED GATHER (task033 section 3). Iterations 1 and 2 replace the dense
// uniform ray set with the defensive mixture: a uniform fibonacci stratum
// plus per-record guided draws from a CDF built off the PREVIOUS iteration's
// table — the record is the band-2 projection of this thread's own integrand,
// so its reconstruction is the guide signal, with no trial pass and no extra
// buffer. Every batch stays an unbiased estimate of the same integral (the
// guide block in BakedGI.hlsli owns the derivation), so the running mean and
// the fold keep their exact identities. Iteration 0 stays uniform. The
// ray-debug replay shares the same build/sample functions, so a dump
// reproduces the draws.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>      FrameCB    : register(b0);
ConstantBuffer<BakedGIBakeParams>   BakeParams : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

// Gather rays for each direction and batch. Iteration 0 traces a uniform
// rotated fibonacci set: every ray carries the same solid angle,
// 4*PI/GATHER_RAYS. Iterations 1 and 2 sample the GUIDED MIXTURE instead
// (BakedGI.hlsli's guide block owns the math): BAKED_GI_GUIDE_UNIFORM_RAYS
// uniform-stratum rays from a fresh 64-point set, then guide-drawn rays with
// weight 1/(K p_j) under the same 4*PI/GATHER_RAYS norm. Eight independent
// batches give 2048 rays per record.
//
// Each batch is a separate dispatch. The output holds the exact linear running
// mean, so no accumulation buffer or nonlinear history filter is needed.
// Cost is bake time only: the payload size and the per-frame read are unchanged.
//
// MUST match BakedGI::kBakeGatherRays / kBakeGatherBatches (BakedGI.h) and
// BAKE_RAYS / BAKE_BATCHES (BakedGI_RayDebug.hlsl).
#define GATHER_RAYS 256
#define GATHER_BATCHES 8
// The mixture's alpha is structural: the uniform stratum is exactly 1/4 of a
// batch, so a guided ray's weight and a uniform ray's weight share one norm.
#if (GATHER_RAYS != 4 * BAKED_GI_GUIDE_UNIFORM_RAYS)
#error "guided mixture: the uniform stratum must be exactly 1/4 of a batch (alpha = 0.25)"
#endif

// Trilinear evaluation of a previous iteration's response table at worldPos,
// projected with the receiver normal — the series-expansion feedback term.
//
// VISIBILITY-WEIGHTED: each cage probe is gated by the same backface +
// Chebyshev weights the runtime fetch uses, read from the baked octahedral
// depth moments — feedback cannot cross walls/roofs. Without the weights the
// blend leaks: a cavity probe's roof-underside gather hit sits in a cell whose
// corners span the roof (dark cavity probes below, legitimately bright
// above-roof probes above), so a plain blend injects their response through
// the slab at ~50% weight — cavity probes with V_i = 0 in ALL 16 directions
// then carry R_j ~ 0.04 (~500 sunDC at runtime). With the weights, the
// above-roof probes are Chebyshev-rejected (the slab is between) and the
// feedback is zero — the physically correct value for a surface that cannot
// see any lit region. Measured in ProbeGI4.rdc.
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
    // isSunPass: the delta light is the sun at ONE hour state. One dispatch per
    // state, so only row 0 works; sunState names which of the 17 records is
    // written. isSunPass 0 is the 16-direction sky transport, which still runs:
    // its table is the live sky quadrature.
    const bool isSunPass = BakeParams.isSunPass != 0u;
    if (probe >= probeCount)
        return;
    if (isSunPass)
    {
        if (dir != 0u)
            return;
    }
    else if (dir >= BAKED_GI_DIRECTIONS)
        return;

    uint3 cell = uint3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    float3 gridMin = float3(BakeParams.gridMinX, BakeParams.gridMinY, BakeParams.gridMinZ);
    float3 probePos = gridMin + float3(cell) * BakeParams.spacing;

    // Record addressing: 16 bins for the binned transport, one record for each
    // sun hour state for the sun transport.
    const uint kRespStride = isSunPass ? BAKED_GI_SUN_STATES : BAKED_GI_DIRECTIONS;
    const uint kRespIdx    = isSunPass ? BakeParams.sunState : dir;

    // The delta direction. isSunPass uses the real sun, so the visibility test and
    // the cosine below ask the right question: this is the whole point of the
    // method, and it is why the fold has no angular error.
    float3 sunDir = float3(BakeParams.sunDirX, BakeParams.sunDirY, BakeParams.sunDirZ);
    float3 wDelta;
    if (isSunPass)
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
    // swap alone). The 1cm existential test runs alongside as a diagnostic
    // bit. Dead probes get zeroed responses; all direction
    // threads read the same verdict, so no thread races on it. ---
    const bool valid = (meta[probe] & BAKED_GI_META_VALID) != 0u;

    if (!valid)
    {
        [unroll]
        for (uint z = 0; z < BAKED_GI_RESPONSE_F4; ++z)
            response[slot + z] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // The seed index. seedDir 0xFFFFFFFF means "this thread's own direction".
    // Any other value pairs this pass with one fibonacci bin. The batch seed
    // keeps adjacent sun states independent and gives every batch a new rotation.
    const uint seedDir = (BakeParams.seedDir == 0xFFFFFFFFu) ? dir : BakeParams.seedDir;
    const uint batchSeed = BakedGIBatchSeed(probe, seedDir, BakeParams.batchIndex);
    const float3x3 gatherRotation = BakedGIBatchRotation(batchSeed);
    RNG rng;
    rng.state = batchSeed;
    rng.inc   = 1;

    // --- Gather: indirect field around the probe under the delta light ---
    //
    // Iteration 0: directions come from the rotated fibonacci set and consume
    // no RNG draws; the RNG remains available to alpha-mask traversal in bake
    // order — bit-identical to the pre-guiding stream. Iterations > 0: the
    // guided mixture (BakedGI.hlsli). The guide CDF is built from
    // responsePrev — the PREVIOUS iteration's table, fixed for the whole
    // iteration by the ping-pong, so one build per batch sees the same data
    // every batch sees. Sampling draws happen at the TOP of each guided ray's
    // loop iteration, before its traversal, in the fixed order
    // BakedGIGuideSample documents — the replay discipline. The per-ray
    // branch is uniform across the wave: `guided` is a root constant and
    // `r` is the shared loop counter.
    StructuredBuffer<float4> responsePrev = ResourceDescriptorHeap[BakeParams.responsePrevSRVIdx];
    StructuredBuffer<float2> visMoments   = ResourceDescriptorHeap[BakeParams.visSRVIdx];

    const bool guided = BakeParams.iteration > 0u;
    BakedGIGuide guide; // read only when guided, so the unset state is inert
    if (guided)
        guide = BakedGIGuideBuild(responsePrev, slot);

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
        // Guided rays carry weight 1/(K p_j); uniform rays carry 1. The norm
        // below is shared, so the two strata sum to the exact mixture.
        float  rayWeight = 1.0f;
        float3 gdir;
        if (!guided)
            gdir = BakedGIRotatedFibonacci(r, GATHER_RAYS, gatherRotation);
        else if (r < BAKED_GI_GUIDE_UNIFORM_RAYS)
            gdir = BakedGIRotatedFibonacci(r, BAKED_GI_GUIDE_UNIFORM_RAYS, gatherRotation);
        else
            gdir = BakedGIGuideSample(guide, rng, rayWeight);

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
            acc[lm] += basis[lm] * L * rayWeight;
    }

    // Shared by both strata: a uniform ray is 4*PI/GATHER_RAYS * 1, a guided
    // ray is 4*PI/GATHER_RAYS * 1/(K p_j), and the mixture's alpha = 64/256
    // makes the per-ray weights reduce to this single norm (guide block,
    // BakedGI.hlsli).
    const float norm = (4.0f * PI / (float)GATHER_RAYS);
    const float batchWeight = 1.0f / (float)(BakeParams.batchIndex + 1u);
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
    {
        const float3 batchValue = acc[lm2] * norm * BakedGIA_l(lm2);
        const float3 previousMean = response[slot + lm2].rgb;
        const float3 mean = (BakeParams.batchIndex == 0u)
            ? batchValue : lerp(previousMean, batchValue, batchWeight);
        response[slot + lm2] = float4(mean, 0.0f);
    }
}
