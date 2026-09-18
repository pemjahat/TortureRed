// =============================================================================
// BakedGI_Bake.hlsl — step-1 transport bake .
//
// One thread per (probe, delta direction W_i):
//   - Validity: short ray from the probe center (backfaces hit when inside
//     geometry) — the only step-1 placement logic; probes in solid geometry
//     are marked invalid and their response is zeroed.
//   - Direct-delta visibility V_i: one shadow ray toward W_i (stored in slot
//     [9].x; the update pass uses it for the occluded first-bounce sky term).
//   - INDIRECT-ONLY directional-irradiance SH9 RGB response R_i to a unit
//     delta light at W_i: uniform-sphere gather rays; each hit's outgoing
//     radiance = diffuseAlbedo/pi * ( exact direct delta at the hit  +
//     feedback from the previous series iteration ). The exact direct is
//     recomputed with a shadow ray at every hit — Sloan's "retrace sun"
//     discipline, so the coarse probe cache never pollutes the bounce source.
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

#define GATHER_RAYS 64

// Trilinear evaluation of a previous iteration's response table at worldPos,
// projected with the receiver normal — the series-expansion feedback term.
float3 EvalResponseIndirect(StructuredBuffer<float4> resp, uint dir,
                            float3 worldPos, float3 n, uint3 dims, float3 gridMin, float spacing)
{
    float3 t = clamp((worldPos - gridMin) / spacing,
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
        uint slot = (flat * BAKED_GI_DIRECTIONS + dir) * BAKED_GI_RESPONSE_F4;
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
            e[lm] += resp[slot + lm].rgb * w;
        wsum += w;
    }
    if (wsum < 1e-6f)
        return float3(0.0f, 0.0f, 0.0f);

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
    if (probe >= probeCount || dir >= BAKED_GI_DIRECTIONS)
        return;

    uint3 cell = uint3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    float3 gridMin = float3(BakeParams.gridMinX, BakeParams.gridMinY, BakeParams.gridMinZ);
    float3 probePos = gridMin + float3(cell) * BakeParams.spacing;
    float3 wDelta = BakedGIDeltaDirection(dir);

    RWStructuredBuffer<float4> response = ResourceDescriptorHeap[BakeParams.responseUAVIdx];
    RWStructuredBuffer<uint>    meta     = ResourceDescriptorHeap[BakeParams.metaUAVIdx];
    uint slot = (probe * BAKED_GI_DIRECTIONS + dir) * BAKED_GI_RESPONSE_F4;

    // --- Validity: probe center in free space (backfaces must hit if inside) ---
    RayDesc probeRay;
    probeRay.Origin    = probePos;
    probeRay.Direction = wDelta;
    probeRay.TMin      = 0.0f;
    probeRay.TMax      = max(0.01f, BakeParams.spacing * 0.005f);
    RayQuery<RAY_FLAG_NONE> insideQuery;
    insideQuery.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, probeRay);
    while (insideQuery.Proceed()) { /* non-opaque candidates do not block */ }
    bool valid = (insideQuery.CommittedStatus() == COMMITTED_NOTHING);
    meta[probe] = valid ? 1u : 0u; // benign same-value race across direction threads

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

        // Feedback: previous iteration's indirect response at the hit.
        float3 Efb = (BakeParams.iteration > 0u)
            ? EvalResponseIndirect(responsePrev, dir, surf.worldPos, surf.normal, dims, gridMin, BakeParams.spacing)
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
