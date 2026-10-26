// =============================================================================
// BakedGI_RayDebug.hlsl — on-demand re-trace of one probe's gather rays.
//
// DEBUG ONLY. This pass never runs in a normal frame. It re-runs the transport
// bake for ONE probe and writes evidence for the human.
//
// TWO RAY COUNTS, on purpose:
//   BAKE_RAYS  = 256 — what the bake traces for each direction. The replay must
//                      use the same count or the self-check compares unlike
//                      quantities.
//   DEBUG_RAYS = 64  — the record slots kept for each direction. The record
//                      buffer must not grow: see BakedGI::kRayDebugTableBase.
//
// WHAT THE 64 SLOTS HOLD. Not the first 64 rays. A ray carries a direct term
// only when NdotD * Vhit > 0 (BakedGI_Bake.hlsl:269), and those are the rays
// worth drawing, so they go FIRST, in ascending ray order. The rest of the slots
// hold the other rays at an even stride: the gather directions are uniform over
// the sphere and in random order, so every n-th one is a uniform subsample and
// the angular coverage of the sphere survives. If more than DEBUG_RAYS rays
// qualify, the ones with the largest NdotD * Vhit win and no context ray is
// kept. Every slot is always written: one branch of each case fills them all.
//
// TWO PASSES over the 256 rays. Pass 1 learns which rays qualify. Pass 2 writes
// the chosen records and accumulates the SH9 the self-check needs. The doubling
// is deliberate. The one-pass alternative holds 256 candidate records in
// groupshared (24 KB) and compacts them at the end; that is more code, and more
// ways to be silently wrong, in a pass whose whole job is to be trustworthy.
//
// REPRODUCTION RULE. Both passes must reproduce the bake's RNG stream EXACTLY.
// The stream is sequential (pcg state, next_float), so the direction of ray r
// depends on the alpha-mask draws of the V_i ray, rays 0..r-1, and their shadow
// rays. Each pass therefore replays the whole sequence from the same seed, in the
// same order, with the same ray flags, and draws for nothing else. Efb draws no
// random numbers, so pass 1 skips EvalResponseIndirect and the two streams still
// agree. Lane 0 of each group does all the work.
//
// The gather loop stays rolled ([loop]). An unrolled RayQuery loop makes the
// driver's PSO compilation appear to hang (see RENDERCORE-MEMORY.md), and at 256
// rays it would be four times worse than the 64 it was measured at.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>        FrameCB : register(b0);
ConstantBuffer<BakedGIRayDebugParams> RayDbg  : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

#define BAKE_RAYS  256u // must match GATHER_RAYS in BakedGI_Bake.hlsl
#define DEBUG_RAYS 64u  // record slots for each direction; BakedGI::kDebugGatherRays

// Ray-record layout, in float4s for each ray. The re-accumulated table starts
// at BakedGI::kRayDebugTableBase, which the C++ side and BakedGI_Debug.hlsl
// both derive from DEBUG_RAYS — so this stride must stay DEBUG_RAYS, not
// BAKE_RAYS, or the table would land 302,592 B past the end of the buffer.
//   rec0 = (gather direction.xyz, hit distance)   rec1 = (diffuse albedo.rgb, NdotD)
//   rec2 = (L.rgb, Vhit)                          rec3 = (Efb.rgb, packed)
//   rec4 = (hit normal.xyz, instance index)       rec5 = (hit world pos.xyz, primitive index)
// rec4/rec5 are the hit's identity. w = -1 marks "no surface", NOT 0: instance 0
// is a legal instance and a dump must not confuse the two. They are stored as
// plain FLOATS, not asfloat bit patterns, so a float-typed buffer dump prints an
// index as a number; instance counts are far below the 2^24 exact-integer limit.
//
// rec3.w packs  status + 4 * sourceRayIndex. BakedGI.hlsli owns the description
// and the decode (BakedGIRayRecStatus / BakedGIRayRecSourceRay).
// Must match BakedGI::kRayRecordF4 (BakedGI.h) and BAKED_GI_RAYFAN_REC_F4
// (BakedGI_Debug.hlsl) — one stride, three files.
#define RAY_RECORDS_BASE   0u
#define RAY_RECORD_F4      6u
#define RAY_DEBUG_TABLE_BASE (BAKED_GI_DIRECTIONS * DEBUG_RAYS * RAY_RECORD_F4)

// Selection state for ONE direction's group. Only lane 0 touches these, so they
// need no barrier: groupshared is simply a scratch block that a lone lane would
// otherwise spill into local memory. 2,048 B per group.
groupshared float g_q[BAKE_RAYS];    // NdotD * Vhit for each source ray; 0 = no direct term
groupshared int   g_slot[BAKE_RAYS]; // record slot for each source ray; -1 = not written

// ---------------------------------------------------------------------------
// Trilinear evaluation of a previous iteration's response table.
//
// COPIED VERBATIM from BakedGI_Bake.hlsl. Keep the two copies in step. The bake
// is the reference. This copy exists so the debug pass can reproduce the
// feedback term (Efb) of the final iteration without a change to the bake.
// ---------------------------------------------------------------------------
float3 EvalResponseIndirect(StructuredBuffer<float4> resp, StructuredBuffer<float2> vis, uint dir,
                            float3 worldPos, float3 n, uint3 dims, float3 gridMin, float spacing)
{
    // xq drives the trilinear CELL and the Chebyshev DISTANCE only; the
    // backface plane comes from the unpushed hit. Mirrors BakedGI_Bake.hlsl.
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

        float3 toProbe = p - xq;
        float  dist    = length(toProbe);
        float  wFull;
        if (dist < 1e-4f)
        {
            wFull = w;
        }
        else
        {
            float3 dirv = toProbe / dist;
            float3 toHit = p - worldPos;
            float  dHit  = length(toHit);
            float  back  = (dHit < 1e-4f) ? 1.0f : saturate(dot(n, toHit / dHit));
            if (back <= 0.0f)
                continue;
            float2 m = BakedGISampleMoments(vis, flat, -dirv);
            wFull = w * back * BakedGIChebyshev(m, dist);
        }

        uint slot = (flat * BAKED_GI_DIRECTIONS + dir) * BAKED_GI_RESPONSE_F4;
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
            e[lm] += resp[slot + lm].rgb * wFull;
        wsum += wFull;
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

// ---------------------------------------------------------------------------
// The V_i ray: one shadow ray from the probe toward the delta direction. Runs
// once per pass, from the pass's own seed, so each pass is an independent full
// replay of the bake.
// ---------------------------------------------------------------------------
float TraceVisRay(inout RNG rng, float3 probePos, float3 wDelta)
{
    RayDesc visRay;
    visRay.Origin    = probePos + wDelta * 0.001f;
    visRay.Direction = wDelta;
    visRay.TMin      = 0.001f;
    visRay.TMax      = 1e4f;
    RayQuery<RAY_FLAG_NONE> visQuery;
    visQuery.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, visRay);
    while (visQuery.Proceed()) { PROCESS_ALPHA_MASK(visQuery, rng); }
    return (visQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// One gather ray of the bake, replayed. Every RNG draw the bake makes for this
// ray happens inside here and nowhere else, so both passes consume the stream
// identically and cannot drift from the bake or from each other.
//
// Efb is NOT computed here: it draws nothing, so pass 1 can skip it.
// ---------------------------------------------------------------------------
struct RayProbe
{
    float3 gdir;     // the sampled gather direction
    float  t;        // committed distance; 1e4f on a miss
    float3 albedo;   // diffuse albedo at the hit, zero on a miss
    float3 normal;   // hit normal, zero on a miss
    float3 worldPos; // hit position, zero on a miss
    float  instance; // hit instance index; -1 on a miss
    float  prim;     // hit primitive index; -1 on a miss
    float  NdotD;    // max(0, dot(hit normal, wDelta))
    float  Vhit;     // 1 when the shadow ray from the hit commits nothing
    float  status;   // 0 = miss, 1 = backface reject, 2 = shaded
    float  q;        // NdotD * Vhit: the direct term's weight, 0 when it has none
};

RayProbe ProbeRay(inout RNG rng, float3 probePos, float3 wDelta)
{
    RayProbe p;
    p.gdir     = float3(0.0f, 1.0f, 0.0f);
    p.t        = 1e4f;
    p.albedo   = float3(0.0f, 0.0f, 0.0f);
    p.normal   = float3(0.0f, 0.0f, 0.0f);
    p.worldPos = float3(0.0f, 0.0f, 0.0f);
    p.instance = -1.0f;
    p.prim     = -1.0f;
    p.NdotD    = 0.0f;
    p.Vhit     = 0.0f;
    p.status   = 0.0f;
    p.q        = 0.0f;

    // Two draws first — the same order as BakedGI_Bake.hlsl:213-214.
    float u1 = next_float(rng);
    float u2 = next_float(rng);
    float z   = 1.0f - 2.0f * u1;
    float phi = 2.0f * PI * u2;
    float s   = sqrt(max(0.0f, 1.0f - z * z));
    p.gdir = float3(s * cos(phi), s * sin(phi), z);

    RayDesc gRay;
    gRay.Origin    = probePos;
    gRay.Direction = p.gdir;
    gRay.TMin      = 0.001f;
    gRay.TMax      = 1e4f;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, gRay);
    while (q.Proceed()) { PROCESS_ALPHA_MASK(q, rng); }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return p; // sky miss: status 0

    Surface surf;
    ResolveHitSurface(gRay, q.CommittedRayT(), q.CommittedInstanceIndex(),
                      q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), surf);

    p.t        = q.CommittedRayT();
    p.albedo   = surf.albedo * (1.0f - surf.metallic);
    p.normal   = surf.normal;
    p.worldPos = surf.worldPos;
    p.instance = (float)q.CommittedInstanceIndex();
    p.prim     = (float)q.CommittedPrimitiveIndex();
    // The identity is filled for a BACKFACE REJECT as well as for a shaded hit:
    // the reject names a real surface too, and that surface is often the one
    // being hunted. NdotD/Vhit/q stay zero on a reject, because the bake skips it.
    p.status   = 1.0f;

    // Backface rejection, the bake's rule: a normal pointing along the gather ray
    // is the far side of single-sided geometry.
    if (dot(surf.normal, p.gdir) >= 0.0f)
        return p; // status 1

    p.NdotD = max(0.0f, dot(surf.normal, wDelta));
    if (p.NdotD > 0.0f)
    {
        RayDesc sRay;
        sRay.Origin    = surf.worldPos + surf.normal * 0.001f;
        sRay.Direction = wDelta;
        sRay.TMin      = 0.001f;
        sRay.TMax      = 1e4f;
        RayQuery<RAY_FLAG_NONE> sq;
        sq.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sRay);
        while (sq.Proceed()) { PROCESS_ALPHA_MASK(sq, rng); }
        p.Vhit = (sq.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;
    }
    p.status = 2.0f;
    p.q      = p.NdotD * p.Vhit;
    return p;
}

// The six float4 of one record, assembled in ONE place so the three status cases
// cannot disagree about a field. rec3.w packs the status and the SOURCE ray
// index (BakedGI.hlsli): the record is a fixed 96 B and no field is spare.
//
// A macro writes it, because a UAV object cannot be passed to a function
// portably across shader compilers.
#define WRITE_RECORD(dbg, recBase, slot, srcRay, p, L, Efb)                        \
    {                                                                              \
        const uint b_ = (recBase) + (slot) * RAY_RECORD_F4;                         \
        (dbg)[b_ + 0u] = float4((p).gdir, (p).t);                                   \
        (dbg)[b_ + 1u] = float4((p).albedo, (p).NdotD);                             \
        (dbg)[b_ + 2u] = float4((L), (p).Vhit);                                     \
        (dbg)[b_ + 3u] = float4((Efb), (p).status + 4.0f * (float)(srcRay));        \
        (dbg)[b_ + 4u] = float4((p).normal, (p).instance);                          \
        (dbg)[b_ + 5u] = float4((p).worldPos, (p).prim);                            \
    }

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    // One group for each direction. Lane 0 replays the sequence, because the
    // RNG stream is sequential (see the header).
    if (dtid.x != 0u)
        return;

    const uint dir = dtid.y;
    if (dir >= BAKED_GI_DIRECTIONS)
        return;

    const uint3 dims    = uint3(RayDbg.dimX, RayDbg.dimY, RayDbg.dimZ);
    const float spacing = RayDbg.spacing;
    const float3 gridMin = float3(RayDbg.gridMinX, RayDbg.gridMinY, RayDbg.gridMinZ);
    const uint probe = RayDbg.probeIndex;

    const uint3 cell = uint3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    const float3 probePos = gridMin + float3(cell) * spacing;
    const float3 wDelta   = BakedGIDeltaDirection(dir);

    RWStructuredBuffer<float4> debug    = ResourceDescriptorHeap[RayDbg.rayRecUAVIdx];
    StructuredBuffer<float4>   respPrev = ResourceDescriptorHeap[RayDbg.responsePrevSRVIdx];
    StructuredBuffer<float2>   visMom   = ResourceDescriptorHeap[RayDbg.visSRVIdx];

    // The seed, twice. The bake's exact formula (BakedGI_Bake.hlsl:185). Each
    // pass replays from it, so neither pass inherits the other's stream.
    const uint seed = pcg_hash(probe * 131u + dir * 7919u + 0x9e3779b9u);

    RNG rng;
    rng.state = seed;
    rng.inc   = 1;
    const float V = TraceVisRay(rng, probePos, wDelta);

    // -------------------------------------------------------------------------
    // PASS 1 — which rays carry a direct term?
    //
    // NdotD * Vhit is 0 for a miss, for a backface reject, and for any hit the
    // sun does not reach. Only a non-zero product puts anything into the direct
    // half of L, so those rays are the evidence. Keep the weight for every ray
    // and decide the slots once the whole direction is known.
    // -------------------------------------------------------------------------
    [loop]
    for (uint r1 = 0; r1 < BAKE_RAYS; ++r1)
    {
        const RayProbe p1 = ProbeRay(rng, probePos, wDelta);
        g_q[r1] = p1.q;
    }

    uint nQual = 0u; // rays that carry a direct term
    uint nCtx  = 0u; // rays that do not
    [loop]
    for (uint r2 = 0; r2 < BAKE_RAYS; ++r2)
    {
        if (g_q[r2] > 0.0f) ++nQual; else ++nCtx;
        g_slot[r2] = -1;
    }

    // Slots fill from 0 upward. Two cases, and slot 9's w records which ran.
    //   case 0: nQual <= DEBUG_RAYS — every qualifier keeps its ray order, and
    //           the free slots take context rays at an even stride.
    //   case 1: nQual >  DEBUG_RAYS — the DEBUG_RAYS largest weights win, in
    //           descending order, and no context ray is kept.
    uint used = 0u;
    if (nQual <= DEBUG_RAYS)
    {
        for (uint r3 = 0; r3 < BAKE_RAYS; ++r3)
            if (g_q[r3] > 0.0f)
                g_slot[r3] = (int)used++;

        const uint ctxSlots = DEBUG_RAYS - used;
        if (ctxSlots > 0u && nCtx > 0u)
        {
            // Even stride over the context rays. They appear in random order, so
            // every n-th one is a uniform subsample of the sphere and not a
            // contiguous wedge of it.
            //
            // The stride is a FLOOR division and the count is capped, not the
            // other way round. Ceil division under-delivers: with nCtx = 252 and
            // ctxSlots = 60 it gives stride 5, which yields only 51 rays, and
            // nine slots would be left holding the previous dump's records. With
            // floor, nCtx / stride >= ctxSlots always, so wrote reaches ctxSlots
            // and every slot is written. That property is what makes the "no
            // unused slot" invariant below true.
            const uint stride = max(1u, nCtx / ctxSlots);
            uint ctxIndex = 0u;
            uint wrote    = 0u;
            [loop]
            for (uint r4 = 0; r4 < BAKE_RAYS && wrote < ctxSlots; ++r4)
            {
                if (g_q[r4] > 0.0f)
                    continue; // a qualifier already has its slot
                if ((ctxIndex % stride) == 0u)
                {
                    g_slot[r4] = (int)used++;
                    ++wrote;
                }
                ++ctxIndex;
            }
        }
    }
    else
    {
        // Selection sort, DEBUG_RAYS times: each pass takes the largest weight
        // not yet taken. 64 x 256 comparisons in one lane, which is nothing
        // against the 256 ray traversals around it.
        for (uint pick = 0; pick < DEBUG_RAYS; ++pick)
        {
            int   bestRay = -1;
            float bestQ   = 0.0f;
            [loop]
            for (uint r5 = 0; r5 < BAKE_RAYS; ++r5)
            {
                if (g_slot[r5] >= 0 || g_q[r5] <= 0.0f)
                    continue;
                if (g_q[r5] > bestQ)
                {
                    bestQ   = g_q[r5];
                    bestRay = (int)r5;
                }
            }
            if (bestRay < 0)
                break;
            g_slot[(uint)bestRay] = (int)used++;
        }
    }

    // -------------------------------------------------------------------------
    // PASS 2 — write the chosen records and accumulate the full SH9.
    //
    // The accumulation covers ALL BAKE_RAYS, not the 64 written: the self-check
    // compares the re-accumulated table against the stored one, and the stored
    // one was built from every ray. Accumulating only the written records would
    // turn the MATCH line into a lie.
    // -------------------------------------------------------------------------
    rng.state = seed;
    rng.inc   = 1;
    TraceVisRay(rng, probePos, wDelta); // V again: identical seed, identical ray

    float3 acc[9] = (float3[9])0;
    const uint recBase = RAY_RECORDS_BASE + (dir * DEBUG_RAYS) * RAY_RECORD_F4;

    [loop]
    for (uint r = 0; r < BAKE_RAYS; ++r)
    {
        const RayProbe p = ProbeRay(rng, probePos, wDelta);

        // One record write below covers all three status cases, so the branches
        // only fill L and Efb. A miss and a backface reject leave both at zero:
        // the bake skips them, so they carry no L and no Efb.
        float3 L   = float3(0.0f, 0.0f, 0.0f);
        float3 Efb = float3(0.0f, 0.0f, 0.0f);

        if (p.status > 1.5f)
        {
            // Shaded: the only branch the bake puts into acc, and the only one
            // that reads the feedback cage.
            Efb = EvalResponseIndirect(respPrev, visMom, dir, p.worldPos,
                                       p.normal, dims, gridMin, spacing);
            L   = p.albedo / PI * (p.NdotD * p.Vhit + Efb);

            float basis[9];
            EvalSH9Basis(p.gdir, basis);
            [unroll]
            for (uint lm = 0; lm < 9; ++lm)
                acc[lm] += basis[lm] * L;
        }

        const int sl = g_slot[r];
        if (sl >= 0)
            WRITE_RECORD(debug, recBase, (uint)sl, r, p, L, Efb);
    }

    // --- Self-check: store the same normalised SH9 the bake stores, with the
    // bake's own divisor. ---
    const float norm = (4.0f * PI / (float)BAKE_RAYS);
    const uint tBase = RAY_DEBUG_TABLE_BASE + dir * BAKED_GI_RESPONSE_F4;
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
        debug[tBase + lm2] = float4(acc[lm2] * norm * BakedGIA_l(lm2), 0.0f);
    // Slot 9: x = V_i, as the bake stores it. w carries the selection flag for
    // this direction — 0 = every qualifier kept, 1 = the largest 64 kept. The
    // self-check compares x, y and z only, and the bake writes 0 to w, so w is
    // the one float in the table no comparison can see.
    debug[tBase + 9u] = float4(V, 0.0f, 0.0f, (nQual > DEBUG_RAYS) ? 1.0f : 0.0f);
}
