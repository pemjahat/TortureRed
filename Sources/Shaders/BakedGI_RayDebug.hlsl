// =============================================================================
// BakedGI_RayDebug.hlsl — on-demand re-trace of one probe's gather rays.
//
// DEBUG ONLY. This pass never runs in a normal frame. It re-runs the transport
// bake for ONE probe and writes evidence for the human:
//   rayRec[h]        64 records, one for each gather ray of direction h.
//   rayRec[4096 ..]  the re-accumulated SH9 response + V_i for each direction,
//                    so the caller can compare it with the stored table.
//
// REPRODUCTION RULE. The pass must reproduce the bake's RNG stream EXACTLY.
// The stream is sequential (pcg state, next_float), so the direction of ray r
// depends on the alpha-mask draws of the V_i ray, rays 0..r-1, and their shadow
// rays. One thread can therefore not reproduce a single ray alone. The pass
// replays the whole sequence from the same seed, in the same order, with the
// same ray flags. Lane 0 of each group does all 64 rays of its direction.
//
// The gather loop stays rolled ([loop]). An unrolled RayQuery loop makes the
// driver's PSO compilation appear to hang (see RENDERCORE-MEMORY.md).
//
// Source of the order and of the duplicated helper: BakedGI_Bake.hlsl:58-256.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>        FrameCB : register(b0);
ConstantBuffer<BakedGIRayDebugParams> RayDbg  : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

#define GATHER_RAYS 64 // must match BakedGI_Bake.hlsl

// Ray-record layout, in float4s for each ray. The re-accumulated table starts
// at RayDebugTableBase.
//   rec0 = (gather direction.xyz, hit distance)   rec1 = (diffuse albedo.rgb, NdotD)
//   rec2 = (L.rgb, Vhit)                          rec3 = (Efb.rgb, status)
//   rec4 = (hit normal.xyz, instance index)       rec5 = (hit world pos.xyz, primitive index)
// rec4/rec5 are the hit's identity. w = -1 marks "no surface", NOT 0: instance 0
// is a legal instance and a dump must not confuse the two. They are stored as
// plain FLOATS, not asfloat bit patterns, so a float-typed buffer dump prints an
// index as a number; instance counts are far below the 2^24 exact-integer limit.
// Must match BakedGI::kRayRecordF4 (BakedGI.h) and BAKED_GI_RAYFAN_REC_F4
// (BakedGI_Debug.hlsl) — one stride, three files.
#define RAY_RECORDS_BASE   0u
#define RAY_RECORD_F4      6u
#define RAY_DEBUG_TABLE_BASE (BAKED_GI_DIRECTIONS * GATHER_RAYS * RAY_RECORD_F4)

// ---------------------------------------------------------------------------
// Trilinear evaluation of a previous iteration's response table.
//
// COPIED VERBATIM from BakedGI_Bake.hlsl:58-119. Keep the two copies in step.
// The bake is the reference. This copy exists so the debug pass can reproduce
// the feedback term (Efb) of the final iteration without a change to the bake.
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

    // --- Seed: the bake's exact formula (BakedGI_Bake.hlsl:158). ---
    RNG rng;
    rng.state = pcg_hash(probe * 131u + dir * 7919u + 0x9e3779b9u);
    rng.inc   = 1;

    // --- Replay the V_i ray first: it consumes RNG before the gather loop. ---
    RayDesc visRay;
    visRay.Origin    = probePos + wDelta * 0.001f;
    visRay.Direction = wDelta;
    visRay.TMin      = 0.001f;
    visRay.TMax      = 1e4f;
    RayQuery<RAY_FLAG_NONE> visQuery;
    visQuery.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, visRay);
    while (visQuery.Proceed()) { PROCESS_ALPHA_MASK(visQuery, rng); }
    const float V = (visQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;

    // --- Replay the 64 gather rays in bake order. ---
    float3 acc[9] = (float3[9])0;
    const uint recBase = RAY_RECORDS_BASE + (dir * GATHER_RAYS) * RAY_RECORD_F4;

    [loop]
    for (uint r = 0; r < GATHER_RAYS; ++r)
    {
        // Two draws first — the same order as BakedGI_Bake.hlsl:186-187.
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

        // Record default: a miss. Status 0 = miss, 1 = backface reject, 2 = shaded.
        float4 r0 = float4(gdir, 1e4f);
        float4 r1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 r2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 r3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        // rec4/rec5: the hit's identity. w = -1 for a miss (no surface), so a
        // reader can tell it from instance 0, which is a legal instance.
        float4 r4 = float4(0.0f, 0.0f, 0.0f, -1.0f);
        float4 r5 = float4(0.0f, 0.0f, 0.0f, -1.0f);

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            Surface surf;
            ResolveHitSurface(gRay, q.CommittedRayT(), q.CommittedInstanceIndex(),
                              q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), surf);

            r0.w = q.CommittedRayT();
            r1   = float4(surf.albedo * (1.0f - surf.metallic), 0.0f);
            // Identity, filled for a BACKFACE REJECT as well as for a shaded hit:
            // the reject names a real surface too, and that surface is often the
            // one being hunted. (rec2/rec3 stay zero on a reject — the bake skips
            // such a hit, so it has no L and no Efb.)
            r4 = float4(surf.normal,  (float)q.CommittedInstanceIndex());
            r5 = float4(surf.worldPos, (float)q.CommittedPrimitiveIndex());

            if (dot(surf.normal, gdir) >= 0.0f)
            {
                r3.w = 1.0f; // backface reject: the bake skips this hit
            }
            else
            {
                const float NdotD = max(0.0f, dot(surf.normal, wDelta));
                float Vhit = 0.0f;
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

                const float3 Efb = EvalResponseIndirect(respPrev, visMom, dir, surf.worldPos,
                                                        surf.normal, dims, gridMin, spacing);

                const float3 diffAlbedo = surf.albedo * (1.0f - surf.metallic);
                const float3 L = diffAlbedo / PI * (NdotD * Vhit + Efb);

                float basis[9];
                EvalSH9Basis(gdir, basis);
                [unroll]
                for (uint lm = 0; lm < 9; ++lm)
                    acc[lm] += basis[lm] * L;

                r1.w = NdotD;
                r2   = float4(L, Vhit);
                r3   = float4(Efb, 2.0f);
            }
        }

        debug[recBase + r * RAY_RECORD_F4 + 0u] = r0;
        debug[recBase + r * RAY_RECORD_F4 + 1u] = r1;
        debug[recBase + r * RAY_RECORD_F4 + 2u] = r2;
        debug[recBase + r * RAY_RECORD_F4 + 3u] = r3;
        debug[recBase + r * RAY_RECORD_F4 + 4u] = r4;
        debug[recBase + r * RAY_RECORD_F4 + 5u] = r5;
    }

    // --- Self-check: store the same normalised SH9 the bake stores. ---
    const float norm = (4.0f * PI / (float)GATHER_RAYS);
    const uint tBase = RAY_DEBUG_TABLE_BASE + dir * BAKED_GI_RESPONSE_F4;
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
        debug[tBase + lm2] = float4(acc[lm2] * norm * BakedGIA_l(lm2), 0.0f);
    debug[tBase + 9u] = float4(V, 0.0f, 0.0f, 0.0f);
}
