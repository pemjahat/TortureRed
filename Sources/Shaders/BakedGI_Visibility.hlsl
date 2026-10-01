// =============================================================================
// BakedGI_Visibility.hlsl — step-3 visibility bake: per-probe octahedral
// depth-moment maps.
//
// One thread group per probe (16x16 threads = one interior texel each). Each
// texel traces 4 distance-only rays jittered within its cone — alpha-tested
// geometry occludes (same rule as the transport bake's V_i rays) — and stores
// (M1, M2) = (mean, mean-squared) hit distance, the two-moment shadow
// representation the runtime query feeds into the one-sided Chebyshev bound
// (BakedGI.hlsli). Variance from partially covered texels (window frames,
// foliage) is what softens the leak rejection into a penumbra.
//
// The padded 18x18 layout (one-texel border duplicating the seam-wrapped
// neighbor — BakedGIWrapVisTexel) makes the query's manual bilinear
// continuous across the octahedral fold.
//
// Pure geometry — no lighting, no TOD dependence, baked once alongside the
// transport bake (dispatched from BakedGI::RecordBake on the same list).
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>    FrameCB : register(b0);
ConstantBuffer<BakedGIVisParams>  Vis     : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

#define VIS_RAYS_PER_TEXEL 4

groupshared float2 g_Moments[BAKED_GI_VIS_RES][BAKED_GI_VIS_RES];

[numthreads(BAKED_GI_VIS_RES, BAKED_GI_VIS_RES, 1)]
void main(uint3 gt : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // Dispatch tiles probes across (x = BAKED_GI_VIS_DISPATCH_X, y): the X
    // dispatch cap is 65535 groups and the probe cap is 131072.
    const uint probe = gid.y * BAKED_GI_VIS_DISPATCH_X + gid.x;
    if (probe >= Vis.probeCount)
        return; // group-uniform: safe before the groupshared barrier

    uint3 dims = uint3(Vis.dimX, Vis.dimY, Vis.dimZ);
    uint3 cell = uint3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    float3 gridMin  = float3(Vis.gridMinX, Vis.gridMinY, Vis.gridMinZ);
    float3 probePos = gridMin + float3(cell) * Vis.spacing;

    // --- Interior texel moments: 4 jittered distance-only rays ---
    RNG rng;
    rng.state = pcg_hash(probe * 7919u + gt.y * 61u + gt.x * 13u + 0x9e3779b9u);
    rng.inc   = 1;

    static const float2 kJitter[VIS_RAYS_PER_TEXEL] =
    {
        float2(-0.25f, -0.25f), float2(0.25f, -0.25f),
        float2(-0.25f,  0.25f), float2(0.25f,  0.25f)
    };

    float m1 = 0.0f, m2 = 0.0f;
    [loop]
    for (uint r = 0; r < VIS_RAYS_PER_TEXEL; ++r)
    {
        float2 uv  = ((float2(gt.xy) + 0.5f) + kJitter[r]) / (float)BAKED_GI_VIS_RES;
        float3 dir = BakedGIDecodeOcta(uv);

        RayDesc ray;
        ray.Origin    = probePos + dir * 0.001f;
        ray.Direction = dir;
        ray.TMin      = 0.001f;
        ray.TMax      = 1e4f;
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
        while (q.Proceed()) { PROCESS_ALPHA_MASK(q, rng); }

        // Misses and hits beyond the sky distance are indistinguishable to a
        // receiver that is always within ~1.5 spacings of its cage probes.
        // skyDist = 2 x spacing (DDGI convention): a large constant (e.g. 100)
        // makes sky texels' M2 (10000) poison any bilinear mix they touch —
        // a 0.5% sky weight on a seam texel explodes the variance and the
        // Chebyshev bound passes the probe (jagged seam leak, ProbeGI3.rdc).
        const float skyDist = BAKED_GI_SKY_DISTANCE_SPACINGS * Vis.spacing;
        float d = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                    ? min(q.CommittedRayT(), skyDist)
                    : skyDist;
        m1 += d;
        m2 += d * d;
    }
    g_Moments[gt.y][gt.x] = float2(m1, m2) / (float)VIS_RAYS_PER_TEXEL;

    GroupMemoryBarrierWithGroupSync();

    // --- Padded write: 324 slots cooperatively over 256 threads; border
    // texels are the seam-wrapped interior copies (see BakedGIWrapVisTexel).
    RWStructuredBuffer<float2> visOut = ResourceDescriptorHeap[Vis.visUAVIdx];
    const uint probeBase = probe * BAKED_GI_VIS_STRIDE;
    [loop]
    for (uint i = gt.y * BAKED_GI_VIS_RES + gt.x; i < BAKED_GI_VIS_STRIDE;
         i += BAKED_GI_VIS_RES * BAKED_GI_VIS_RES)
    {
        int2 c  = int2((int)(i % BAKED_GI_VIS_PAD) - 1, (int)(i / BAKED_GI_VIS_PAD) - 1); // [-1,16]
        int2 ic = BakedGIWrapVisTexel(c);
        visOut[probeBase + i] = g_Moments[ic.y][ic.x];
    }
}
