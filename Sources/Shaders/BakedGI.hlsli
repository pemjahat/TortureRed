#ifndef BAKED_GI_HLSLI
#define BAKED_GI_HLSLI

// =============================================================================
// BakedGI.hlsli — shared helpers for the baked GI probe system.
//
// Representation :
//   Per probe and per delta direction W_i (16 fibonacci-hemisphere dirs), the
//   bake stores the INDIRECT-ONLY directional-irradiance SH9 RGB response to a
//   unit delta light at W_i, plus the direct-delta visibility V_i. Both the sun
//   and the sky recombine from the same table at runtime:
//     e_sky = sum_i wq * L_sky(W_i) * [ R_i  +  A_l Y_lm(W_i) V_i ]
//             (sky bounces)              (occluded first-bounce sky)
//     e_sun = E_sun * interp_j(R_j)      (sun bounces only — receiver direct
//                                          sun stays with the runtime-exact path)
//   Step 3 adds a per-probe octahedral 16x16 depth-moment map (see the
//   visibility section below): the deferred fetch weights the probe blend by
//   a Chebyshev occlusion bound from those moments (geometry stored between a
//   probe and the receiver rejects that probe), plus a backface weight and
//   the DDGI 2021 unified self-shadow bias.
// =============================================================================

#define BAKED_GI_DIRECTIONS    16
#define BAKED_GI_RESPONSE_F4   10  // float4s per (probe, dir): [0..8] = SH9 RGB lm coeffs, [9].x = V_i

// Fibonacci-hemisphere direction set (deterministic — bake and update MUST agree).
float3 BakedGIDeltaDirection(uint i)
{
    float y = ((float)i + 0.5f) / (float)BAKED_GI_DIRECTIONS;
    float phi = (float)i * 2.39996323f; // golden angle
    float r = sqrt(max(0.0f, 1.0f - y * y));
    return normalize(float3(r * cos(phi), y, r * sin(phi)));
}

// Lambertian cosine-convolution band factors (Ramamoorthi 2001): A0 = pi,
// A1 = 2pi/3, A2 = pi/4 — matches Sky_ProjectSH9.hlsl's irradiance basis.
// Takes the SH9 COEFFICIENT index lm (0..8): band 0 = {0}, band 1 = {1,2,3},
// band 2 = {4..8} (EvalSH9Basis ordering).
float BakedGIA_l(uint lm)
{
    const uint band = (lm <= 0u) ? 0u : ((lm <= 3u) ? 1u : 2u);
    return (band == 0u) ? PI : ((band == 1u) ? 2.09439510f : 0.78539816f);
}

// Flat probe-cell index from a 3D cell (row-major XYZ).
uint BakedGIFlatten(uint3 c, uint3 dims)
{
    return (c.z * dims.y + c.y) * dims.x + c.x;
}

// ---------------------------------------------------------------------------
// Step 3 — per-probe octahedral depth-moment visibility.
//
// Each probe stores a 16x16 octahedral map of depth moments: texel t holds
// (M1, M2) = (mean, mean-squared) hit distance over 4 rays jittered within
// the texel's cone (BakedGI_Visibility.hlsl); misses and hits beyond
// 2 x probe spacing count as sky distance. Pure geometry — TOD-invariant,
// baked once with the transport. The padded 18x18 layout duplicates the
// seam-wrapped neighbor into a one-texel border so the manual bilinear below
// is continuous across the octahedral fold.
//
// The sky-distance clamp is 2 x spacing (DDGI's convention), NOT a large
// constant: cage receivers are always within ~1.5 spacings of their probes,
// so anything beyond 2 spacings is equally "sky" for the Chebyshev test —
// but its M2 is NOT equivalent for the BILINEAR mix. A 100 m sky texel
// (M2 = 10000) contaminating a seam texel (0.2 m) at even 0.5% bilinear
// weight explodes the blended variance (var ~55) and the Chebyshev bound
// then passes the probe — a jagged per-pixel accept/reject staircase along
// geometry seams (capture-verified, ProbeGI3.rdc). With sky = 2 x spacing
// the same mix keeps var ~0.02 and both sides of the seam reject. The small
// clamp also keeps M2 fp16-representable for the future step-4 compression.
// ---------------------------------------------------------------------------
#define BAKED_GI_VIS_RES        16
#define BAKED_GI_VIS_PAD        18
#define BAKED_GI_VIS_STRIDE     (BAKED_GI_VIS_PAD * BAKED_GI_VIS_PAD) // 324 float2 per probe
#define BAKED_GI_SKY_DISTANCE_SPACINGS 2.0f // sky clamp = this x spacing (DDGI convention; see comment above)
#define BAKED_GI_CHEB_MIN       0.0f  // floor on the cubed Chebyshev weight — DELIBERATELY ZERO here
                                       // (DDGI/Adria ship ~0.05: their moments are re-traced per frame
                                       // and they have no fallback path. Our moments are static and the
                                       // tier ladder owns anti-black. A nonzero floor leaks floor x HDR
                                       // through around-the-corner cage members that pass the backface
                                       // test — the corner-band artifact. Raise only for A/B.)
#define BAKED_GI_VIS_DISPATCH_X 4096u // visibility-bake dispatch X tile (C++ mirrors this)

// --- Octahedral mapping (unit direction <-> [0,1]^2), Y-up on the xz plane ---
float2 BakedGISignNotZero(float2 v)
{
    return float2((v.x >= 0.0f) ? 1.0f : -1.0f, (v.y >= 0.0f) ? 1.0f : -1.0f);
}

float2 BakedGIOctWrap(float2 v)
{
    return (float2(1.0f, 1.0f) - abs(v.yx)) * BakedGISignNotZero(v);
}

float2 BakedGIEncodeOcta(float3 d)
{
    float3 n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    float2 uv = n.xz;
    if (n.y < 0.0f)
        uv = BakedGIOctWrap(uv);
    return uv * 0.5f + 0.5f;
}

float3 BakedGIDecodeOcta(float2 uv)
{
    float2 t = uv * 2.0f - 1.0f;
    float3 n = float3(t.x, 1.0f - abs(t.x) - abs(t.y), t.y);
    if (n.y < 0.0f)
        n.xz = BakedGIOctWrap(n.xz);
    return normalize(n);
}

// Seam-correct wrap for bilinear taps that fall off the 16x16 interior map:
// crossing an octahedral edge = crossing the fold's cut, whose spherical
// neighbor is the edge-mirrored texel (derived from the cut structure; the
// double application handles corner taps).
int2 BakedGIWrapVisTexel(int2 c)
{
    [unroll]
    for (uint it = 0; it < 2u; ++it)
    {
        if      (c.x < 0)  c = int2(0,  15 - c.y);
        else if (c.x > 15) c = int2(15, 15 - c.y);
        else if (c.y < 0)  c = int2(15 - c.x, 0);
        else if (c.y > 15) c = int2(15 - c.x, 15);
    }
    return clamp(c, int2(0, 0), int2(15, 15));
}

// Padded-map (18x18) linear index of an interior texel.
uint BakedGIVisPadIndex(int2 c)
{
    return (uint)(c.y + 1) * BAKED_GI_VIS_PAD + (uint)(c.x + 1);
}

// Manual bilinear (M1, M2) fetch from a probe's padded moment map along dir.
float2 BakedGISampleMoments(StructuredBuffer<float2> vis, uint probe, float3 dir)
{
    float2 uv = BakedGIEncodeOcta(dir) * (float)BAKED_GI_VIS_RES; // texel space [0,16]
    float2 st = uv - 0.5f;                    // corner space; floor(st) in [-1,15]
    float2 fr = frac(st);
    int2   p0 = int2(floor(st));

    float2 m00 = vis[probe * BAKED_GI_VIS_STRIDE + BakedGIVisPadIndex(BakedGIWrapVisTexel(p0 + int2(0, 0)))];
    float2 m10 = vis[probe * BAKED_GI_VIS_STRIDE + BakedGIVisPadIndex(BakedGIWrapVisTexel(p0 + int2(1, 0)))];
    float2 m01 = vis[probe * BAKED_GI_VIS_STRIDE + BakedGIVisPadIndex(BakedGIWrapVisTexel(p0 + int2(0, 1)))];
    float2 m11 = vis[probe * BAKED_GI_VIS_STRIDE + BakedGIVisPadIndex(BakedGIWrapVisTexel(p0 + int2(1, 1)))];
    return lerp(lerp(m00, m10, fr.x), lerp(m01, m11, fr.x), fr.y);
}

// One-sided Chebyshev bound (two-moment / VSM form): the probability weight
// that the probe's stored geometry along the sampled direction sits BEYOND
// the receiver distance d. d <= mean -> nothing between -> fully visible.
// The bound is CUBED — the sharpening shipped in DDGI / simco50's
// D3D12_Research / Adria: the raw two-moment bound decays slowly, so a wall
// between probe and receiver still leaves a large weight; cubing sharpens
// the rejection curve while moment variance (window frames, foliage) keeps
// a soft penumbra.
//
// NO floor (unlike DDGI/Adria's 0.05): their floor is anti-flicker insurance
// for per-frame re-traced moments and a substitute for a fallback path. Our
// moments are baked and static — nothing flickers — and the tier ladder
// already handles total cage rejection. A 5% floor here leaked ~5% of sunlit
// HDR probes through around-the-corner cage members that pass the backface
// test (probes across a flat wall fail it; probes around a corner do not):
// the Sponza corner-band leak, verified absent in the path tracer.
float BakedGIChebyshev(float2 m, float d)
{
    if (d <= m.x)
        return 1.0f;
    float variance = max(m.y - m.x * m.x, 0.0f);
    float cheb = variance / (variance + (d - m.x) * (d - m.x));
    return max(cheb * cheb * cheb, BAKED_GI_CHEB_MIN); // cubed: sharper leak rejection
}

// ---------------------------------------------------------------------------
// Deferred fetch (Lighting.hlsl): DDGI-style weighted probe blend over the
// lit probe grid.
//
//   w_p = trilinear_p(x') * backface_p * chebyshev_p
//
//   x'        — receiver position advanced by the DDGI 2021 unified
//               self-shadow bias (n*0.2 + wo*0.8) * (0.75*D)*B, B = 0.3:
//               slides the query off the receiver's own surface so the
//               probe's stored geometry (which includes that surface) does
//               not occlude the receiver itself.
//   backface  — saturate(dot(N, dir to probe)): cage members behind the
//               receiver's tangent plane (through-floor / through-wall)
//               contribute zero.
//   chebyshev — one-sided bound from the probe's baked octahedral depth
//               moments along the probe->receiver direction: geometry
//               stored between the probe and the receiver rejects the probe.
//
// Fallback ladder (never black, never worse than the step-1 fetch): if every
// cage probe is Chebyshev-rejected, retry with backface-only weights; if
// those also die, degrade to plain validity-renormalized trilinear.
//
// A/B toggle (FrameCB.bakedGIVisCheck): 0 disables all three weights AND the
// self-shadow bias — the fetch is then exactly the step-1 plain
// validity-renormalized trilinear, for before/after comparison.
//
// REQUIRES in scope at include time:
//   - ConstantBuffer<FrameConstants> FrameCB : register(b0)
//   - EvalSH9Basis (PBR.hlsl, included via Common.hlsl / CommonTracing.hlsl)
// ---------------------------------------------------------------------------
float3 SampleBakedGIProbe(float3 worldPos, float3 N, float3 V)
{
    uint3  dims    = uint3(FrameCB.bakedGIDimX, FrameCB.bakedGIDimY, FrameCB.bakedGIDimZ);
    float3 gridMin = float3(FrameCB.bakedGIGridMinX, FrameCB.bakedGIGridMinY, FrameCB.bakedGIGridMinZ);
    float  spacing = FrameCB.bakedGISpacing;

    // A/B toggle: 1 = step-3 weighted fetch, 0 = plain step-1 trilinear.
    const bool visOn = (FrameCB.bakedGIVisCheck != 0u);

    // Unified self-shadow bias (DDGI 2021, B = 0.3). The biased point also
    // drives the trilinear cell lookup — up to ~0.23*D of cage drift, by design.
    float3 xq = visOn
        ? worldPos + (N * 0.2f + V * 0.8f) * (0.75f * spacing * 0.3f)
        : worldPos; // A/B off: unbiased cage — exactly the step-1 fetch

    // Continuous cell coordinates; probes live on integer cells in [0, dim-1].
    float3 t = clamp((xq - gridMin) / spacing,
                     float3(0.0f, 0.0f, 0.0f), dims - 1.0f);
    int3   base = int3(min(floor(t), max(dims - 2.0f, float3(0.0f, 0.0f, 0.0f))));
    float3 f    = t - base;

    StructuredBuffer<float4> lit  = ResourceDescriptorHeap[FrameCB.bakedGIProbeSRVIndex];
    StructuredBuffer<uint>   meta = ResourceDescriptorHeap[FrameCB.bakedGIProbeMetaSRVIndex];
    StructuredBuffer<float2> vis  = ResourceDescriptorHeap[FrameCB.bakedGIProbeVisSRVIndex];

    // Pass 1 — cage weights: validity * trilinear * backface * Chebyshev.
    float wTri[8] = (float[8])0;
    float wBF[8]  = (float[8])0;
    float wFull[8] = (float[8])0;
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        int3  off = int3(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        float3 w3 = lerp(float3(1.0f, 1.0f, 1.0f) - f, f, off);
        float w = w3.x * w3.y * w3.z;

        uint flat = BakedGIFlatten(base + off, dims);
        if (!(meta[flat] & 1u))
            continue;
        wTri[k] = w;

        if (!visOn)
            continue; // A/B off: plain step-1 fetch — wBF/wFull stay 0 -> tier 0

        float3 toProbe = gridMin + float3(base + off) * spacing - xq;
        float  dist    = length(toProbe);
        if (dist < 1e-4f)
        {
            // Receiver sits on the probe: trivially visible.
            wBF[k] = w;
            wFull[k] = w;
            continue;
        }

        float3 dir  = toProbe / dist;
        float  back = saturate(dot(N, dir));
        wBF[k] = w * back;
        if (back > 0.0f)
        {
            // Moment lookup: the map stores rays fired FROM the probe, so the
            // occlusion test needs the probe->receiver direction — the NEGATION
            // of dir (dir = receiver->probe, which the backface test above
            // correctly uses as-is; simco50's DDGI does the same negation:
            // GetDDGIProbeUV(..., -probeDirection, ...)). Sampling at +dir
            // read the probe's FAR hemisphere (usually open sky, M1 = sky
            // distance -> weight 1) so walled-off probes were never rejected —
            // the Sponza intersection leak, capture-verified (ProbeGI2.rdc).
            float2 m = BakedGISampleMoments(vis, flat, -dir);
            wFull[k] = wBF[k] * BakedGIChebyshev(m, dist);
        }
    }

    // Tier selection: full -> backface-only -> plain validity-renormalized.
    float sumFull = 0.0f, sumBF = 0.0f;
    [unroll]
    for (uint k1 = 0; k1 < 8; ++k1) { sumFull += wFull[k1]; sumBF += wBF[k1]; }
    const uint tier = (sumFull > 1e-4f) ? 2u : ((sumBF > 1e-4f) ? 1u : 0u);

    // Pass 2 — blend SH coefficients with the selected tier's weights.
    float3 coef[9] = (float3[9])0;
    float  wsum = 0.0f;
    [unroll]
    for (uint k2 = 0; k2 < 8; ++k2)
    {
        float w = (tier == 2u) ? wFull[k2] : ((tier == 1u) ? wBF[k2] : wTri[k2]);
        if (w <= 0.0f)
            continue;
        int3 off = int3(k2 & 1u, (k2 >> 1u) & 1u, (k2 >> 2u) & 1u);
        uint flat = BakedGIFlatten(base + off, dims);
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
            coef[lm] += lit[flat * 9u + lm].rgb * w;
        wsum += w;
    }

    if (wsum < 1e-6f)
        return float3(0.0f, 0.0f, 0.0f);

    // Evaluate directional irradiance at N (lit coefficients use the same
    // plain-SH convention as the sky SH9 buffer).
    float basis[9];
    EvalSH9Basis(N, basis);
    float3 E = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
    {
        coef[lm2] /= wsum;
        E += coef[lm2] * basis[lm2];
    }
    return max(E, 0.0f.xxx);
}

// ---------------------------------------------------------------------------
// Diagnostic split of the probe cage (leak map): returns (wValid, wBack) —
// the validity-renormalized weight total and the portion contributed by
// probes BEHIND the receiver's tangent plane (dot(N, probePos - x) <= 0).
// wBack / wValid is the backface-contamination fraction the Backface Leak Map
// displays; it is exactly the weight backface rejection would remove.
// Deliberately the PLAIN cage (no self-shadow bias, no backface/Chebyshev
// weights): this measures what the raw trilinear fetch WOULD leak, as the
// baseline the step-3 weights are judged against.
// REQUIRES in scope at include time: ConstantBuffer<FrameConstants> FrameCB.
// ---------------------------------------------------------------------------
float2 BakedGICageContamination(float3 worldPos, float3 N)
{
    uint3  dims    = uint3(FrameCB.bakedGIDimX, FrameCB.bakedGIDimY, FrameCB.bakedGIDimZ);
    float3 gridMin = float3(FrameCB.bakedGIGridMinX, FrameCB.bakedGIGridMinY, FrameCB.bakedGIGridMinZ);

    // Identical cage computation to SampleBakedGIProbe.
    float3 t = clamp((worldPos - gridMin) / FrameCB.bakedGISpacing,
                     float3(0.0f, 0.0f, 0.0f), dims - 1.0f);
    int3   base = int3(min(floor(t), max(dims - 2.0f, float3(0.0f, 0.0f, 0.0f))));
    float3 f    = t - base;

    StructuredBuffer<uint> meta = ResourceDescriptorHeap[FrameCB.bakedGIProbeMetaSRVIndex];

    float wValid = 0.0f;
    float wBack  = 0.0f;
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        int3  off = int3(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        float3 w3 = lerp(float3(1.0f, 1.0f, 1.0f) - f, f, off);
        float w = w3.x * w3.y * w3.z;

        uint flat = BakedGIFlatten(base + off, dims);
        if (meta[flat] & 1u)
        {
            wValid += w;
            // Sign test only — normalization cannot flip the sign.
            float3 probePos = gridMin + float3(base + off) * FrameCB.bakedGISpacing;
            if (dot(N, probePos - worldPos) <= 0.0f)
                wBack += w;
        }
    }
    return float2(wValid, wBack);
}

#endif // BAKED_GI_HLSLI
