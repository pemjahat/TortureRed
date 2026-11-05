#ifndef BAKED_GI_HLSLI
#define BAKED_GI_HLSLI

// =============================================================================
// BakedGI.hlsli — shared helpers for the baked GI probe system.
//
// Representation :
//   Per probe and per delta direction W_i (16 fibonacci-hemisphere dirs), the
//   bake stores the INDIRECT-ONLY directional-irradiance SH9 RGB response to a
//   unit delta light at W_i. One record holds 9 float4 and nothing else — there
//   is no visibility scalar in it. The 16 records are the SKY quadrature basis:
//     e_sky = sum_i wq * L_sky(W_i) * [ R_i  +  A_l Y_lm(W_i) V(W_i) ]
//             (sky bounces)              (occluded first-bounce sky)
//   V(W_i) is the DIRECT sky visibility. It comes from one per-probe raw SH9
//   field, BAKED_GI_SKYVIS_SH, which the visibility pass projects from the same
//   1024 rays it traces for the depth moments. The fold reconstructs that field
//   at W_i. Nothing per-bin is stored for it.
//   Per probe and per SUN HOUR STATE k (17 states along the equinox arc), the
//   bake stores the same kind of response for a unit light at the sun's direction
//   at hour 6 + 0.75 k. The fold interpolates TWO of them:
//     e_sun = E_sun * lerp(R[i0], R[i0+1], t)   (sun bounces only — the
//                                               receiver's direct sun stays with
//                                               the runtime-exact path)
//   Step 3 adds a per-probe octahedral 16x16 depth-moment map (see the
//   visibility section below): the deferred fetch weights the probe blend by
//   a Chebyshev occlusion bound from those moments (geometry stored between a
//   probe and the receiver rejects that probe), plus a backface weight and
//   the DDGI 2021 unified self-shadow bias.
// =============================================================================

#define BAKED_GI_DIRECTIONS    16
#define BAKED_GI_RESPONSE_F4   9   // float4s per (probe, dir): [0..8] = SH9 RGB lm coeffs. No slot 9.

// Fibonacci-hemisphere direction set (deterministic — bake and update MUST agree).
// These 16 directions are the SKY QUADRATURE basis. They are not the sun: the sun
// has its own 17 hour states below. The two sets coexist on purpose.
float3 BakedGIDeltaDirection(uint i)
{
    float y = ((float)i + 0.5f) / (float)BAKED_GI_DIRECTIONS;
    float phi = (float)i * 2.39996323f; // golden angle
    float r = sqrt(max(0.0f, 1.0f - y * y));
    return normalize(float3(r * cos(phi), y, r * sin(phi)));
}

// ---------------------------------------------------------------------------
// Equinox hour states — the sun direction set.
//
// The sun is not a free direction. The date is fixed at the equinox, so the sun
// path is a great circle and the arc above the horizon is exactly 180 degrees at
// every latitude. One parameter covers it: the hour of TRUE SOLAR TIME, where
// hour 12 is solar noon by definition. The equation of time, the longitude and
// the timezone are therefore not inputs.
//
//   H = 15 * (hour - 12) degrees        (the hour angle; Unreal SunPosition
//                                        uses the same expression)
//   dir = ( -sin H,  cos(lat) cos H,  sin(lat) cos H )
//
// Unit length by construction: sin^2 H + cos^2 H = 1, and the y and z terms are
// one cosine split over two axes. No normalize, so there is no near-zero
// denominator and no NaN case.
//
// MUST match BakedGI::SunDirectionAtHour in BakedGI.cpp. One formula, one CPU
// mirror. A drift between the two puts the bake and the fold on different arcs,
// and nothing reports it.
// ---------------------------------------------------------------------------
#define BAKED_GI_SUN_STATES      17
#define BAKED_GI_SUN_STATE_HOUR0 6.0f   // state 0 is at sunrise
#define BAKED_GI_SUN_STATE_STEP  0.75f  // 17 states over 12 hours, 11.25 degrees apart

float3 BakedGISunDirectionAtHour(float hour, float latRad)
{
    const float H = radians(15.0f * (hour - 12.0f));
    return float3(-sin(H), cos(latRad) * cos(H), sin(latRad) * cos(H));
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
// Ray-debug delta source. ONE value, three spellings: these two, the C++
// BakedGI::RayDebugSource (BakedGI.h), and the uint BakedGIRayDebugParams::source
// (SharedTypes.h).
//
// The bake stores two response tables for each probe, with different strides. The
// sky table has BAKED_GI_DIRECTIONS (16) records, indexed by the fibonacci bin.
// The sun table has BAKED_GI_SUN_STATES (17) records, indexed by the hour state. A
// replay that reads the wrong stride reads another probe's record, so the source
// selects the table, the stride, the record index, the delta direction and the RNG
// seed together. Do not renumber: 0 must stay the shipped sun behaviour.
// ---------------------------------------------------------------------------
#define BAKED_GI_RAYDEBUG_SOURCE_SUN 0u
#define BAKED_GI_RAYDEBUG_SOURCE_SKY 1u

// ---------------------------------------------------------------------------
// Ray-debug record, rec3.w. ONE definition, two users: BakedGI_RayDebug.hlsl
// encodes it and BakedGI_Debug.hlsl decodes it. It lives here so the two cannot
// drift.
//
// The 6-float4 record this packing was written against had no spare field, so
// the STATUS and the SOURCE RAY INDEX share rec3.w:
//
//     rec3.w = status + 4 * sourceRayIndex
//
//   status          0 = sky miss, 1 = backface reject, 2 = shaded
//   sourceRayIndex  flattened batch-local index: batch * BAKE_RAYS + ray
//
// Exactness: fp32 represents every integer up to 2^24. Eight batches give a
// largest packed value of 2 + 4 * 2047 = 8190, so fmod and the round trip stay exact.
//
// WARNING: rec3.w is NOT the status any more. Compare it to 0.5 or 2.0 direct
// and you read the packed value. Always go through these two.
//
// rec6 and rec7 were APPENDED later (stride 6 -> 8 float4). They carry the
// SECONDARY ray, so this packing was never reopened. See
// BAKED_GI_RAY_RECORD_F4 in SharedTypes.h for the full layout, and
// BakedGIRayRecShadowStatus below for the secondary status.
// ---------------------------------------------------------------------------
float BakedGIRayRecStatus(float packed)
{
    return fmod(packed, 4.0f);
}

uint BakedGIRayRecSourceRay(float packed)
{
    return (uint)((packed - fmod(packed, 4.0f)) * 0.25f + 0.5f);
}

// The secondary ray status, read from rec6.w. It is a PLAIN float, so a compare
// is enough and no decode is needed. The names come from SharedTypes.h
// (BAKED_GI_RAY_SHADOW_NONE / _OPEN / _BLOCKED), so the writer, the viewer and
// the CPU readback test the same three values.
//
//   NONE     no shadow ray was traced: the record is not shaded, or NdotD <= 0
//   OPEN     the shadow ray committed nothing, so Vhit is 1 and rec6.xyz is W
//   BLOCKED  the shadow ray committed at least one opaque hit, so Vhit is 0 and
//            rec6.xyz is that hit's world position
float BakedGIRayRecShadowStatus(float rec6w)
{
    return rec6w;
}

// ---------------------------------------------------------------------------
// Probe meta (validity) bit layout — written once per bake by the
// classification stage in BakedGI_Visibility.hlsl, read by the transport bake,
// the runtime fetch, the update pass, and the debug views.
//   bit 0: VALID under the DDGI classification — THE live bit every consumer
//          gates on (kept at bit 0 so existing `meta & 1` checks are unchanged)
//   bit 1: valid under the OLD 1cm existential test (diagnostic only — lets
//          the placement view and bake stats show the flip between criteria)
//   bit 2: dead by DDGI rule 1 — >= 25% of the 32 stable rays hit backfaces
//   bit 3: dead by DDGI rule 2 — nearest frontface beyond 3x spacing
//          (unreachable while depths clamp at 2x spacing — DDGI's own code
//          neuters this rule identically; kept for fidelity)
//   bit 4: FILLED by bake-time donor dilation; runtime data exists, but the
//          fixed-origin depth moments remain untrusted for Chebyshev weights
// ---------------------------------------------------------------------------
#define BAKED_GI_META_VALID         0x1u
#define BAKED_GI_META_OLD_VALID     0x2u
#define BAKED_GI_META_DEAD_BACKFACE 0x4u
#define BAKED_GI_META_DEAD_FAR      0x8u
#define BAKED_GI_META_FILLED        0x10u

// Bake, dilation, and debug consumers use this helper: data exists when either
// geometric validity or donor-backed fill is present.
bool BakedGIProbeHasData(uint meta)
{
    return (meta & (BAKED_GI_META_VALID | BAKED_GI_META_FILLED)) != 0u;
}

// Runtime receiver eligibility. VALID always qualifies. FILLED qualifies only
// when the Application publishes raw-versus-dilated runtime mode as dilated.
// Raw A/B mode therefore ignores every FILLED bit without changing the meta buffer.
bool BakedGIProbeHasRuntimeData(uint meta)
{
    if ((meta & BAKED_GI_META_VALID) != 0u)
        return true;
    return (FrameCB.bakedGIUseFilled != 0u) && ((meta & BAKED_GI_META_FILLED) != 0u);
}

// DDGI's classification ray set (D3D12_Research DDGICommon.hlsli,
// DDGI_NUM_STABLE_RAYS = 32): fixed full-sphere spherical-fibonacci
// directions. Per-frame systems keep them temporally stable so probe states
// don't flicker; bake-time here that just means deterministic — every bake
// classifies with the same 32 directions.
#define BAKED_GI_CLASSIFY_RAYS 32

// Spherical fibonacci over the FULL sphere (DDGI's SphericalFibonacci,
// faithful copy — z is the pole).
float3 BakedGISphericalFibonacci(uint i, uint n)
{
    const float PHI = sqrt(5.0f) * 0.5f + 0.5f;
    float fraction = (i * (PHI - 1.0f)) - floor(i * (PHI - 1.0f));
    float phi       = 2.0f * PI * fraction;
    float cosTheta  = 1.0f - (2.0f * i + 1.0f) / (float)n;
    float sinTheta  = sqrt(saturate(1.0f - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float3 BakedGIClassifyDirection(uint i, uint n)
{
    return BakedGISphericalFibonacci(i, n);
}

// One deterministic stream for each (probe, response record, batch). The
// iteration is absent on purpose: every transport term uses the same samples.
uint BakedGIBatchSeed(uint probe, uint seedDir, uint batchIndex)
{
    return pcg_hash(probe * 131u + seedDir * 7919u + batchIndex * 0x85ebca6bu
                    + 0x9e3779b9u);
}

// Uniform random rotation from three nested hashes. Rodrigues' formula keeps
// the fibonacci set intact while each batch covers a different orientation.
float3x3 BakedGIBatchRotation(uint batchSeed)
{
    const uint h0 = pcg_hash(batchSeed);
    const uint h1 = pcg_hash(h0);
    const uint h2 = pcg_hash(h1);
    const float invU32 = 1.0f / 4294967296.0f;
    const float z   = 1.0f - 2.0f * float(h0) * invU32;
    const float phi = 2.0f * PI * float(h1) * invU32;
    const float angle = 2.0f * PI * float(h2) * invU32;
    const float r = sqrt(max(0.0f, 1.0f - z * z));
    const float3 axis = float3(r * cos(phi), r * sin(phi), z);
    const float s = sin(angle);
    const float c = cos(angle);
    const float t = 1.0f - c;
    const float x = axis.x, y = axis.y, zAxis = axis.z;
    return float3x3(
        t * x * x + c,          t * x * y - s * zAxis, t * x * zAxis + s * y,
        t * x * y + s * zAxis,  t * y * y + c,         t * y * zAxis - s * x,
        t * x * zAxis - s * y,  t * y * zAxis + s * x, t * zAxis * zAxis + c);
}

float3 BakedGIRotatedFibonacci(uint i, uint n, float3x3 rotation)
{
    return normalize(mul(rotation, BakedGISphericalFibonacci(i, n)));
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

// ---------------------------------------------------------------------------
// SKY-VISIBILITY SH9 — the ONE source of the V in the runtime fold's
// `A_l Y_lm(W_i) V(W_i)` term.
//
// This buffer holds 9 RAW SH9 coefficients for each probe, projected from the
// SAME 16x16x4 = 1024 visibility rays the moment map already traces, so it adds
// no traversal and no ray budget:
//
//     L_lm = (4pi / 1024) * sum_s V_s * Y_lm(d_s)     V_s binary, 0 or 1
//
// This field costs no shadow ray and carries the whole visibility field, so the
// fold either reconstructs it or zeros V — there is no third source.
//
// NO cosine A_l factor and NO normalisation by Y00: the runtime fold multiplies
// V by a per-bin response, so it needs the visibility FIELD, and not an
// irradiance. Reconstruct with plain EvalSH9Basis and clamp to [0, 1].
//
// WHY THE CLAMP IS MANDATORY. A band-limited reconstruction overshoots. The
// sun-visibility thread measured a band-2 fit of a visibility field leaving
// [0, 1] by +28 percent (ARCHITECTURE-TortureRed-Visibility: "A normalised
// kernel is a convex combination; an SH projection is a reconstruction"). A mean
// cannot leave [0, 1]. This field can, so the fold clamps it back into the range
// the quantity must stay in.
//
// Accumulation is an equal-weight mean over the sample distribution, so a fully
// open probe reconstructs to exactly 1: (4pi/N) * N * Y00 * Y00 = 4pi * Y00^2
// = 1. The octahedral sample density is not exactly uniform in solid angle, so
// the fit is a low-order approximation of the field.
//
// WHAT TO EXPECT FROM THIS FIELD. Band 2 resolves about two to three bins, so a
// LOCALISED aperture spreads over its neighbours instead of staying in one bin.
// A CPU mirror of this projection reports a 20-degree aperture on one bin axis
// as 0.40 on that bin and 0.14 on a neighbour. A ground plane reads 0.45..0.59
// across the low bins. Smoothing of that size is the expected behaviour of a
// band-2 field, not a defect.
//
// MEMORY. 9 floats = 36 B for each probe. 418,176 B at 11,616 probes and
// 4.72 MB at the 131,072-probe cap. It is a bake output, not a per-frame
// payload: the fold reads 36 B for each probe on a fold frame and nothing on a
// clean one.
// ---------------------------------------------------------------------------
#define BAKED_GI_SKYVIS_SH_COEFFS 9u

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
//               not occlude the receiver itself. x' drives the trilinear CELL
//               and the Chebyshev DISTANCE. It does NOT drive the backface
//               plane — see below.
//   backface  — saturate(dot(N, dir to probe)): cage members behind the
//               receiver's tangent plane (through-floor / through-wall)
//               contribute zero. Measured from the UNPUSHED receiver: the
//               tangent plane is a property of the surface. Taking it from x'
//               deletes every member inside the bias of the surface, which on
//               Sponza (0.0601 m wall-to-layer gap, 0.225 m bias) is the
//               whole adjacent layer — the one holding ~94% of the trilinear
//               weight — and renormalises the receiver onto the layer 1 m
//               behind it, through the wall. Measured r/g 3.16 vs 1.04.
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
// The lit table is an explicit parameter, not a fixed FrameCB read: the update
// pass writes the TOTAL, the sky-sourced and the sun-sourced SH9 tables
// separately, and the indirect-only debug view (Lighting.hlsl) needs to fetch
// any one of the three through this same filter. SampleBakedGIProbe below is
// the normal path and always reads the TOTAL table.
//
// REQUIRES in scope at include time:
//   - ConstantBuffer<FrameConstants> FrameCB : register(b0)
//   - EvalSH9Basis (PBR.hlsl, included via Common.hlsl / CommonTracing.hlsl)
// ---------------------------------------------------------------------------
float3 SampleBakedGIProbeFrom(uint litSrvIndex, float3 worldPos, float3 N, float3 V)
{
    uint3  dims    = uint3(FrameCB.bakedGIDimX, FrameCB.bakedGIDimY, FrameCB.bakedGIDimZ);
    float3 gridMin = float3(FrameCB.bakedGIGridMinX, FrameCB.bakedGIGridMinY, FrameCB.bakedGIGridMinZ);
    float  spacing = FrameCB.bakedGISpacing;

    // A/B toggle: 1 = step-3 weighted fetch, 0 = plain step-1 trilinear.
    const bool visOn = (FrameCB.bakedGIVisCheck != 0u);

    // Unified self-shadow bias (DDGI 2021, B = 0.3). The biased point also
    // drives the trilinear cell lookup — up to ~0.23*D of cage drift, by
    // design — but it does NOT drive the backface plane below, which is
    // measured from the unpushed receiver. Do not merge the two again.
    float3 xq = visOn
        ? worldPos + (N * 0.2f + V * 0.8f) * (0.75f * spacing * 0.3f)
        : worldPos; // A/B off: unbiased cage — exactly the step-1 fetch

    // Continuous cell coordinates; probes live on integer cells in [0, dim-1].
    float3 t = clamp((xq - gridMin) / spacing,
                     float3(0.0f, 0.0f, 0.0f), dims - 1.0f);
    int3   base = int3(min(floor(t), max(dims - 2.0f, float3(0.0f, 0.0f, 0.0f))));
    float3 f    = t - base;

    StructuredBuffer<float4> lit  = ResourceDescriptorHeap[litSrvIndex];
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
        const uint metaWord = meta[flat];
        if (!BakedGIProbeHasRuntimeData(metaWord))
            continue;
        const bool filled = (metaWord & BAKED_GI_META_FILLED) != 0u;
        wTri[k] = w;

        if (!visOn)
            continue; // A/B off: plain step-1 fetch — wBF/wFull stay 0 -> tier 0

        float3 p       = gridMin + float3(base + off) * spacing;
        float3 toProbe = p - xq;
        float  dist    = length(toProbe);
        if (dist < 1e-4f)
        {
            // A filled origin has no trusted moments, even at zero distance.
            wBF[k] = w;
            wFull[k] = filled ? 0.0f : w;
            continue;
        }

        float3 dir = toProbe / dist;

        // Backface origin: the UNPUSHED receiver. The question this weight
        // answers is "is this cage member in front of the surface", and the
        // surface is at worldPos; xq only exists to keep the Chebyshev
        // distance off the cliff. Taken from xq, the test deleted the whole
        // adjacent probe layer on any surface closer than the bias to it, and
        // the renormalisation handed the receiver the light behind that
        // surface. Measured on Sponza: the adjacent probe held 90.2% of the
        // fetch weight with an unbiased read and 0% with this one.
        float3 toSurf = p - worldPos;
        float  dSurf  = length(toSurf);
        float  back   = (dSurf < 1e-4f) ? 1.0f : saturate(dot(N, toSurf / dSurf));

        wBF[k] = w * back;
        // Filled probes keep trilinear and backface weights but never read their
        // fixed-origin moments. They cannot enter the full Chebyshev tier.
        if (back > 0.0f && !filled)
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

// Normal runtime fetch: the TOTAL lit table (sky + sun, written every frame by
// BakedGI_Update.hlsl as the exact sum of the two debug splits).
float3 SampleBakedGIProbe(float3 worldPos, float3 N, float3 V)
{
    return SampleBakedGIProbeFrom(FrameCB.bakedGIProbeSRVIndex, worldPos, N, V);
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
        if (BakedGIProbeHasRuntimeData(meta[flat]))
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
