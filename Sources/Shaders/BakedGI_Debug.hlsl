// =============================================================================
// BakedGI_Debug.hlsl — probe debug overlay (post-composite LDR).
//
// Nine views (FrameConstants::bakedGIDebugView):
//   1 Placement      — cube per grid cell, classification A/B: green = valid
//                      under both criteria, red = bad under both, BLUE =
//                      newly bad (old 1cm test said valid, DDGI's >= 25%
//                      backface rule kills it), yellow = newly valid.
//   2 Lit Irradiance — billboard SH ball per valid probe: each sphere point
//                      evaluates the probe's lit SH9 coefficients at its own
//                      normal, showing the directional structure of e(p);
//                      scaled by bakedGIDebugScale (lit values are
//                      FP16Scale'd scene HDR).
//   5 Lit Sky Source  — same ball, reading the sky-sourced split buffer:
//                      sky bounces + occluded sky-direct (zero where V_i = 0
//                      for the direct part).
//   6 Lit Sun Source  — same ball, reading the sun-sourced split buffer:
//                      sun bounces only (direct sun never enters the probes).
//                      Modes 2/5/6 read the three update-pass outputs and
//                      always sum: total = sky + sun.
//   3 Sky Visibility — cube per cell, grayscale = mean of the 16 baked
//                      directional visibilities V_i (sky-open fraction);
//                      red = invalid.
//   4 Visibility Rays— thin oriented box per (probe, fibonacci direction):
//                      green = open sky (V_i = 1), red = blocked. Camera-
//                      distance culled (spacing * 8 window), invalid probes
//                      skipped.
//   7 Ray Fan        — thin oriented box for each of the 64 gather rays of the
//                      SUN-DOMINANT direction (FrameConstants::bakedGIRayDebugDir)
//                      of ONE probe (bakedGIRayDebugProbe), read from the
//                      ray-debug buffer that the debug dump writes.
//                      Length = the recorded hit distance, capped at
//                      bakedGIRayDebugLength spacings. Colour from
//                      bakedGIRayDebugChannel. No camera-distance cull: there
//                      is one probe, and the whole fan is the point.
//
//                      ONE channel swaps the geometry: DeltaDirections
//                      (BAKED_GI_RAYFAN_CH_DELTADIRS) draws the 16 fibonacci
//                      delta directions of that same probe, one thin box each,
//                      and reads NO record — so it needs no dump and works
//                      before one. Its colour is the NdotD channel's mapping
//                      (plain linear grayscale, no gain, no log):
//                      value = bakedGISunWeight[j].x / max_k(...), so WHITE is the
//                      direction the sun weights most, which is the direction
//                      the fan itself draws. The fold is probe INDEPENDENT —
//                      cos^8(dot(W_j, toSun)) never reads the probe — so the 16
//                      colours are the same at any valid cell: this channel
//                      reports the fold, not a per-cell response. The weights
//                      are the update pass's own cos^8 fold, published by the
//                      CPU, so the white box and the printed kernel-weight
//                      ranking describe one computation.
//   8 Efb Contribution — same SH ball as the lit views, but the value is
//                      ball[selected term] - ball[term 1], read from the two
//                      per-bake-term response balls that the bake's fold pass
//                      writes (BakedGI_IterFold.hlsl). By the bake's own
//                      identity R^k = R^0 + G(Efb_k) that difference IS the
//                      contribution the series-feedback term added by the
//                      selected term, so this is the one view that shows what
//                      the feedback channel painted, term by term. The DC
//                      normalization is the SELECTED term's own ball DC, so a
//                      probe's DC reads as the feedback's share of that term's
//                      transport — in [0, 1] by construction, because the
//                      contribution is non-negative and its ball can never
//                      exceed the term's own ball. Term 1 subtracts a ball from
//                      itself and is black: that is the gate working, not a
//                      failure. Selection and the term->index mapping are the
//                      app's (Application's "Iteration" combo).
//
//   9 Cage Trace    — the Efb borrow drawn as geometry, for ONE probe
//                      (bakedGIRayDebugProbe). Given that probe it DERIVES the
//                      whole chain, in this order:
//                        1. the delta direction = the sun fold's argmax
//                           (FrameConstants::bakedGIRayDebugDir, the same pick
//                           the Ray Fan draws);
//                        2. of that direction's 64 gather rays, the one ray with
//                           the largest CARRIED FEEDBACK among the rays the
//                           feedback DOMINATES (a shaded ray whose Efb term
//                           exceeds its NdotD*Vhit term) — the diagnostic found
//                           is that such a ray's radiance was gathered from a
//                           different cell than the one its own hit point
//                           addresses. A ranking cannot reach a ray that is
//                           merely NOT the brightest, so
//                           FrameConstants::bakedGIRayDebugRay lets the UI name
//                           the ray instead: its pool is the same dominance rule,
//                           enumerated on the CPU from the dump's readback
//                           (BakedGI::GetRayCandidates). A named ray is drawn as
//                           it is — WHITE segment = it is in that pool, MAGENTA
//                           = it is not (a sun move since the dump);
//                        3. a line from the probe to that ray's hit point, and
//                           the 8 trilinear cage probes the bake's feedback
//                           gather actually sampled there (BakedGI_Bake.hlsl:58-
//                           118 replayed verbatim), each coloured by its own
//                           share of the gather: GREEN = the largest share,
//                           AMBER = the middle, RED = the smallest or a member
//                           the backface test culled (share 0);
//                        4. the trilinear CELL of the pushed read point as a
//                           thin wireframe, because the read point is
//                           worldPos + n*0.1*spacing (:65) and that push can
//                           cross the cell plane and reweight the cell. It no
//                           longer CULLS a face: the backface plane is now
//                           measured from the hit, not from the pushed point.
//                      X-RAY: this view skips the pixel shader's depth test. A
//                      wiring trace is only legible if the cage is visible, and
//                      the cage probes sit around, behind and inside the very
//                      surface the ray hit. Every other view keeps its test.
//                      The ramp is normalised by the LARGEST share among the 8,
//                      not by 1.0: a collapsed cage (wsum 0.039, one member
//                      carrying 0.93) must still read green-to-red, and an
//                      absolute ramp would paint all eight dark red and say
//                      nothing. The absolute level is a readout number, not a
//                      colour — see BakedGI::LogRayDebugReadout.
//
// Post-composite: drawn after the TAA resolve/tonemap directly onto the
// backbuffer at output resolution — display-referred colors, no taaEnabled
// branching, temporal history unpolluted. Occlusion is a manual reverse-Z
// compare against the GBuffer depth SRV. Geometry is generated entirely from
// SV_VertexID/SV_InstanceID (no vertex buffers); SV_VertexID restarts per
// instance, so the cell index comes from SV_InstanceID.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// BakedGI.hlsli references FrameCB — include after the declaration above.
#include "BakedGI.hlsli"

#define BAKED_GI_DEBUG_PLACEMENT  1u
#define BAKED_GI_DEBUG_LIT        2u
#define BAKED_GI_DEBUG_SKYVIS     3u
#define BAKED_GI_DEBUG_VISRAYS    4u
#define BAKED_GI_DEBUG_LIT_SKY    5u
#define BAKED_GI_DEBUG_LIT_SUN    6u
#define BAKED_GI_DEBUG_RAYFAN     7u
#define BAKED_GI_DEBUG_EFB        8u
#define BAKED_GI_DEBUG_CAGE       9u

// "No response ball" sentinel for bakedGIEfbIterSRVIndex /
// bakedGIEfbBaseSRVIndex. One value with two spellings: this is the shader's,
// and BakedGI::kEfbBallNoSRV (BakedGI.h) is the C++ one. 0 is a legal
// descriptor index, so it cannot mean "none" — the mode-8 branch early-outs on
// the sentinel before it declares either SRV, so a missing capture can never
// index the descriptor heap.
#define BAKED_GI_EFB_NO_SRV       0xFFFFFFFFu

// Gather rays for each direction. Local to this file on purpose: BakedGI.hlsli
// and BakedGI_Bake.hlsl stay untouched. The value must match GATHER_RAYS in
// BakedGI_Bake.hlsl and BakedGI::kGatherRays.
#define BAKED_GI_RAYFAN_RAYS 64u

// Float4s for each ray record — the stride this view walks. Must match
// RAY_RECORD_F4 (BakedGI_RayDebug.hlsl, the writer) and BakedGI::kRayRecordF4
// (BakedGI.h, the C++ readout). rec0..rec3 are the shading record this view
// colours; rec4/rec5 carry the hit's normal, instance index, world position and
// primitive index and are not drawn, only read back.
#define BAKED_GI_RAYFAN_REC_F4 6u

// UV-sphere tessellation (same layout as IrCache_DebugSpheres: 8x8 quads).
#define DBG_N_STACKS    8
#define DBG_N_SLICES    8
#define DBG_SPHERE_VERTS (DBG_N_STACKS * DBG_N_SLICES * 6)

struct VSOut
{
    float4 pos   : SV_POSITION;
    float4 color : COLOR;
};

// 12 triangles of a [-1,1]^3 cube.
static const float3 kCubeTris[36] =
{
    // +X
    float3( 1, -1, -1), float3( 1,  1, -1), float3( 1,  1,  1),
    float3( 1, -1, -1), float3( 1,  1,  1), float3( 1, -1,  1),
    // -X
    float3(-1, -1,  1), float3(-1,  1,  1), float3(-1,  1, -1),
    float3(-1, -1,  1), float3(-1,  1, -1), float3(-1, -1, -1),
    // +Y
    float3(-1,  1, -1), float3(-1,  1,  1), float3( 1,  1,  1),
    float3(-1,  1, -1), float3( 1,  1,  1), float3( 1,  1, -1),
    // -Y
    float3(-1, -1,  1), float3(-1, -1, -1), float3( 1, -1, -1),
    float3(-1, -1,  1), float3( 1, -1, -1), float3( 1, -1,  1),
    // +Z
    float3(-1, -1,  1), float3( 1, -1,  1), float3( 1,  1,  1),
    float3(-1, -1,  1), float3( 1,  1,  1), float3(-1,  1,  1),
    // -Z
    float3( 1, -1, -1), float3(-1, -1, -1), float3(-1,  1, -1),
    float3( 1, -1, -1), float3(-1,  1, -1), float3( 1,  1, -1),
};

// Clip-space sentinel: z < 0 with w = 1 fails the reverse-Z clip test, so the
// whole triangle is discarded (used for per-instance culls).
static const float4 kDegenerate = float4(0.0f, 0.0f, -1.0f, 1.0f);

// Sphere quad corner decode (IrCache_DebugSpheres layout):
// quad 0-63, 6 verts each: triangles (0,1,2) and (1,3,2).
static const uint kCornerTable[6] = { 0u, 1u, 2u, 1u, 3u, 2u };

float3 BakedGIProbeCellPos(uint probe, uint3 dims, float3 gridMin, float spacing)
{
    uint3 cell = uint3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    return gridMin + float3(cell) * spacing;
}

// One vertex of a thin oriented box: local x/y span [-halfWidth, +halfWidth]
// across the box, local z spans [0, len] along w, and the box starts at origin
// (the probe centre). c is a corner of the unit cube (kCubeTris).
//
// ONE copy of the orientation math, shared by both ray views (mode 4 visibility
// rays, mode 7 ray fan): they differ only in which direction they pass and in
// which len/halfWidth. A second copy is how the two would stop pointing the
// same way.
float3 OrientedBoxVertex(float3 origin, float3 w, float len, float halfWidth, float3 c)
{
    const float3 up        = (abs(w.y) < 0.99f) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 tangent   = normalize(cross(up, w));
    const float3 bitangent = cross(w, tangent);
    const float  along     = (c.z * 0.5f + 0.5f) * len; // map [-1,1] -> [0, len]
    return origin + tangent * (c.x * halfWidth) + bitangent * (c.y * halfWidth) + w * along;
}

// ---------------------------------------------------------------------------
// Ray Fan colours (mode 7). Local-prefixed, so no name can collide with the
// bake, the update pass, or the visibility pass. The constants are the ones
// from the mockup the user approved.
// ---------------------------------------------------------------------------

// Radiance-L display gain. The mockup value. L is diffAlbedo/PI * (NdotD*Vhit +
// Efb), so the direct term alone is bounded by 1/PI = 0.318, and a dark grazing
// hit lands near 1e-3 — that wide range is what this gain lifts. Retune HERE if
// the channel clips: the whole mapping lives in this one place.
#define BAKED_GI_RAYFAN_L_GAIN 900.0f
// Efb log ramp top. Efb blends response-table entries, so it shares the table's
// units (order 0.1 to 1) and not the lit buffer's scale.
#define BAKED_GI_RAYFAN_EFB_MAX 1.0f
// Dim factor on an INVALID probe. An invalid probe's stored table is all zero,
// so the table-derived channels (4 L, 5 Efb, 7 regime) describe nothing. Dim
// exactly those three and leave the ray geometry channels at full strength.
#define BAKED_GI_RAYFAN_INVALID_DIM 0.25f
// "No ray-debug buffer" sentinel for bakedGIRayDebugSRVIndex. 0 is a legal
// descriptor index, so the app publishes 0xFFFFFFFF instead. The mode-7 branch
// early-outs on it before it declares the SRV.
#define BAKED_GI_RAYFAN_NO_SRV 0xFFFFFFFFu
// Half-width of a ray box, in probe spacings. 0.01 was chosen when the fan drew
// all 1024 rays and the boxes fused at mode 4's 0.015. The fan now draws 64, so
// this value is conservative rather than necessary — mode 4's 0.015 is safe to
// use here if the boxes read too thin.
#define BAKED_GI_RAYFAN_BOX_HALF 0.01f
// Ray Fan colour channel that swaps the GEOMETRY: it draws the 16 fibonacci
// delta directions of the fan's probe instead of the 64 gather rays, and reads
// no record. Must match BakedGI::kRayFanChannelDeltaDirs (BakedGI.h) and the
// "DeltaDirections" entry of the Ray Fan Channel combo in Application.cpp.
#define BAKED_GI_RAYFAN_CH_DELTADIRS 8u

// Record component 3 of a ray: 0 = sky miss, 1 = backface reject, 2 = shaded.
float3 RayFanStatusColor(float status)
{
    if (status > 1.5f)
        return float3(0.937f, 0.624f, 0.153f); // #ef9f27 shaded
    if (status > 0.5f)
        return float3(0.231f, 0.510f, 0.965f); // #3b82f6 backface reject
    return float3(0.200f, 0.200f, 0.200f);     // #333333 sky miss
}

// Log-normalised grayscale. A value at or above vMax is white, 0 is black. The
// log keeps a short hit and a long hit apart on the same ramp.
float RayFanLogGray(float v, float vMax)
{
    return saturate(log10(1.0f + max(v, 0.0f)) / max(log10(1.0f + vMax), 1e-6f));
}

// Regime: which term feeds the ray's radiance. Green = the direct delta at the
// hit (NdotD * Vhit) dominates, neutral = an even split, red = the series
// feedback (Efb) dominates. The leak channel reads at a glance.
float3 RayFanRegimeColor(float direct, float efb)
{
    const float denom = direct + efb;
    if (denom < 1e-6f)
        return float3(0.086f, 0.086f, 0.086f); // neutral: the ray contributes nothing
    const float  t       = saturate(efb / denom);
    const float3 green   = float3(0.180f, 0.490f, 0.196f); // #2e7d32
    const float3 neutral = float3(0.086f, 0.086f, 0.086f); // #161616
    const float3 red     = float3(0.898f, 0.224f, 0.208f); // #e53935
    return (t < 0.5f) ? lerp(green, neutral, t * 2.0f)
                      : lerp(neutral, red, (t - 0.5f) * 2.0f);
}

// ---------------------------------------------------------------------------
// DeltaDirections channel (mode 7, BAKED_GI_RAYFAN_CH_DELTADIRS).
//
// The largest of the 16 published weights — the divisor of the grayscale ramp.
// The MAXIMUM, not the sum, on purpose: the published weights are normalised to
// sum to 1, so the winner holds only ~0.5 whenever the sun sits between two
// directions, and a share-based ramp would then leave NO direction white — in
// exactly the configuration this channel exists to report. With the maximum as
// the divisor the dominant direction reads 1.0 whatever the sun does.
//
// A TIE at the maximum paints every tied direction white. That is the weights
// being reported, not an arbitration: BakedGI::SunFoldWeights still returns the
// lowest index, and the UI text names that one.
// ---------------------------------------------------------------------------
float BakedGISunWeightMax()
{
    float m = 0.0f;
    [unroll]
    for (uint j = 0; j < BAKED_GI_DIRECTIONS; ++j)
        m = max(m, FrameCB.bakedGISunWeight[j].x); // .x: the field is a float4 array
    return m;
}

// ---------------------------------------------------------------------------
// Cage Trace (mode 9). Instance layout, 36 verts (kCubeTris) each:
//    0        the probe -> hit-point segment, the winning gather ray
//    1..8     the hit point -> cage probe k segment, one for each corner
//    9..16    a marker cube at cage probe k, the same ramp colour
//    17       a marker cube at the probe
//    18       a marker cube at the hit point
//    19..30   the 12 edges of the trilinear cell of the pushed read point
// The count is published to the renderer through the ONE constant below, so the
// draw call cannot disagree with the geometry.
// ---------------------------------------------------------------------------
#define BAKED_GI_CAGE_SEG_PROBE    0u
#define BAKED_GI_CAGE_SEG_BASE     1u   // 1 .. 8
#define BAKED_GI_CAGE_MARK_BASE    9u   // 9 .. 16
#define BAKED_GI_CAGE_MARK_PROBE  17u
#define BAKED_GI_CAGE_MARK_HIT    18u
#define BAKED_GI_CAGE_EDGE_BASE   19u   // 19 .. 30
#define BAKED_GI_CAGE_INSTANCES   31u

// Half-widths and marker half-sizes, in probe spacings. Thin next to the 1.0 m
// cell so a 40-segment fan stays readable.
#define BAKED_GI_CAGE_SEG_HALF   0.010f
#define BAKED_GI_CAGE_EDGE_HALF  0.004f
#define BAKED_GI_CAGE_MARK_SIZE  0.055f

// The 12 edges of a unit cube, as corner-index pairs. Corner c decodes to
// (c&1, (c>>1)&1, (c>>2)&1) — the same packing the cage's `off` uses.
static const uint2 kCubeEdges[12] =
{
    uint2(0u, 1u), uint2(2u, 3u), uint2(4u, 5u), uint2(6u, 7u), // along x
    uint2(0u, 2u), uint2(1u, 3u), uint2(4u, 6u), uint2(5u, 7u), // along y
    uint2(0u, 4u), uint2(1u, 5u), uint2(2u, 6u), uint2(3u, 7u), // along z
};

float3 CageCorner(uint c)
{
    return float3((float)(c & 1u), (float)((c >> 1u) & 1u), (float)((c >> 2u) & 1u));
}

// Green = the largest share of the gather, red = the smallest (or a culled
// member, whose share is exactly 0). Through amber so the middle reads.
float3 CageShareColor(float t)
{
    const float3 green = float3(0.180f, 0.690f, 0.196f);
    const float3 amber = float3(0.902f, 0.663f, 0.157f);
    const float3 red   = float3(0.898f, 0.184f, 0.161f);
    t = saturate(t);
    return (t < 0.5f) ? lerp(red, amber, t * 2.0f)
                      : lerp(amber, green, (t - 0.5f) * 2.0f);
}

// The probe -> hit segment of a HAND-PICKED ray whose hit is NOT in the picker's
// pool: "direct < Efb" fails at that hit, so the cage below it is not the borrow
// the view exists to show. That happens when the sun moved after the dump and
// the chosen ray index now belongs to a different direction's fan — the index
// stays legal, the meaning does not, and a colour is the only way to say so
// without a second pick. MAGENTA: no other element of this view uses it (green /
// amber / red are the cage shares, white / cyan the two endpoints, grey the
// cell wireframe).
static const float3 kCagePickOutsidePool = float3(0.870f, 0.180f, 0.870f);

// The winning gather ray of one direction: the shaded ray with the largest
// carried feedback among the rays the feedback DOMINATES.
//
// SCORE = luminance(diffAlbedo * Efb). The bake's band-0 coefficient is a
// constant times the ray's radiance (all 64 rays share the solid angle
// 4*PI/64), so ranking by carried radiance IS ranking by contribution to the
// stored response — and the albedo must stay inside the product: a ray onto a
// dark or blue surface carries a large Efb and contributes almost nothing.
//
// The DOMINANCE test (Efb > NdotD*Vhit at the hit) is the mean "Direct vs Efb"
// regime the Ray Fan's channel 7 paints red, and it is the same rule the app's
// CPU pool applies (BakedGI::LogRayDebugReadout step 2c), so the combo lists
// exactly the rays this argmax ranks. The strict "direct == 0" set is its
// subset (Efb > 0 implies Efb > direct there), so this rule covers both readings.
//
// forcedRay is FrameConstants::bakedGIRayDebugRay: a ray index the UI named, or
// >= BAKED_GI_RAYFAN_RAYS for "no override". A FORCED ray is taken as it is —
// the dominance test is NOT repeated. The user picked it out of a list that had
// already passed the test, and repeating it could only disagree on the last bit
// of a luminance and silently draw nothing, which reads as a bug. Whether the
// forced ray is in the pool is reported by `isBorrow` at the call site instead.
struct CageRay
{
    bool   found;
    bool   isBorrow;    // Efb > NdotD*Vhit at the hit: the ray is inside the picker's pool
    uint   ray;
    float3 hitPos;
    float3 hitNormal;
    float  albedoLum;
    float  efbLum;      // = the score
    float  directLum;
};

CageRay FindCageRay(StructuredBuffer<float4> dbg, uint dir, uint forcedRay)
{
    const float3 kLum = float3(0.2126f, 0.7152f, 0.0722f);

    CageRay best;
    best.found = false;
    best.isBorrow = false;
    best.ray = 0u;
    best.hitPos = float3(0.0f, 0.0f, 0.0f);
    best.hitNormal = float3(0.0f, 0.0f, 1.0f);
    best.albedoLum = 0.0f;
    best.efbLum = -1.0f;
    best.directLum = 0.0f;

    // One ray when the UI named one, all 64 when it did not. The forced range is
    // a half-open [r, r+1). A FORCED index out of range is rejected at the call
    // site (it early-outs), so this cannot read past the direction's records.
    const bool forced = (forcedRay < BAKED_GI_RAYFAN_RAYS);
    const uint first  = forced ? forcedRay : 0u;
    const uint last   = forced ? (forcedRay + 1u) : BAKED_GI_RAYFAN_RAYS;

    [loop]
    for (uint r = first; r < last; ++r)
    {
        const uint b = (dir * BAKED_GI_RAYFAN_RAYS + r) * BAKED_GI_RAYFAN_REC_F4;
        const float4 rec1 = dbg[b + 1u]; // (diffAlbedo.rgb, NdotD)
        const float4 rec2 = dbg[b + 2u]; // (L.rgb, Vhit)
        const float4 rec3 = dbg[b + 3u]; // (Efb.rgb, status)
        const float4 rec4 = dbg[b + 4u]; // (hit normal.xyz, instance)
        const float4 rec5 = dbg[b + 5u]; // (hit world pos.xyz, primitive)

        // Status: 0 = sky miss, 1 = backface reject, 2 = shaded. Only a shaded
        // ray has an L and an Efb to compare.
        if (abs(rec3.w - 2.0f) > 0.5f)
            continue;

        const float directLum = dot(rec1.rgb * (rec1.w * rec2.w), kLum);
        const float efbLum    = dot(rec1.rgb * rec3.rgb, kLum);
        const bool  borrow    = (efbLum > directLum);

        // The pool filter, auto path only. See this function's header.
        if (!forced && !borrow)
            continue;

        if (efbLum > best.efbLum)
        {
            best.found      = true;
            best.isBorrow   = borrow;
            best.ray        = r;
            best.hitPos     = rec5.xyz;
            best.hitNormal  = rec4.xyz;
            best.albedoLum  = dot(rec1.rgb, kLum);
            best.efbLum     = efbLum;
            best.directLum  = directLum;
        }
    }
    return best;
}

// The 8-corner trilinear cage the bake's feedback gather samples at one hit
// point. A VERBATIM replay of BakedGI_Bake.hlsl:58-118 with the weights KEPT
// instead of immediately folded into the SH sum — the bake discards them, and
// the weights are the whole diagnostic.
struct CageSample
{
    float3 pos[8];
    uint   flat[8];
    float  full[8];   // wFull, the unnormalised gather weight
    float  share[8];  // wFull / wsum
    float  wsum;
    float3 xq;        // the pushed read point (worldPos + n * 0.1 * spacing)
    float3 cellMin;   // the trilinear cell the pushed point addresses
};

CageSample BuildCage(StructuredBuffer<float2> vis, float3 worldPos, float3 n,
                     uint3 dims, float3 gridMin, float spacing)
{
    CageSample c;
    const float3 xq = worldPos + n * (0.1f * spacing);

    const float3 t = clamp((xq - gridMin) / spacing, 0.0f.xxx, float3(dims) - 1.0f);
    const int3   base = int3(min(floor(t), max(float3(dims) - 2.0f, 0.0f.xxx)));
    const float3 f = t - float3(base);

    c.xq      = xq;
    c.cellMin = gridMin + float3(base) * spacing;

    float wsum = 0.0f;
    [unroll]
    for (uint k = 0u; k < 8u; ++k)
    {
        const int3   off  = int3(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        const float3 w3   = lerp(1.0f.xxx - f, f, float3(off));
        const float  w    = w3.x * w3.y * w3.z;
        const uint   flat = BakedGIFlatten(uint3(base + off), dims);
        const float3 p    = gridMin + float3(base + off) * spacing;
        const float3 toProbe = p - xq;
        const float  dist    = length(toProbe);

        float wFull;
        if (dist < 1e-4f)
        {
            wFull = w; // the read point sits on the probe: trivially visible
        }
        else
        {
            const float3 dirv = toProbe / dist;
            // Backface from the HIT, not from the pushed point — the same
            // origin the bake and the render fetch now use
            // (BakedGI_Bake.hlsl, BakedGI.hlsli). A debug view that kept the
            // old origin would draw a cull the bake no longer performs.
            const float3 toHit = p - worldPos;
            const float  dHit  = length(toHit);
            const float  back  = (dHit < 1e-4f) ? 1.0f : saturate(dot(n, toHit / dHit));
            // Culled BEFORE the moment fetch, exactly as the bake does. A
            // member behind the hit surface's tangent plane contributes 0 and
            // is drawn red.
            wFull = (back <= 0.0f)
                  ? 0.0f
                  : w * back * BakedGIChebyshev(BakedGISampleMoments(vis, flat, -dirv), dist);
        }

        c.pos[k]  = p;
        c.flat[k] = flat;
        c.full[k] = wFull;
        wsum += wFull;
    }

    c.wsum = wsum;
    const float inv = (wsum > 1e-6f) ? (1.0f / wsum) : 0.0f;
    [unroll]
    for (uint k2 = 0u; k2 < 8u; ++k2)
        c.share[k2] = c.full[k2] * inv;
    return c;
}

VSOut VSMain(uint vid : SV_VertexID, uint instance : SV_InstanceID)
{
    VSOut o;
    o.pos   = kDegenerate;
    o.color = float4(1.0f, 1.0f, 1.0f, 1.0f);

    const uint   mode    = FrameCB.bakedGIDebugView;
    const uint3  dims    = uint3(FrameCB.bakedGIDimX, FrameCB.bakedGIDimY, FrameCB.bakedGIDimZ);
    const float3 gridMin = float3(FrameCB.bakedGIGridMinX, FrameCB.bakedGIGridMinY, FrameCB.bakedGIGridMinZ);
    const float  spacing = FrameCB.bakedGISpacing;
    const uint   probeCount = dims.x * dims.y * dims.z;

    StructuredBuffer<float4> response = ResourceDescriptorHeap[FrameCB.bakedGIResponseSRVIndex];
    StructuredBuffer<uint>   meta     = ResourceDescriptorHeap[FrameCB.bakedGIProbeMetaSRVIndex];

    // --- Mode 7: the sun-dominant direction's 64 gather rays of ONE probe ---
    // The record layout is the compute pass's: six float4 for each ray, all 64
    // rays of direction 0 first, then direction 1, and so on. The box geometry
    // is the mode-4 box with the RECORDED gather direction in place of the
    // fibonacci direction — that swap is the whole point of the view.
    // The direction is the sun fold's argmax (FrameConstants::bakedGIRayDebugDir),
    // so the fan shows the rays that carry sun energy, and 64 boxes stay legible.
    if (mode == BAKED_GI_DEBUG_RAYFAN)
    {
        // --- DeltaDirections: the 16 fold directions of this probe ---
        // Placed BEFORE the record path deliberately. This channel reads no
        // record, so it must not depend on the ray-debug buffer existing and
        // must not index the descriptor heap with the no-buffer sentinel. The
        // fan's own probe and the fan's own cull apply — one probe, and no
        // camera-distance window — so this channel always sits at the same cell
        // as the 64-ray channels.
        if (FrameCB.bakedGIRayDebugChannel == BAKED_GI_RAYFAN_CH_DELTADIRS)
        {
            // One instance is one fibonacci direction. The guard is not a
            // formality: the divisor array has exactly BAKED_GI_DIRECTIONS
            // entries, so an unguarded instance index reads past it.
            if (instance >= BAKED_GI_DIRECTIONS)
                return o;
            const uint probe = FrameCB.bakedGIRayDebugProbe;
            if (probe >= probeCount)
                return o;

            // Grayscale, the NdotD channel's mapping: the value written
            // straight, no gain and no log ramp, so WHITE is 1.0 = the
            // most-weighted direction. Black is a zero weight — which includes
            // the no-sun case, where every weight is 0 and all 16 boxes are
            // black rather than one of them being falsely painted white.
            const float wMax = BakedGISunWeightMax();
            const float t = (wMax > 1e-6f)
                          ? saturate(FrameCB.bakedGISunWeight[instance].x / wMax)
                          : 0.0f;
            o.color = float4(t.xxx, 1.0f);

            const float3 probePos = BakedGIProbeCellPos(probe, dims, gridMin, spacing);
            const float3 w = BakedGIDeltaDirection(instance);
            const float3 world = OrientedBoxVertex(probePos, w, spacing * 0.35f,
                                                   spacing * 0.015f, kCubeTris[vid % 36u]);
            o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
            return o;
        }

        // The app publishes BAKED_GI_RAYFAN_NO_SRV until a dump creates the
        // buffer. Early-out BEFORE the SRV declaration, so a missing buffer can
        // never index the descriptor heap.
        if (FrameCB.bakedGIRayDebugSRVIndex == BAKED_GI_RAYFAN_NO_SRV)
            return o;
        // One instance is one ray: dir selects the direction's block in the
        // record buffer, r the ray inside it. Guard both before the index math.
        const uint dir = FrameCB.bakedGIRayDebugDir;
        const uint r   = instance;
        if (dir >= BAKED_GI_DIRECTIONS)
            return o;
        if (r >= BAKED_GI_RAYFAN_RAYS)
            return o;
        const uint fanProbe = FrameCB.bakedGIRayDebugProbe;
        if (fanProbe >= probeCount)
            return o;

        StructuredBuffer<float4> dbg = ResourceDescriptorHeap[FrameCB.bakedGIRayDebugSRVIndex];
        const uint recBase = (dir * BAKED_GI_RAYFAN_RAYS + r) * BAKED_GI_RAYFAN_REC_F4;
        const float4 rec0 = dbg[recBase + 0u]; // (gather direction.xyz, hit distance; 1e4 = miss)
        const float4 rec1 = dbg[recBase + 1u]; // (diffuse albedo.rgb, NdotD)
        const float4 rec2 = dbg[recBase + 2u]; // (L.rgb, Vhit)
        const float4 rec3 = dbg[recBase + 3u]; // (Efb.rgb, status)

        const float t      = rec0.w;
        const float status = rec3.w;
        const bool  miss   = (status < 0.5f);

        // MISSES DRAW AT THE CAP. Dropping them would be tidier, but then the
        // status channel could not show which way the probe sees open sky. The
        // cost is a full-length box in the albedo channel, and there it is black
        // (a miss carries no albedo) — that is the correct reading, not a bug.
        const float cap = max(spacing * FrameCB.bakedGIRayDebugLength, 1e-4f);
        const float len = saturate(t / cap) * cap; // t = 1e4 for a miss -> the cap

        // Colour channel (FrameConstants::bakedGIRayDebugChannel). Channel 1
        // (albedo) is the default and the fallback for an unknown value.
        const uint ch = FrameCB.bakedGIRayDebugChannel;
        float3 col;
        if (ch == 0u)
            col = RayFanStatusColor(status);
        else if (ch == 2u)
            col = rec1.w.xxx; // NdotD, already in [0, 1]
        else if (ch == 3u)
            col = (rec2.w > 0.5f) ? float3(0.909f, 0.909f, 0.909f)  // #e8e8e8 lit
                                  : float3(0.086f, 0.086f, 0.086f); // #161616 shadowed
        else if (ch == 4u)
            col = saturate(rec2.rgb * BAKED_GI_RAYFAN_L_GAIN);
        else if (ch == 5u)
            col = RayFanLogGray(dot(rec3.rgb, float3(0.2126f, 0.7152f, 0.0722f)), BAKED_GI_RAYFAN_EFB_MAX).xxx;
        else if (ch == 6u)
            col = miss ? float3(0.086f, 0.086f, 0.086f) : RayFanLogGray(t, cap).xxx;
        else if (ch == 7u)
            col = RayFanRegimeColor(rec1.w * rec2.w, dot(rec3.rgb, float3(0.2126f, 0.7152f, 0.0722f)));
        else
            col = rec1.rgb; // 1 = albedo

        // An invalid probe has an all-zero stored table (BakedGI_Bake.hlsl:147-155),
        // so the three channels read from that table are degenerate. Dim those
        // three only: the ray geometry and the record channels stay readable.
        if (FrameCB.bakedGIRayDebugValid == 0u && (ch == 4u || ch == 5u || ch == 7u))
            col *= BAKED_GI_RAYFAN_INVALID_DIM;

        o.color = float4(col, 1.0f);

        const float3 fanPos    = BakedGIProbeCellPos(fanProbe, dims, gridMin, spacing);
        const float3 w         = normalize(rec0.xyz);
        const float3 world = OrientedBoxVertex(fanPos, w, len, spacing * BAKED_GI_RAYFAN_BOX_HALF,
                                               kCubeTris[vid % 36u]);

        o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
        return o;
    }

    // --- Mode 9: Cage Trace — the Efb borrow as geometry ---
    // Everything above is derived from the probe index alone: the direction is
    // the sun fold's argmax, the ray is that direction's largest carried
    // feedback among the feedback-dominated rays, and the cage is the bake's
    // own gather replay at that ray's hit point. Nothing here is a second
    // computation of anything the bake does.
    //
    // ONE exception, and it is deliberate: bakedGIRayDebugRay lets the UI name
    // the ray instead of ranking for it, which is the only way to reach a ray
    // that is not the brightest one. The list it names from is the CPU pool of
    // the SAME rule, read from the dump's own readback.
    if (mode == BAKED_GI_DEBUG_CAGE)
    {
        // Early-out BEFORE declaring the SRV, the same way mode 7 does: 0 is a
        // legal descriptor index, so the no-buffer sentinel cannot index it.
        if (FrameCB.bakedGIRayDebugSRVIndex == BAKED_GI_RAYFAN_NO_SRV)
            return o;
        if (instance >= BAKED_GI_CAGE_INSTANCES)
            return o;

        const uint dir       = FrameCB.bakedGIRayDebugDir;
        const uint cageProbe = FrameCB.bakedGIRayDebugProbe;
        // >= BAKED_GI_RAYFAN_RAYS is the "no override" sentinel. Tested here, not
        // inside FindCageRay, so an out-of-range value cannot select a record
        // beyond this direction's 64 — it falls back to the argmax, which is the
        // behaviour a stale or hand-typed index should have.
        const uint forcedRay = (FrameCB.bakedGIRayDebugRay < BAKED_GI_RAYFAN_RAYS)
                             ? FrameCB.bakedGIRayDebugRay
                             : BAKED_GI_RAYFAN_RAYS;
        if (dir >= BAKED_GI_DIRECTIONS || cageProbe >= probeCount)
            return o;

        StructuredBuffer<float4> dbg = ResourceDescriptorHeap[FrameCB.bakedGIRayDebugSRVIndex];

        const CageRay win = FindCageRay(dbg, dir, forcedRay);
        if (!win.found)
            return o; // nothing in this direction is feedback-dominated

        // The depth moments are read from the SAME descriptor the runtime fetch
        // reads (BakedGI.hlsli:273) and no transition is added: the bake writes
        // them with a UAV barrier and never changes their state, so the fetch and
        // this pass see one state. A bake must exist for the index to be live;
        // the renderer gates on BakedGI::IsValid for exactly that reason.
        StructuredBuffer<float2> vis = ResourceDescriptorHeap[FrameCB.bakedGIProbeVisSRVIndex];
        const CageSample cs = BuildCage(vis, win.hitPos, win.hitNormal, dims, gridMin, spacing);

        // Ramp divisor: the LARGEST share among the 8, not 1.0. A collapsed cage
        // must still read green-to-red; an absolute ramp would paint every member
        // dark red and report nothing.
        float maxShare = 0.0f;
        [unroll]
        for (uint k = 0u; k < 8u; ++k)
            maxShare = max(maxShare, cs.share[k]);
        const float invMax = (maxShare > 1e-9f) ? (1.0f / maxShare) : 0.0f;

        const float3 probePos = BakedGIProbeCellPos(cageProbe, dims, gridMin, spacing);
        const float3 corner   = kCubeTris[vid % 36u];

        float3 world;
        float3 col;

        if (instance == BAKED_GI_CAGE_SEG_PROBE)
        {
            // The winning gather ray itself, drawn from the probe to its hit.
            // WHITE = the feedback dominates at that hit, so this is the borrow
            // the view exists to show. MAGENTA = a hand-picked ray whose hit is
            // NOT dominant (see kCagePickOutsidePool): the cage below is a real
            // gather, but it is not an Efb borrow. Both are drawn, because a
            // silent blank on an explicitly chosen ray reads as a failure.
            const float3 d   = win.hitPos - probePos;
            const float  len = length(d);
            col   = win.isBorrow ? float3(0.930f, 0.930f, 0.960f) : kCagePickOutsidePool;
            world = (len < 1e-5f)
                  ? probePos
                  : OrientedBoxVertex(probePos, d / len, len, spacing * BAKED_GI_CAGE_SEG_HALF, corner);
        }
        else if (instance >= BAKED_GI_CAGE_SEG_BASE && instance < BAKED_GI_CAGE_SEG_BASE + 8u)
        {
            const uint   k   = instance - BAKED_GI_CAGE_SEG_BASE;
            const float3 d   = cs.pos[k] - win.hitPos;
            const float  len = length(d);
            col   = CageShareColor(cs.share[k] * invMax);
            world = (len < 1e-5f)
                  ? win.hitPos
                  : OrientedBoxVertex(win.hitPos, d / len, len, spacing * BAKED_GI_CAGE_SEG_HALF, corner);
        }
        else if (instance >= BAKED_GI_CAGE_MARK_BASE && instance < BAKED_GI_CAGE_MARK_BASE + 8u)
        {
            // A marker at each cage probe, the same ramp colour as its segment,
            // so a member whose segment is edge-on is still visible.
            const uint k = instance - BAKED_GI_CAGE_MARK_BASE;
            col   = CageShareColor(cs.share[k] * invMax);
            world = cs.pos[k] + corner * (spacing * BAKED_GI_CAGE_MARK_SIZE);
        }
        else if (instance == BAKED_GI_CAGE_MARK_PROBE)
        {
            col   = float3(0.950f, 0.950f, 0.950f);
            world = probePos + corner * (spacing * BAKED_GI_CAGE_MARK_SIZE);
        }
        else if (instance == BAKED_GI_CAGE_MARK_HIT)
        {
            col   = float3(0.250f, 0.850f, 0.950f);
            world = win.hitPos + corner * (spacing * BAKED_GI_CAGE_MARK_SIZE);
        }
        else
        {
            // The 12 edges of the trilinear cell of the PUSHED read point. The
            // gap between this box and the hit point is the 0.1*spacing push,
            // and crossing the box's far face is what culls a whole cage face.
            const uint2 e = kCubeEdges[min(instance - BAKED_GI_CAGE_EDGE_BASE, 11u)];
            const float3 a = cs.cellMin + CageCorner(e.x) * spacing;
            const float3 b = cs.cellMin + CageCorner(e.y) * spacing;
            const float3 d = b - a;
            const float  len = length(d);
            col   = float3(0.420f, 0.440f, 0.520f);
            world = (len < 1e-5f)
                  ? a
                  : OrientedBoxVertex(a, d / len, len, spacing * BAKED_GI_CAGE_EDGE_HALF, corner);
        }

        o.color = float4(col, 1.0f);
        o.pos   = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
        return o;
    }

    // --- Mode 4: per-(probe, direction) visibility rays ---
    if (mode == BAKED_GI_DEBUG_VISRAYS)
    {
        const uint probe = instance / BAKED_GI_DIRECTIONS;
        const uint dir   = instance % BAKED_GI_DIRECTIONS;
        if (probe >= probeCount)
            return o;

        const float3 probePos = BakedGIProbeCellPos(probe, dims, gridMin, spacing);
        if (!(meta[probe] & 1u))
            return o; // invalid probes have zeroed transport — nothing to show
        if (distance(probePos, FrameCB.cameraPosition.xyz) > spacing * 8.0f)
            return o; // near-camera window keeps the ray count readable

        const float3 w = BakedGIDeltaDirection(dir);
        const uint slot = (probe * BAKED_GI_DIRECTIONS + dir) * BAKED_GI_RESPONSE_F4;
        const float V   = response[slot + 9].x;
        o.color = (V > 0.5f) ? float4(0.1f, 0.9f, 0.2f, 1.0f)      // open sky
                             : float4(0.95f, 0.15f, 0.15f, 1.0f);   // blocked

        // Thin oriented box from the probe center along w (local z -> w).
        const float3 world = OrientedBoxVertex(probePos, w, spacing * 0.35f, spacing * 0.015f,
                                               kCubeTris[vid % 36u]);

        o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
        return o;
    }

    // --- Per-probe views (placement / lit irradiance / sky visibility) ---
    const uint probe = instance;
    if (probe >= probeCount)
        return o;

    const float3 probePos = BakedGIProbeCellPos(probe, dims, gridMin, spacing);
    const bool   valid    = (meta[probe] & 1u) != 0u;

    if (mode == BAKED_GI_DEBUG_LIT || mode == BAKED_GI_DEBUG_LIT_SKY || mode == BAKED_GI_DEBUG_LIT_SUN ||
        mode == BAKED_GI_DEBUG_EFB)
    {
        if (!valid)
            return o;

        // UV-sphere vertex: unit sphere position doubles as the eval normal.
        const uint v         = vid % DBG_SPHERE_VERTS;
        const uint quadIdx   = v / 6u;
        const uint localIdx  = v % 6u;
        const uint stackIdx  = quadIdx / DBG_N_SLICES;
        const uint sliceIdx  = quadIdx % DBG_N_SLICES;
        const uint corner    = kCornerTable[localIdx];
        const uint si = stackIdx + (corner >> 1u);
        const uint li = sliceIdx + (corner & 1u);
        const float sph   = (float)si / (float)DBG_N_STACKS * PI;         // [0, PI]
        const float theta = (float)li / (float)DBG_N_SLICES * 2.0f * PI;  // [0, 2PI]
        const float3 n = float3(sin(sph) * cos(theta), cos(sph), sin(sph) * sin(theta));

        float basis[9];
        EvalSH9Basis(n, basis);

        float3 e    = float3(0.0f, 0.0f, 0.0f);
        float  norm = 1.0f; // both branches assign; the initializers only keep the compiler quiet

        if (mode == BAKED_GI_DEBUG_EFB)
        {
            // Efb contribution. Early-out BEFORE declaring an SRV: the app
            // publishes the sentinel until a bake has captured the terms, and
            // 0xFFFFFFFF is not a descriptor this pass may read. Each mode guards
            // the TWO sources it actually reads, not all four.
            const bool oneDir = (FrameCB.bakedGIEfbDir < BAKED_GI_DIRECTIONS);

            if (oneDir)
            {
                if (FrameCB.bakedGIEfbTermSelSRVIndex == BAKED_GI_EFB_NO_SRV ||
                    FrameCB.bakedGIEfbTermBaseSRVIndex == BAKED_GI_EFB_NO_SRV)
                    return o;
            }
            else if (FrameCB.bakedGIEfbIterSRVIndex == BAKED_GI_EFB_NO_SRV ||
                     FrameCB.bakedGIEfbBaseSRVIndex == BAKED_GI_EFB_NO_SRV)
            {
                return o;
            }

            // The selected term's response and term 1's, as SH9 coefficients. The
            // two modes differ ONLY in the source: the 16-direction fold, or one
            // direction of the transport table.
            float3 cSel[9]  = (float3[9])0;
            float3 cBase[9] = (float3[9])0;

            if (oneDir)
            {
                // UN-AVERAGED. The fold is a SUM over 16 directions, so a single
                // red direction is diluted against 15 neutral ones and can be
                // invisible in the ball. Reading that direction's own table entry
                // shows it on its own. wq is the fold's quadrature weight: it sits
                // in the ball AND in that ball's DC normaliser, so it cancels in
                // the displayed ratio — it is kept so this branch is literally
                // BakedGI_IterFold.hlsl's formula with the sum restricted to one
                // direction.
                const float wq = 2.0f * PI / (float)BAKED_GI_DIRECTIONS;
                const uint  slot = (probe * BAKED_GI_DIRECTIONS + FrameCB.bakedGIEfbDir)
                                 * BAKED_GI_RESPONSE_F4;
                StructuredBuffer<float4> tabSel  = ResourceDescriptorHeap[FrameCB.bakedGIEfbTermSelSRVIndex];
                StructuredBuffer<float4> tabBase = ResourceDescriptorHeap[FrameCB.bakedGIEfbTermBaseSRVIndex];
                [unroll]
                for (uint i = 0; i < 9; ++i)
                {
                    cSel[i]  = wq * tabSel[slot + i].rgb;
                    cBase[i] = wq * tabBase[slot + i].rgb;
                }
            }
            else
            {
                StructuredBuffer<float4> ballSel  = ResourceDescriptorHeap[FrameCB.bakedGIEfbIterSRVIndex];
                StructuredBuffer<float4> ballBase = ResourceDescriptorHeap[FrameCB.bakedGIEfbBaseSRVIndex];
                [unroll]
                for (uint i = 0; i < 9; ++i)
                {
                    cSel[i]  = ballSel[probe * 9u + i].rgb;
                    cBase[i] = ballBase[probe * 9u + i].rgb;
                }
            }

            // The difference of the two SH9 fields, evaluated at the sphere
            // normal. Both sides describe the same quantity for the same probe, so
            // the difference is (R^k - R^0) — the series-feedback contribution
            // itself, not an approximation of it.
            float3 diff = float3(0.0f, 0.0f, 0.0f);
            [unroll]
            for (uint lmd = 0; lmd < 9; ++lmd)
                diff += (cSel[lmd] - cBase[lmd]) * basis[lmd];

            // Normalization: the SELECTED term's own DC, the same reading the lit
            // views take from the total buffer (a probe's mean irradiance maps to
            // ~1.0 x gain). Its DC is the reference, so the ball's DC reads as the
            // feedback's share of that term's transport. In the single-direction
            // mode it is that DIRECTION's DC, so both modes keep one reading.
            //
            // The ratio cannot exceed 1: the contribution is non-negative and
            // term[k] = term[0] + contribution componentwise, so DC(diff) <=
            // DC(term). The 1e-9 floor is therefore only a divide guard — a probe
            // whose stored table is empty has both DCs at zero and renders dark,
            // rather than dividing by zero.
            const float dcSel = dot(cSel[0], float3(0.2126f, 0.7152f, 0.0722f));
            e    = diff;
            norm = max(dcSel, 1e-9f);
        }
        else
        {
            // Lit SH9 evaluation at the sphere normal: the ball's shading IS e(p).
            // Mode selects the buffer: total, or the sky/sun source split written
            // by the update pass (total = sky + sun, so the three views add up).
            const uint litSRV = (mode == BAKED_GI_DEBUG_LIT_SKY) ? FrameCB.bakedGIProbeSkySRVIndex
                            : (mode == BAKED_GI_DEBUG_LIT_SUN) ? FrameCB.bakedGIProbeSunSRVIndex
                            : FrameCB.bakedGIProbeSRVIndex;
            StructuredBuffer<float4> lit = ResourceDescriptorHeap[litSRV];

            // Auto-normalize by the TOTAL buffer's DC luminance — the same
            // reference for all three lit views, so sky + sun = total holds
            // visually. (A fixed 200x scale saturated everything: lit DC is ~4e3
            // in FP16Scale'd units, not the ~1e-2 the default assumed.)
            // FLOOR at a meaningfully-lit probe's DC scale: per-probe-only
            // normalization renders near-BLACK probes as half-bright balls in
            // every lit view (split/total -> ~0.5 when both are ~0), which reads
            // as "both splits leaking" over dark/sealed regions whose data is
            // clean (capture-verified: cavity probes sun 532 -> 4 after the
            // visibility-weighted feedback fix). With the floor, dark probes
            // render dark; probes above the floor keep the ratio view unchanged.
            StructuredBuffer<float4> litTotal = ResourceDescriptorHeap[FrameCB.bakedGIProbeSRVIndex];
            const float dcLum = dot(litTotal[probe * 9u + 0].rgb, float3(0.2126f, 0.7152f, 0.0722f));
            norm = max(dcLum, 1e-9f);

            float3 eLit = float3(0.0f, 0.0f, 0.0f);
            [unroll]
            for (uint lm = 0; lm < 9; ++lm)
                eLit += lit[probe * 9u + lm].rgb * basis[lm];
            e = eLit;
        }

        // DC maps to ~1.0 x gain; band structure swings around it.
        o.color = float4(saturate(max(e, 0.0f.xxx) / norm * FrameCB.bakedGIDebugScale), 1.0f);

        const float3 world = probePos + n * (spacing * 0.12f);
        o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
        return o;
    }

    // Cube modes: pick the color source.
    if (mode == BAKED_GI_DEBUG_SKYVIS)
    {
        float vMean = 0.0f;
        [unroll]
        for (uint i = 0; i < BAKED_GI_DIRECTIONS; ++i)
        {
            const uint slot = (probe * BAKED_GI_DIRECTIONS + i) * BAKED_GI_RESPONSE_F4;
            vMean += response[slot + 9].x;
        }
        vMean /= (float)BAKED_GI_DIRECTIONS;
        o.color = valid ? float4(vMean.xxx, 1.0f)                    // sky-open fraction
                        : float4(0.95f, 0.15f, 0.15f, 1.0f);         // invalid
    }
    else // BAKED_GI_DEBUG_PLACEMENT
    {
        // Classification A/B (meta bits, see BakedGI.hlsli): the DDGI
        // criteria vs the old 1cm test, so the criteria swap's flips are
        // visible in place. BLUE is the experiment's headline — probes the
        // old test called valid that DDGI's classification kills.
        const uint  m         = meta[probe];
        const bool  ddgiValid = (m & BAKED_GI_META_VALID)     != 0u;
        const bool  oldValid  = (m & BAKED_GI_META_OLD_VALID) != 0u;
        if (ddgiValid)
            o.color = oldValid ? float4(0.1f, 0.9f, 0.2f, 1.0f)   // valid under both
                               : float4(0.9f, 0.9f, 0.2f, 1.0f);  // newly valid (old test killed it)
        else
            o.color = oldValid ? float4(0.2f, 0.4f, 1.0f, 1.0f)   // NEWLY BAD (old test kept it)
                               : float4(0.95f, 0.15f, 0.15f, 1.0f); // bad under both
    }

    const float3 c = kCubeTris[vid % 36u];
    const float3 world = probePos + c * (spacing * 0.04f);
    o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    // Cage Trace is X-RAY. A wiring trace is only legible if the whole cage is
    // visible, and its probes sit around, behind and inside the very surface the
    // traced ray hit. Decided BEFORE the depth fetch, so this view does not read
    // the depth descriptor at all. Every other view keeps its test.
    if (FrameCB.bakedGIDebugView == BAKED_GI_DEBUG_CAGE)
        return i.color;

    // Manual occlusion against the GBuffer depth (same intent as the old DSV
    // GREATER_EQUAL test): map this output-res pixel to its internal-res
    // texel the way NaiveTsr_Resolve does. NDC z is unaffected by the xy
    // jitter, so the comparison is exact.
    //
    // CONVENTION (capture-verified, ProbeGI5 pixelhistory): this engine's
    // projection produces clip.z = near (0.1) * view.w and clip.w = view
    // depth, so SV_Position.z = near/dist — EXACTLY the encoding the GBuffer
    // depth texture stores (the viewport's MinDepth/MaxDepth are 0/1, so
    // viewport z equals NDC z; nearer = larger). The fragment z and the
    // stored scene z are directly comparable — no remap. (An earlier edit
    // remapped z*2-1 "back to NDC" on a wrong GL-style assumption and
    // clipped every ball; reverted after pixelhistory proved occluded ball
    // fragments are discarded by the plain comparison.)
    Texture2D<float> depthTex = ResourceDescriptorHeap[FrameCB.depthIndex];
    float2 scale = float2(FrameCB.internalWidth, FrameCB.internalHeight)
                 / float2(FrameCB.outputWidth, FrameCB.outputHeight);
    int2 ip = min(int2(i.pos.xy * scale),
                  int2(FrameCB.internalWidth, FrameCB.internalHeight) - 1);
    float sceneZ = depthTex[ip];
    clip(i.pos.z - sceneZ + 1e-5f); // visible when the ball is nearer

    return i.color;
}

// =============================================================================
// Backface Leak Map — fullscreen post-composite pass (separate PSO, alpha-
// blended so sky pixels pass the tonemapped scene through). Per geometry
// pixel: reconstruct the GBuffer surface, split the trilinear cage weight
// into front/back-of-tangent-plane (BakedGICageContamination), and paint the
// contamination fraction. Display-referred colors — no inverse tonemap.
//   green = clean cage | green->red ramp = contamination | blue = no valid
//   probes (dark-leak risk) | sky (alpha 0) = scene shows through.
// =============================================================================
struct LeakOut
{
    float4 pos : SV_POSITION;
};

LeakOut LeakVS(uint vid : SV_VertexID)
{
    // Y-flipped fullscreen triangle (front-facing under default culling).
    LeakOut o;
    float2 tc = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(tc.x * 2.0f - 1.0f, 1.0f - tc.y * 2.0f, 0.0f, 1.0f);
    return o;
}

float4 LeakPS(LeakOut i) : SV_Target
{
    // Map this output-res pixel to its internal-res GBuffer texel (same
    // mapping as the depth test above).
    Texture2D<float>  depthTex  = ResourceDescriptorHeap[FrameCB.depthIndex];
    Texture2D<float4> normalTex = ResourceDescriptorHeap[FrameCB.normalIndex];

    float2 scale = float2(FrameCB.internalWidth, FrameCB.internalHeight)
                 / float2(FrameCB.outputWidth, FrameCB.outputHeight);
    int2 ip = min(int2(i.pos.xy * scale),
                  int2(FrameCB.internalWidth, FrameCB.internalHeight) - 1);

    float depth = depthTex[ip];
    if (depth <= 0.0f)
        return float4(0.0f, 0.0f, 0.0f, 0.0f); // sky — pass the scene through

    float3 N = normalize(normalTex[ip].rgb * 2.0f - 1.0f);

    // World position reconstruction (same as RestirDI_Temporal).
    float2 uv     = (float2(ip) + 0.5f) / float2(FrameCB.internalWidth, FrameCB.internalHeight);
    float4 ndc     = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= viewPos.w;
    float3 worldPos = mul(viewPos, FrameCB.viewInverse).xyz;

    float2 wb       = BakedGICageContamination(worldPos, N);
    float  wValid   = wb.x;
    float  wBack    = wb.y;

    float3 d;
    if (wValid < 1e-6f)
        d = float3(0.0f, 0.2f, 1.0f); // no valid probes — dark-leak risk
    else
    {
        float contamination = saturate(wBack / wValid);
        d = lerp(float3(0.1f, 0.9f, 0.2f), float3(0.95f, 0.15f, 0.15f), contamination);
    }
    return float4(d, 1.0f);
}
