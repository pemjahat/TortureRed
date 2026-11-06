#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#ifdef __cplusplus
    #include <DirectXMath.h>
    #include <cstdint>
    using float4   = DirectX::XMFLOAT4;
    using float4x4 = DirectX::XMFLOAT4X4;
    using float3   = DirectX::XMFLOAT3;
    using float2   = DirectX::XMFLOAT2;
    using uint     = uint32_t;
    #define ROW_MAJOR 
#else
    #define ROW_MAJOR row_major
#endif

// Anti-aliasing / post-processing modes
#define AA_MODE_NONE          0u  // No AA — single sample, no accumulation, no TAA
#define AA_MODE_ACCUMULATION  1u  // Progressive accumulation (path tracer converges over time)
#define AA_MODE_TAA           2u  // Temporal Anti-Aliasing with sub-pixel jitter

// fp16 HDR range management — pre-scale high-dynamic-range values so they fit
// within fp16 representable range without overflow. GPU-side clamp to FP16Max
// before any fp16 write; CPU-side multiply sun/sky radiance/irradiance by FP16Scale.
// At exposure time, the inverse scale (2^10) is baked into exp2(exposure + 10).
// See docs/task011-fp16hdrscale.md for details.
static const float FP16Scale    = 0.0009765625f;       // 2^-10
static const float FP16Max      = 65504.0f;            // fp16 max

// π — single source of truth for C++ and HLSL (shaders reach it via
// Common.hlsl -> Shared/SharedTypes.h; C++ via GraphicsTypes.h). Deliberately
// NOT named PHI: HLSL identifiers are case-insensitive and `phi` is the
// established azimuth / golden-angle local name in the shaders, so a PHI
// constant would shadow-collide with them. The value is bit-identical to a
// 3.14159265f literal.
static const float PI           = 3.14159265358979f;

#define RESTIR_RESERVOIR_DEBUG_OFF        0u
#define RESTIR_RESERVOIR_DEBUG_POSITION   1u
#define RESTIR_RESERVOIR_DEBUG_NORMAL     2u
#define RESTIR_RESERVOIR_DEBUG_RADIANCE   3u
#define RESTIR_RESERVOIR_DEBUG_WEIGHTSUM  4u
#define RESTIR_RESERVOIR_DEBUG_SOURCE_PDF 5u
#define RESTIR_RESERVOIR_DEBUG_TARGET_PDF 6u
#define RESTIR_RESERVOIR_DEBUG_RIS_WEIGHT 7u
#define RESTIR_RESERVOIR_DEBUG_TEMPORAL_TARGET_PDF 8u
#define RESTIR_RESERVOIR_DEBUG_SPATIAL_SHIFTED_TARGET_PDF 9u
#define RESTIR_RESERVOIR_DEBUG_W         10u

// ReSTIR DI debug modes
#define RESTIR_DI_DEBUG_OFF            0u
#define RESTIR_DI_DEBUG_LIGHT_INDEX    1u
#define RESTIR_DI_DEBUG_M_COUNT        2u
#define RESTIR_DI_DEBUG_WEIGHT         3u
#define RESTIR_DI_DEBUG_VISIBILITY_AGE 4u

// ---------------------------------------------------------------------------
// Ray-debug RECORD layout. ONE definition for three readers: the writer
// (BakedGI_RayDebug.hlsl), the viewer (BakedGI_Debug.hlsl) and the CPU readback
// (BakedGI::kRayRecordF4 in BakedGI.h). This header is compiled from source on
// both sides, so the GPU and the CPU cannot disagree about the stride.
//
// A record is 8 float4 = 128 B. rec6 and rec7 carry the SECONDARY ray — the
// shadow ray that the gather hit fires toward the active delta direction W:
//
//   rec0 = (gather direction.xyz, hit distance; 1e4 = miss)
//   rec1 = (diffuse albedo.rgb, NdotD)
//   rec2 = (L.rgb, Vhit)
//   rec3 = (Efb.rgb, status + 4 * sourceRayIndex)   status packed, see BakedGI.hlsli
//   rec4 = (hit normal.xyz, hit instance index; -1 = no surface)
//   rec5 = (hit world pos.xyz, hit primitive index; -1 = no surface)
//   rec6 = (secondary ray payload.xyz, secondary ray status)
//   rec7 = (secondary distance, secondary instance, secondary primitive, 0)
//
// rec6.xyz depends on the secondary status:
//   NONE     zero. No secondary ray was traced: the record is not shaded, or
//            NdotD <= 0, so the bake fires no shadow ray (BakedGI_Bake.hlsl).
//   OPEN     the unit active delta direction W, the direction the ray was traced
//            toward. Vhit is 1. Nothing was hit.
//   BLOCKED  the world position of the first committed hit. Vhit is 0.
//
// rec7 uses -1 in all three slots when it carries no value (NONE and OPEN).
// -1 rather than 0: instance 0 is a legal instance, distance 0 is a legal
// distance, and a reader must be able to tell "none" from "the first one".
//
// The payload exists so a white Vhit can be checked without trusting it. W is
// STORED, not re-derived in the viewer, because the SUN source's W comes from
// the replayed hour state, which the viewer does not receive; a derivation
// there would need a new FrameConstants field.
#define BAKED_GI_RAY_RECORD_F4 8u

// Secondary ray status, in rec6.w. Plain float, no packing: the field has room.
// One definition for the writer, the viewer, the CPU readback and the log.
#define BAKED_GI_RAY_SHADOW_NONE    0u
#define BAKED_GI_RAY_SHADOW_OPEN    1u
#define BAKED_GI_RAY_SHADOW_BLOCKED 2u

// ---------------------------------------------------------------------------
// Photon-map deposit layout (first-bounce sun deposits, debug-first). ONE
// definition for three readers, the same contract as the ray record above: the
// writer (BakedGI_PhotonEmit.hlsl), the viewer (BakedGI_Debug.hlsl) and the
// CPU buffer sizing / asserts (BakedGI::kPhotonCap / kPhotonF4 in BakedGI.h).
//
// One deposit is 3 float4 = 48 B. The unit-delta convention holds (sun
// irradiance 1), so a deposit's power is diffuseAlbedo * quadArea / N:
//   rec0 = (position.xyz, Phi.r)
//   rec1 = (Phi.g, Phi.b, n.s, sun-ray hit distance)
//   rec2 = (surface normal.xyz, 0)
// The counts buffer holds probeCount + 1 uints: one append counter for each
// cell (it counts ATTEMPTS and rises past the cap), plus one slot for rays
// whose hit left the probe grid. The CPU walks the same counters
// (BakedGI::LogPhotonStats) to build the view's dense INDEX LIST: one entry
// for each STORED deposit (cell * cap + slot), so an instance of view 10
// reaches every occupied cell and not only the first cells of the grid.
#define BAKED_GI_PHOTON_CAP 64u   // deposit slots for each grid cell
#define BAKED_GI_PHOTON_F4  3u    // float4s for each deposit

// Core Frame Constants
struct FrameConstants {
    ROW_MAJOR float4x4 viewProj;
    ROW_MAJOR float4x4 viewInverse;
    ROW_MAJOR float4x4 projectionInverse;
    ROW_MAJOR float4x4 viewProjPrevious;
    ROW_MAJOR float4x4 viewInversePrevious;
    float4 cameraPosition;
    float4 prevCameraPosition;
    uint frameIndex;
    int albedoIndex;
    int normalIndex;
    int materialIndex;
    int depthIndex;
    float exposure;
    uint enableRestir;            // path-tracer ReSTIR GI
    uint enableAvoidCaustics;
    uint enableIndirectSpecular;
    uint enableRestirGI;          // raster-path ReSTIR GI (exclusive with bakedGIMode)
    uint useRTXDI;
    uint numLights;
    uint lightSamplingMode;
    uint lightLUTBufferIndex;
    uint screenWidth;
    uint screenHeight;
    float sharcSceneScale;
    uint sharcAccumulationFrameNum;
    uint sharcStaleFrameNum;
    uint sharcDebug;
    uint restirReservoirDebugMode;
    uint mouseSelectedPixelX;
    uint mouseSelectedPixelY;
    uint pathVizEnabled;  // 1 for exactly one frame after left-click
    uint enableReservoirLobeCheck; // 1 = lobe-aware temporal/spatial reuse (lobe-matched PDF, roughness-scaled Jacobian/history)
    uint enableNrdRelax;
    uint enableNrdValidation;
    float rtrRoughReuseThreshold; // Roughness above which specular reuses diffuse candidate (default 0.6)

    // TAA / Temporal Super-Resolution
    float2 taaJitter;           // Sub-pixel jitter in pixel units of internal resolution
    uint   taaEnabled;          // 1 = TAA active
    uint   internalWidth;       // Internal render resolution width
    uint   internalHeight;      // Internal render resolution height
    uint   outputWidth;         // Output (swap chain) resolution width (1920)
    uint   outputHeight;        // Output (swap chain) resolution height (1080)
    uint   taaHistoryIndex;     // Ping-pong index for TAA history (0 or 1)
    float  taaUpsamplingFactor; // Temporal upsampling factor (1.0 - 4.0)
    uint   taaFrameCounter;    // Monotonically increasing counter for TAA jitter sequence (never reset)
    uint   _pad0;              // 8-byte padding (2x uint) to align projectionInverseUnjittered to 16 bytes (offset 512)
    uint   _pad1;
    ROW_MAJOR float4x4 projectionInverseUnjittered; // Unjittered projection inverse for motion vectors
    ROW_MAJOR float4x4 viewProjUnjittered;          // Unjittered view-proj for post-composite debug overlays

    // ReSTIR DI (Direct Illumination) flags
    uint enableRestirDI;              // 1 = ReSTIR DI active (direct lighting from local lights)
    uint restirDIDebugMode;           // RESTIR_DI_DEBUG_* enum
    uint _diPad0;
    uint _diPad1;

    // Sky system (Hosek-Wilkie)
    uint skyCubemapIndex;       // Bindless SRV index of the baked sky TextureCube
    uint skySH9BufferIndex;     // Bindless SRV index of the SkySH9Buffer (9x float4)
    float skyTurbidity;         // Hosek-Wilkie turbidity [1, 10] — fixed default in code
    float skyGroundAlbedo;      // Hosek-Wilkie ground albedo [0, 1] — fixed default in code

    // Baked GI probe system (task017 step 1) — EXCLUSIVE indirect-GI source:
    // bakedGIMode 1 = probe system, 0 = ReSTIR GI; never both.
    uint   bakedGIMode;             // 1 = baked probes, 0 = ReSTIR GI
    uint   bakedGIValid;            // 1 = bake completed and buffers are live
    uint   bakedGIProbeSRVIndex;    // StructuredBuffer<float4> lit probes, 9 per probe (total = sky + sun)
    uint   bakedGIProbeMetaSRVIndex;// StructuredBuffer<uint> per-probe validity (bit 0)
    uint   bakedGIProbeSkySRVIndex; // sky-sourced lit split (debug balls): sky bounces + occluded sky direct
    uint   bakedGIProbeSunSRVIndex; // sun-sourced lit split (debug balls): sun bounces only
    uint   bakedGIProbeVisSRVIndex; // StructuredBuffer<float2> per-probe octahedral 16x16 depth moments
                                    // (padded 18x18, (M1, M2) per texel) — step-3 Chebyshev visibility
    float  bakedGIGridMinX;         // grid AABB min (world)
    float  bakedGIGridMinY;
    float  bakedGIGridMinZ;
    float  bakedGISpacing;          // probe spacing (world units)
    uint   bakedGIDimX;             // grid dims (probes per axis, cells are [0, dim-1])
    uint   bakedGIDimY;
    uint   bakedGIDimZ;

    // Probe debug overlay (BakedGI_Debug.hlsl): view enum + lit-preview scale.
    uint   bakedGIDebugView;        // 0 off, 1 placement, 2 lit irradiance, 3 sky visibility,
                                    // 4 unused, 5 lit sky source, 6 lit sun source,
                                    // 7 ray fan (one probe's gather rays),
                                    // 8 efb contribution (per-probe SH ball of R^k - R^0 for one bake term;
                                    //   the term is the UI's "Iteration" 1..3 — see bakedGIEfbIterSRVIndex)
    // Per-probe raw SH9 SKY VISIBILITY: 9 floats for each probe, stride 4 B.
    // StructuredBuffer<float>, written by the visibility bake and read by the V
    // view (bakedGIDebugView 3). It is the ONE source of the V the fold
    // multiplies, so no response-table slot carries a visibility any more.
    //
    // 0xFFFFFFFF means "no SH data": no bake has run, or a Probes/ cache load
    // restored the tables without this section. 0 is a legal descriptor index, so
    // it cannot mean "none" — the view early-outs on the sentinel before it
    // declares the SRV. BakedGI::kSkyVisNoSRV is the C++ spelling.
    uint   bakedGISkyVisSRVIndex;
    float  bakedGIDebugScale;       // lit-ball gain (DC-normalized shading multiplier)

    // Fullscreen backface-contamination map (BakedGI_Debug.hlsl LeakPS,
    // post-composite): fraction of the trilinear blend from probes BEHIND
    // the receiver's tangent plane.
    uint   bakedGILeakDebug;        // 1 = active (draws over the tonemapped scene)
    uint   bakedGIVisCheck;         // 1 = step-3 weighted fetch (bias + backface + Chebyshev), 0 = plain step-1 trilinear (A/B)

    // DDGI probe system. 
    // a runtime octahedral irradiance atlas, re-traced every frame. EXCLUSIVE
    // with bakedGIMode and enableRestirGI; the UI arbitrates the three.
    //
    // The atlas indices below point at the CURRENT frame's targets, so the
    // deferred fetch reads the atlas the resolves wrote this frame. The
    // renderer transitions both atlases to a pixel-read state at the end of
    // the DDGI record.
    uint   ddgiMode;                // 1 = DDGI is the indirect source
    uint   ddgiValid;               // 1 = the atlas and the probe buffers are live
    uint   ddgiIrradianceSRVIndex;  // Texture2D<float4> irradiance atlas (tone-curved, gamma 5)
    uint   ddgiDepthSRVIndex;       // Texture2D<float2> depth-moment atlas (M1, M2)
    uint   ddgiProbeStateSRVIndex;  // StructuredBuffer<uint> 0 = active, 1 = inactive
    uint   ddgiProbeOffsetSRVIndex; // StructuredBuffer<float4> relocation offsets (xyz)
    float  ddgiGridMinX;            // grid AABB min (world)
    float  ddgiGridMinY;
    float  ddgiGridMinZ;
    float  ddgiSpacing;             // probe spacing (world units)
    uint   ddgiDimX;                // grid dims (probes per axis)
    uint   ddgiDimY;
    uint   ddgiDimZ;
    uint   ddgiProbesPerRow;        // atlas row length in probe tiles (linear packing)
    uint   ddgiNumRaysPerProbe;     // rays cast for each probe (32 stable + rotating)
    uint   ddgiMaxRaysPerProbe;     // ray hit buffer stride for each probe
    uint   ddgiDebugView;           // 0 off, 1 irradiance spheres, 2 probe states, 3 depth spheres
    float  ddgiDebugScale;          // irradiance sphere gain

    // Indirect-only scene view (Lighting.hlsl early-out). 1 = the shaded image
    // is replaced by the indirect-GI term alone: direct sun, local lights,
    // ReSTIR composition and indirect specular are all skipped, so the bounced
    // light can be inspected on its own. Sky pixels render black, because the
    // sky is direct environment light and would mask the signal.
    // Appended last on purpose: every earlier offset is unchanged.
    uint   giIndirectOnly;          // 1 = render the indirect contribution only
    uint   giIndirectOnlySource;    // 0 = full (sky + sun), 1 = sky only, 2 = sun only
                                    // (honoured only by the baked probe system, which
                                    //  is the only source carrying a sun/sky split;
                                    //  DDGI stores one unsplit indirect field)
    uint   giIndirectOnlyAlbedo;    // 0 = the surface albedo, 1 = all-white albedo.
                                    // Removes the albedo texture as a variable, so
                                    // the indirect term can be read as a pure
                                    // lighting field. The Lambert 1/PI is kept, so
                                    // the white view stays directly comparable with
                                    // the normal-albedo one.

    // Ray Fan debug view (bakedGIDebugView == 7): the 64 gather rays of ONE delta
    // record of ONE probe, read from the ray-debug buffer. The record is one the
    // dump replayed: the sun hour state, or one of the fibonacci sky bins
    // (bakedGIRayDebugSource names which). A SKY dump records ALL bins, so the sky
    // bin is chosen for display and needs no second dump. These are the same records
    // the text dump ranks, so the fan and the dump always describe one probe.
    // Appended last on purpose: every earlier offset is unchanged.
    uint  bakedGIRayDebugSRVIndex;  // StructuredBuffer<float4>: the ray records, then the re-accumulated table.
                                    // 0xFFFFFFFF = no ray-debug buffer yet. 0 is a legal descriptor index,
                                    // so it cannot mean "none": the fan shader early-outs on the sentinel.
    uint  bakedGIRayDebugProbe;     // the probe the fan belongs to (matches the dump)
    uint  bakedGIRayDebugChannel;   // 0 status, 1 albedo, 2 NdotD, 3 Vhit, 4 Efb, 5 distance,
                                    // 6 direct-vs-Efb, 7 DeltaDirections, 8 VHit Ray.
                                    // Channel 7 also swaps the GEOMETRY: the 16 fibonacci delta
                                    // directions at this probe instead of the 64 gather rays,
                                    // grayscale by the sun fold's weight, white = the direction
                                    // the fan itself draws. It reads no record, so it needs no
                                    // dump. See BakedGI::kRayFanChannelDeltaDirs.
                                    // Channel 8 also swaps the geometry, but it READS records: one
                                    // thin segment for each shaded gather ray, from that ray's hit
                                    // toward the active delta direction its shadow ray was traced
                                    // toward. It draws rec5, rec6 and rec7. Cyan = the secondary
                                    // ray is open, red = it stopped at the recorded first hit. See
                                    // BakedGI::kRayFanChannelVHitRay.
                                    // 0..8 is the whole set.
    float bakedGIRayDebugLength;    // ray length cap, in probe spacings
    uint  bakedGIRayDebugValid;     // 1 = do not dim. The probe is valid, or its validity is
                                    // unknown (the meta read failed). 0 = the probe is KNOWN
                                    // invalid, so the fan dims the L, Efb and regime channels.
                                    // Set on every dump and bound to its probe, so no verdict
                                    // carries to a later probe. The Application fills this again
                                    // right after the dump, so the bound frame has the verdict.
    uint  bakedGIRayDebugDir;       // which record GROUP of the ray-debug buffer the fan draws.
                                    // This is the DISPLAYED group, not the dump's request: SUN
                                    // always 0 (the sun table holds one record for each probe),
                                    // SKY the displayed fibonacci bin, which the panel's bin
                                    // browser can move without a dump. 0xFFFFFFFF means "no
                                    // dump", and the fan rejects it the same way it rejects a
                                    // group out of range. Published by
                                    // Application::FillRayDebugConstants from BakedGI, never from
                                    // the UI selection directly.

    // Sky-quadrature weights, read by the Ray Fan's DeltaDirections channel
    // (bakedGIRayDebugChannel == 7). One entry for each BakedGI::kDirections /
    // BAKED_GI_DIRECTIONS direction, in index order.
    //
    // These are sky weights, not sun weights: the sun is 17 baked hour states, so
    // no per-bin sun weight exists. The 16 directions are the hemisphere sky
    // quadrature, which weights every direction equally, so
    // every entry is 1/16 and the channel draws them all at the same brightness.
    //
    // The field is KEPT at this size and this offset on purpose: a shrunken array
    // would shift every field after it, and the DeltaDirections channel still
    // needs per-direction values.
    //
    // The channel ramps on this array DIVIDED BY ITS OWN MAXIMUM, not on the
    // share itself, so a future non-uniform set still reads white on its winner.
    // That divisor is a display choice and lives in BakedGI_Debug.hlsl
    // (BakedGISunWeightMax's call site).
    //
    // LAYOUT — a cbuffer array of SCALARS is VECTOR-STRIDED: dxc gives each
    // element a full 16-byte register, so `float x[16]` is 244 bytes wide in HLSL
    // and 64 in C++. Measured on this file's own compiled blob, not assumed: with
    // the scalar array the blob reflected bakedGISunWeight at 864 and the field
    // after it at 1108 (the first free component of the last element's register).
    // With the float4 array below, that same next field sits at 1120 on both
    // sides. A tight C++ scalar array therefore put every following field 188
    // bytes early, which is why the Efb indices read zero (past the constant
    // buffer's size) and the sun-weight ramp read two directions out. A float4
    // array has no such trap — its element stride is its own size, so both
    // languages agree — and one float4 for each direction is exactly one 16-byte
    // register each. Only .x carries the weight.
    // Do not turn this back into a scalar array, and do not add a field between
    // this array and the next without re-measuring both sides.
    //
    // Appended last on purpose: every earlier offset is unchanged.
    uint   _sunWeightPad0;          // 856, 860: the 8 bytes HLSL leaves before the
    uint   _sunWeightPad1;          //           16-byte boundary the array needs
    float4 bakedGISunWeight[16];    // 864: one entry for each direction, weight in .x
                                    //      = BakedGI::kDirections / BAKED_GI_DIRECTIONS

    // Per-bake-term response balls — the Efb contribution view (bakedGIDebugView
    // == 8). The bake folds each term's transport table into a per-probe SH9
    // "response ball" (BakedGI_IterFold.hlsl), so the view can show ONE term's
    // table without reading the 16x9 table in the vertex shader.
    //
    // Both indices are "no ball" sentinels (0xFFFFFFFF, see
    // BakedGI::kEfbBallNoSRV) until a bake has folded the terms, and the view
    // early-outs on the sentinel. 0 is a legal descriptor index, so it cannot
    // mean "none".
    //
    // The view shows ball[selected] - ball[term 1]: the second index is always
    // term 1, the no-feedback baseline (the bake gates feedback off in term 1,
    // BakedGI_Bake.hlsl:236), so the difference IS the contribution the series
    // feedback (Efb) term added by the selected term. Selecting term 1 makes the
    // two indices equal and the view black — by construction, not by failure.
    //
    // Appended last on purpose: every earlier offset is unchanged.
    uint bakedGIEfbIterSRVIndex;    // the ball of the SELECTED term (UI "Iteration" 1..3 -> 0..2)
    uint bakedGIEfbBaseSRVIndex;    // the ball of term 1 — the no-feedback baseline

    // Single-direction ("un-averaged") mode of the same view. The fold is a SUM
    // over the 16 delta directions, so one tinted direction is diluted against 15
    // neutral ones and can be invisible in the ball. bakedGIEfbDir selects ONE of
    // the 16: the view then reads the two TRANSPORT TABLES directly and shows
    // wq*(R^k_d - R^0_d) — the fold with its sum restricted to one direction, so
    // that direction is shown on its own.
    //
    // Any value outside 0..BakedGI::kDirections-1 means "fold all 16". The default
    // is BakedGI::kEfbDirAll (0xFFFFFFFF), deliberately NOT 0: direction 0 is a
    // real direction and a default of 0 would silently un-average it.
    //
    // Both table indices carry the same "no capture" sentinel as the balls, and
    // the view early-outs per mode on the TWO sources that mode reads. The tables
    // are the fold pass's own input, so a bake that folded also snapshotted them.
    //
    // Appended last on purpose: every earlier offset is unchanged.
    uint bakedGIEfbTermSelSRVIndex;  // transport table of the SELECTED term
    uint bakedGIEfbTermBaseSRVIndex; // transport table of term 1 (R^0)
    uint bakedGIEfbDir;              // 0..15 = that direction alone; else fold all 16

    // Cage Trace (bakedGIDebugView == 9) gather-ray OVERRIDE.
    //
    // By itself the view derives its ray from the records: the largest carried
    // feedback among the rays the feedback DOMINATES, inside the record group the
    // dump wrote. That derivation is a ranking, and a ranking is the wrong tool when
    // the ray being hunted is not the brightest one. This field lets the UI name the
    // ray instead.
    //
    // 0..BakedGI::kDebugGatherRays-1 = draw that gather-ray RECORD INDEX of the
    // dumped record group (bakedGIRayDebugDir). BakedGI::kNoRayPick (0xFFFFFFFF) =
    // no override: the shader keeps its own argmax. A ray index may be 0, so 0
    // cannot mean "none". This is a RECORD index, not a source gather-ray index: the
    // debug pass replays BakedGI::kBakeGatherRays rays and keeps kDebugGatherRays
    // records.
    //
    // The pool the UI enumerates is built on the CPU from the dump's own
    // readback and holds exactly the rays where the feedback term beats the
    // direct term at the hit — "direct < Efb", the same rule the shader's argmax
    // applies (BakedGI::GetRayCandidates, BakedGI::LogRayDebugReadout step 2c).
    //
    // A FORCED index skips that dominance test in the shader. The user picked the
    // ray out of a list that already passed it, so repeating the test could only
    // disagree on the last bit of a luminance and silently draw nothing. The
    // regime is reported in the colour instead: WHITE probe->hit segment = the
    // feedback dominates at that hit, MAGENTA = it does not, which is what an
    // override out of another dump's pool (the probe, the source or the bin changed
    // since the dump) looks like.
    //
    // Appended last on purpose: every earlier offset is unchanged.
    uint bakedGIRayDebugRay;

    // The REAL sun, for the Ray Fan's DeltaDirections channel (debug view 7,
    // BAKED_GI_RAYFAN_CH_DELTADIRS). That channel draws the 16 fibonacci delta
    // directions, and without this field it could not draw the direction those
    // bins are being compared against. The gap between the sun and the bins is
    // then visible in the 3D view instead of readable only in the Sun Fold Bins
    // table. The shader draws it as a 17th, thicker, ORANGE box.
    //
    // xyz = the to-sun UNIT direction at the current HOUR, from
    //       BakedGI::SunDirectionAtHour. It is the GI sun, not an independent
    //       light direction: Application::Render writes the same direction into
    //       the directional light, so the two stay equal.
    //       w = 1.0 inside daylight, 0.0 otherwise — outside
    //       hour 6 to 18 the sun term is zero, so the marker does not draw. With
    //       w = 0 the xyz is written as zero too, so no stale direction survives.
    //
    // Ownership: Application::FillSunFoldConstants writes it every frame, from the
    // same BakedGI::SunBlendForHour call that fills the Ray Fan labels. One
    // source, so the label text and the 3D marker cannot disagree.
    //
    // LAYOUT — this is a float4 appended after a uint, so it needs explicit pads
    // in C++ only. HLSL will not split a vector across a 16-byte register: it
    // places a float4 at the next 16-byte boundary, which is 1152, while C++
    // would place the field at 1144 (a XMFLOAT4 is 4-byte aligned). The two uint
    // pads below move the C++ offset to 1152. This is the same arithmetic, and
    // the same reason, as _sunWeightPad0/_sunWeightPad1 above. Measured, not
    // assumed: offsetof says bakedGIRayDebugRay 1140, the pads 1144 and 1148,
    // this field 1152, sizeof 1168; the compiled shader blob reports the same
    // four numbers. Re-measure both sides when any field BEFORE this one changes
    // size or count: the pads hold this alignment for this exact prefix only.
    //
    // Appended last on purpose: every earlier offset is unchanged.
    uint   _sunDirPad0;             // 1144, 1148: the 8 bytes HLSL leaves before the
    uint   _sunDirPad1;             //           16-byte boundary this field needs
    float4 bakedGISunDir;           // 1152: xyz = to-sun unit direction, w = sun exists

    // Runtime data eligibility for FILLED probes. The receiver shaders accept
    // FILLED only when this flag is one. Raw A/B mode publishes zero, so FILLED
    // meta bits stay on disk but cannot enter a runtime cage.
    // Appended after the float4. Measured on both sides: offset 1168. The upload
    // buffer rounds the byte count to 256, so the constant-buffer binding covers
    // this field.
    uint bakedGIUseFilled;          // 1168: 1 = runtime cage accepts FILLED probes

    // The delta source the last dump replayed, and the sky bin ON DISPLAY. These
    // describe the RECORDS IN THE BUFFER, not the UI selection, so a label can
    // never name a source other than the one the bytes came from. A UI change of
    // the source selection does not touch them; only a completed dump does.
    //
    //   bakedGIRayDebugSource  0 = SUN (the shipped replay), 1 = SKY. Any other
    //                          value is read as SUN.
    //   bakedGIRayDebugSkyDir  0..BAKED_GI_DIRECTIONS-1 in SKY mode. 0xFFFFFFFF
    //                          means "no dump", or "the SUN source", and the
    //                          DeltaDirections channel then highlights no bin.
    //
    // A SKY dump records all BAKED_GI_DIRECTIONS bins, so the displayed bin is a
    // VIEW choice, not a property of the dump. The panel's bin browser moves it,
    // and the cyan marker of the DeltaDirections channel follows it, so the marker
    // and the records on screen always name one bin.
    // Read by the Ray Fan's DeltaDirections channel (bakedGIRayDebugChannel == 7)
    // and by the UI text. These uints are appended after the 4-byte scalar at 1168,
    // so none of them needs a pad: HLSL places a scalar on a 4-byte boundary and no
    // vector crosses a 16-byte boundary here. Measured on both sides:
    // bakedGIUseFilled 1168, bakedGIRayDebugSource 1172, bakedGIRayDebugSkyDir
    // 1176, sizeof 1180. Re-measure both sides when a field BEFORE these
    // changes size or count.
    uint bakedGIRayDebugSource;     // 1172: 0 = SUN, 1 = SKY — the dumped source
    uint bakedGIRayDebugSkyDir;     // 1176: the displayed sky bin, or 0xFFFFFFFF

    // Photon map (first-bounce sun deposits, view 10). The map is baked on
    // demand for ONE hour state (BakedGI::RecordPhotonEmit), so these describe
    // the RESIDENT map, not a UI selection — the same contract as the ray-debug
    // source fields above.
    //
    //   bakedGIPhotonSRVIndex   StructuredBuffer<float4> deposits,
    //                           BAKED_GI_PHOTON_F4 float4 each, or the
    //                           kPhotonNoSRV sentinel (0 is a legal descriptor).
    //   bakedGIPhotonCount      deposited photons = the view's instance count.
    //   bakedGIPhotonSunState   the hour state the map was baked for.
    //   bakedGIPhotonFluxScale  rays / quadArea. A deposit's power times this
    //                           scale is its diffuse albedo, so the view colours
    //                           by where the flux landed without an arbitrary
    //                           gain.
    //   bakedGIPhotonIdxSRVIndex StructuredBuffer<uint>, the dense index list
    //                           of the STORED deposits (instance -> record),
    //                           built by LogPhotonStats from the count
    //                           readback, or the kPhotonIdxNoSRV sentinel.
    //
    // Appended last on purpose: every earlier offset is unchanged. All five
    // are 4-byte scalars after a uint, so no pad is needed on either side.
    // Measured on both sides: 1180, 1184, 1188, 1192, 1196, sizeof 1200.
    uint  bakedGIPhotonSRVIndex;    // 1180: deposits SRV, or 0xFFFFFFFF
    uint  bakedGIPhotonCount;       // 1184: deposited photon count
    uint  bakedGIPhotonSunState;    // 1188: hour state of the resident map
    float bakedGIPhotonFluxScale;   // 1192: rays / quadArea (albedo recovery)
    uint  bakedGIPhotonIdxSRVIndex; // 1196: stored-deposit index list SRV, or 0xFFFFFFFF
};

// Root constants (compute slot 12 / b2) for the BakedGI bake dispatches.
// One (probe, delta-direction) thread each; see Shaders/BakedGI_Bake.hlsl.
struct BakedGIBakeParams {
    float gridMinX, gridMinY, gridMinZ;
    float spacing;
    uint  dimX, dimY, dimZ;
    uint  probeCount;
    uint  responseUAVIdx;   // RW output (this iteration)
    uint  responsePrevSRVIdx; // SRV feedback (previous iteration)
    uint  metaUAVIdx;
    uint  iteration;        // 0 = no feedback (first series term)
    uint  visSRVIdx;        // SRV: per-probe octahedral depth moments — the feedback
                            // read is visibility-weighted (backface + Chebyshev) so
                            // series light cannot cross walls/roofs (ProbeGI4 leak)

    // ---- Delta-light source: sky transport, or one sun hour state ----
    // isSunPass 0: the thread's `dir` selects one of the 16 fibonacci bins, and the
    //              pass writes the 16-direction table. Unchanged.
    // isSunPass 1: the delta light is the SUN AT ONE HOUR STATE. Only dir 0 works,
    //              `wDelta` comes from sunDirX/Y/Z below, sunState names the state,
    //              and responseUAVIdx / responsePrevSRVIdx point at the per-probe
    //              SUN table (stride BAKED_GI_SUN_STATES, indexed by sunState)
    //              instead of the 16-direction table. ONE pair of indices,
    //              interpreted per pass: a second pair would be two places to get
    //              wrong.
    // BOTH passes run on every bake: the 16-direction table is the live sky
    // quadrature, and the sun table is what the hour fold interpolates. Skipping
    // either one breaks the frame.
    uint  isSunPass;
    float sunDirX, sunDirY, sunDirZ;   // TO-SUN unit direction, isSunPass 1 only

    // ---- Diagnostic: pair the sun seed with a fibonacci bin ----
    // The bake's RNG seed depends on the direction index, so the sun pass (which
    // runs with dir 0) gathers a DIFFERENT set of directions than any bin. A
    // comparison between the two methods then mixes an angular difference with a
    // sampling difference. This field overrides the index that enters the hash,
    // so both passes can trace the SAME gather directions and only wDelta
    // differs — the same hit surfaces are re-tested toward the sun.
    //   0xFFFFFFFF = use the thread's own `dir`. Today's behaviour, exactly.
    //   0..15      = use that bin index in the seed hash instead.
    // BakedGI::kSunSeedOwnDir is the C++ spelling of 0xFFFFFFFF. The ray-replay
    // pass carries the same field, or its self-check reports MISMATCH.
    //
    // APPENDED last, so every earlier offset is unchanged. A uint needs 4-byte
    // alignment only, so it lands directly after sunDirZ at 68 with no pad. The
    // struct becomes 72 B = 18 dwords, which the root-constant call passes in
    // full, so the shader never reads a stale dword. Measured on both sides:
    // dxc reports the member at 68 and the struct at 72.
    uint  seedDir;
    // ---- Equinox hour states: which state this dispatch bakes ----
    // 0..BakedGI::kSunStates-1. isSunPass 1 only. The sun record stride is
    // BAKED_GI_SUN_STATES, so the shader indexes
    // slot = (probe * 17 + sunState) * kResponseFloat4s.
    // APPENDED again, so the offsets above are unchanged: 72, sizeof 76 (19
    // dwords), still far below the 30-dword DDGIParams that sizes root slot 12.
    uint  sunState;

    // Progressive gather batch. Batch zero initializes the response. Later
    // batches update the linear running mean with weight 1 / (batchIndex + 1).
    // APPENDED last: offset 76, sizeof 80 (20 dwords), below the 30-dword slot.
    uint  batchIndex;
};

// Root constants for the per-frame lit-probe update dispatch.
//
// OFFSETS, measured on both sides after the 2026-10-29 cleanup:
// sunIrradianceX 0, hour 12, responseSRVIdx 16, litSunUAVIdx 32, dimZ 60,
// sunResponseSRVIdx 64, sizeof 68. The offsets are contiguous: nothing here is a
// pad. SharedTypes.h is compiled from source on both sides, so a shifted layout
// costs only a shader cache wipe.
struct BakedGIUpdateParams {
    float sunIrradianceX, sunIrradianceY, sunIrradianceZ; // sun perpendicular irradiance (RGB)
    // 12: the GI sun input. TRUE SOLAR TIME in hours, 0..24, hour 12 = solar noon.
    // The direction is not an input: the 17 baked hour states already bracket the
    // hour.
    float hour;
    uint  responseSRVIdx;  // final response table (SRV), the 16-record sky basis
    uint  metaSRVIdx;      // probe validity (SRV)
    uint  litUAVIdx;       // lit probe output, TOTAL (runtime fetch source)
    uint  litSkyUAVIdx;    // lit probe output, sky-sourced split (debug views)
    uint  litSunUAVIdx;    // lit probe output, sun-sourced split (debug views)
    float gridMinX, gridMinY, gridMinZ;
    float spacing;
    uint  dimX, dimY, dimZ;
    uint  sunResponseSRVIdx; // SRV: per-probe sun response, stride BAKED_GI_SUN_STATES

    // ---- Sky visibility source for the fold's V ----
    //
    // APPENDED after the fields above, so every earlier offset is unchanged.
    // Measured on both sides: skyVisSHSRVIdx 68, skyVisMode 72, sizeof 76 (19
    // dwords). The shared root slot 12 is sized by the 30-dword DDGIParams, so
    // this struct does not grow the slot.
    //
    // The buffer holds 9 raw SH9 coefficients for each probe, stride 4 bytes, so
    // 36 B for each probe. It is the sky visibility projected from the SAME
    // 16x16x4 rays the depth-moment map uses, with no cosine A_l factor. Read it
    // only when skyVisMode is 0: the CPU sends 0 only when the buffer exists, so
    // index 0 never reaches a read.
    uint  skyVisSHSRVIdx; // SRV: per-probe sky-visibility SH9, stride 9 floats
    uint  skyVisMode;     // 0 = SH9 buffer (the default), 1 = V = 0 (no buffer read).
                          // These are BakedGI::SkyVisMode's values, and there are
                          // exactly two.
};


// Root constants for the on-demand ray-debug dispatch. DEBUG ONLY — this pass
// never runs in a normal frame. One 64-thread group for each direction of ONE
// probe; lane 0 replays the 64 gather rays of BakedGI_Bake.hlsl and writes one
// 128 B record for each ray. The re-accumulated table shares the same buffer,
// after the records. See Shaders/BakedGI_RayDebug.hlsl and
// BAKED_GI_RAY_RECORD_F4 above for the record layout.
//
// The shared root-constants slot (root parameter 12) is sized by DDGIParams,
// which is 30 dwords. This struct is 18 dwords, so the slot does not change.
struct BakedGIRayDebugParams {
    float gridMinX, gridMinY, gridMinZ;
    float spacing;
    uint  dimX, dimY, dimZ;
    uint  probeIndex;          // the probe to inspect
    uint  rayRecUAVIdx;        // RW: ray records, then the re-accumulated table
    uint  responsePrevSRVIdx;  // SRV: previous bake iteration's table (the Efb source)
    uint  visSRVIdx;           // SRV: per-probe octahedral depth moments
    // The fan replays ONE trace toward ONE delta source, exactly as the bake runs
    // it, so the self-check compares against that source's record: the sun table's
    // hour state when source = SUN, the sky table's fibonacci bin when source =
    // SKY. `sunDir` is read only by the SUN source.
    float sunDirX, sunDirY, sunDirZ;   // TO-SUN unit direction

    // Same field, same meaning, same expression as BakedGIBakeParams::seedDir.
    // The replay reproduces the bake's RNG stream by construction, so a bake that
    // seeds from a bin while the replay seeds from dir 0 traces other rays and the
    // self-check reports MISMATCH for a reason that is not a defect. Both sides
    // read one caller variable, never two copies.
    //
    // SUN source: the hour state index. SKY source: the field is inert. The sky
    // bake seeds bin j from j, and each group of a SKY dispatch IS one bin, so the
    // shader seeds every group from its own group index. A single shared index
    // would seed all 16 groups from one bin, and 15 of them would report MISMATCH
    // for a reason that is not a defect.
    uint  seedDir;
    // Which hour state to replay, 0..BakedGI::kSunStates-1. READ ONLY WHEN
    // `source` IS SUN. It is still written on a SKY dump, so the value is never
    // an uninitialised field.
    uint  sunState;

    // --- WHICH DELTA SOURCE THE DUMP REPLAYS ---
    //
    // The bake stores TWO response tables for each probe. The SKY table has
    // BAKED_GI_DIRECTIONS (16) records, one for each fibonacci quadrature
    // direction. The SUN table has BAKED_GI_SUN_STATES (17) records, one for each
    // equinox hour state. A replay must name the table it reads, because the same
    // numeric index means a different record in each one.
    //
    //   source = 0  SUN  the shipped replay: one hour state, its own direction and
    //                    its own seed. Bits-for-bit the behaviour before this field.
    //   source = 1  SKY  EVERY fibonacci direction: group d traces toward
    //                    BakedGIDeltaDirection(d) and compares against that bin's
    //                    sky record. One dispatch therefore fills all 16 record
    //                    groups, so the UI can show any bin without a new dump.
    //
    // Any value other than 1 is read as SUN, so the default struct (all zero)
    // keeps the shipped path.
    //
    // The two spellings are BAKED_GI_RAYDEBUG_SOURCE_SUN / _SKY (BakedGI.hlsli)
    // in the shaders and BakedGI::RayDebugSource (BakedGI.h) in C++.
    uint  source;

    // 68 B = 17 dwords. Measured with dxc: seedDir 56, sunState 60, source 64,
    // sizeof 68. The root-constant slot is sized by the 30-dword DDGIParams, so it
    // still fits. There is no sky-bin field: a SKY dispatch takes the bin from the
    // group index, which is why one dispatch can fill all 16 groups.
};

// Root constants (compute slot 12 / b2) for the on-demand photon-map emit
// dispatch. DEBUG-FIRST: the transport bake does not read the photon grid yet.
// One thread per sun ray; see Shaders/BakedGI_PhotonEmit.hlsl.
//
// ALL SCALARS, no vectors: HLSL never splits a vector across a 16-byte
// boundary but C++ packs a float3 at 4 bytes, so the quad and sun directions
// are spelled as individual floats — the same discipline BakedGIBakeParams
// uses for gridMin. Every offset is then identical on both sides with no pad.
//
// 108 B = 27 dwords, below the 30-dword DDGIParams that sizes root slot 12.
// Measured on both sides: sunState 44, quadOrigin 48, raysX 84, sizeof 108.
struct BakedGIPhotonEmitParams {
    float gridMinX, gridMinY, gridMinZ;
    float spacing;
    uint  dimX, dimY, dimZ;
    uint  probeCount;                    // cells; counts[probeCount] is the out-of-grid slot
    float sunDirX, sunDirY, sunDirZ;     // TO-SUN unit direction of the hour state
    uint  sunState;                      // 0..BakedGI::kSunStates-1 (seed + labels)

    // The emission quad: corner plus the two FULL-extent edge vectors. The CPU
    // builds it from the scene bounds projected perpendicular to the sun, with
    // the quad placed outside the scene on the sun side.
    float quadOriginX, quadOriginY, quadOriginZ;
    float quadRightX, quadRightY, quadRightZ;
    float quadUpX, quadUpY, quadUpZ;

    uint  raysX, raysY;                  // emission grid resolution (N = raysX * raysY)
    uint  photonUAVIdx;                  // RW: deposits, BAKED_GI_PHOTON_F4 float4 each
    uint  countUAVIdx;                   // RW: append counters, probeCount + 1 uints
    uint  photonCap;                     // BAKED_GI_PHOTON_CAP (the CPU mirror sizes the buffer)
    float rayTMax;                       // crosses the whole scene from the quad
};

// Root constants for the visibility-moment bake dispatch (step 3). One
// 16x16-thread group per probe; see Shaders/BakedGI_Visibility.hlsl.
struct BakedGIVisParams {
    float gridMinX, gridMinY, gridMinZ;
    float spacing;
    uint  dimX, dimY, dimZ;
    uint  probeCount;
    uint  visUAVIdx;        // RW output: probeCount x 324 (M1, M2) depth moments
    uint  metaUAVIdx;       // RW output: per-probe validity bits (DDGI classification
                            // + old-criteria diagnostic bit — see BakedGI.hlsli)
    // APPENDED: RW output for the sky-visibility SH9 A/B, 9 floats for each probe
    // (stride 4 bytes, probeCount * 9 elements). The visibility pass writes it
    // from the same rays that fill the moments, so it costs no extra traversal.
    // Measured with dxc: offset 40, sizeof 44 (11 dwords) — under the shared slot.
    uint  skyVisSHUAVIdx;
};

// Root constants for the per-term response-ball fold (debug view 8). One
// 64-thread group covers 64 probes; each thread reads ONE bake term's transport
// table and writes that probe's SH9 response ball. See Shaders/BakedGI_IterFold.hlsl.
//
// No grid geometry: the fold visits probes by flat index and reads the table at
// a fixed stride, so it needs neither the origin, the spacing nor the dims. The
// three fields below are the whole kernel input.
struct BakedGIIterFoldParams {
    uint probeCount;        // probes in the grid; threads at or past it return
    uint tableSRVIdx;       // SRV: the transport table of ONE bake term (probeCount x 16 x 10 float4)
    uint ballUAVIdx;        // RW: probeCount x 9 float4 SH9 — one response ball for each probe
};

// Root constants for one bake-time dilation dispatch. One thread handles one
// (probe, record) pair and copies or averages all 10 float4 values in that record.
// The sky dispatch writes meta through record zero. The sun dispatch reuses the
// same donor weights without writing meta. Every field is a 4-byte scalar.
struct BakedGIDilationParams {
    uint dimX, dimY, dimZ;
    uint probeCount;
    uint recordCount;       // 16 sky records or 17 sun records
    uint tableSRVIdx;       // raw final table on iteration zero, then dilated input
    uint tableUAVIdx;       // dilated destination only; never a raw transport table
    uint metaSRVIdx;
    uint metaUAVIdx;
    uint writeMeta;         // one for sky, zero for sun
};

// -----------------------------------------------------------------------------
// DDGI — root constants (compute slot 12 / b2) for
// the trace dispatch and the three resolve dispatches. Every field is a 4-byte
// scalar, so the C++ and the HLSL layouts agree with no padding.
//
// One block serves all four entry points; each entry point reads the fields it
// needs and ignores the rest. The bindless heap indices therefore change role
// between passes — see the note on each field.
//
// REQUIRES in shader scope: nothing. The shaders read the volume geometry from
// this block, not from FrameConstants, so the trace and the resolves stay
// independent of the consumer's fetch state.
// -----------------------------------------------------------------------------
struct DDGIParams {
    // Random rotation applied to the ROTATING ray subset (index >= 32). The 32
    // stable rays are unrotated so probe-state classification cannot flicker.
    float randomVectorX, randomVectorY, randomVectorZ;
    float randomAngle;
    // Temporal blend of the new sample into the atlas history. The value is the
    // HISTORY weight, so 0.98 keeps 98% of the previous frame. Frame 0 sends 0,
    // which makes the new sample replace the cleared history exactly.
    float historyBlendWeight;
    // Live sun — trace pass only. A frontface hit evaluates the sun at the hit
    // with a shadow ray (the BakedGI retrace-sun discipline), so the probe
    // stores indirect light only and the receiver's direct sun stays with the
    // runtime-exact path.
    float sunIrradianceX, sunIrradianceY, sunIrradianceZ; // perpendicular irradiance (RGB)
    float sunDirX, sunDirY, sunDirZ;                      // TO-SUN unit direction
    // Probe grid. The atlases pack the FLATTENED probe index (see
    // DDGI.hlsli), and the row length follows from probeCount alone.
    float gridMinX, gridMinY, gridMinZ;
    float spacing;
    uint  dimX, dimY, dimZ;         // probes per axis
    uint  numRaysPerProbe;          // rays cast for each probe
    uint  maxRaysPerProbe;          // ray hit buffer stride for each probe
    // Bindless heap indices. A resource that one pass reads and another pass
    // writes needs TWO indices: CreateStructuredBuffer / CreateTexture allocate
    // a separate descriptor for the SRV role and for the UAV role, and a
    // descriptor can only serve the role it was created for. (BakedGI splits
    // metaUAVIdx / metaSRVIdx the same way.)
    //   rayHitUAVIdx            trace: RWStructuredBuffer write
    //   rayHitSRVIdx            irradiance / depth resolve: read
    //   irradianceHistorySRVIdx trace feedback + irradiance resolve history: read
    //   irradianceTargetUAVIdx  irradiance resolve: RWTexture2D write
    //   depthHistorySRVIdx      depth resolve history: read
    //   depthTargetUAVIdx       depth resolve: RWTexture2D write
    //   probeOffsetSRVIdx       trace: read
    //   probeOffsetUAVIdx       probe-state resolve: RWStructuredBuffer write
    //   probeStateSRVIdx        trace + irradiance / depth resolve: read
    //   probeStateUAVIdx        probe-state resolve: RWStructuredBuffer write
    uint  rayHitUAVIdx;
    uint  rayHitSRVIdx;
    uint  irradianceHistorySRVIdx;
    uint  irradianceTargetUAVIdx;
    uint  depthHistorySRVIdx;
    uint  depthTargetUAVIdx;
    uint  probeOffsetSRVIdx;
    uint  probeOffsetUAVIdx;
    uint  probeStateSRVIdx;
    uint  probeStateUAVIdx;
};

struct BindlessIndices {
    uint InputIdx0;
    uint InputIdx1;
    uint InputIdx2;
    uint OutputIdx0;
    uint OutputIdx1;
    uint OutputIdx2;
    uint PathVizLineBufferIdx; // UAV (CS write) or SRV (VS read) index for path viz lines
};

// Path visualization line types (bits[3:0] of typeAndValid)
// bit[4] = valid flag
#define PATHVIZ_TYPE_PRIMARY    0u // yellow:  camera -> first surface
#define PATHVIZ_TYPE_BOUNCE1    1u // white:   first surface -> bounce 1
#define PATHVIZ_TYPE_BOUNCE2    2u // cyan:    bounce 1 -> bounce 2
#define PATHVIZ_TYPE_BOUNCE3    3u // blue:    bounce 2 -> bounce 3
#define PATHVIZ_TYPE_TEMPORAL   4u // orange:  center pixel -> temporal sample
#define PATHVIZ_TYPE_SPATIAL    5u // magenta: center pixel -> spatial winner
#define MAX_PATH_VIZ_LINES  16

struct PathVizLine {
    float3 start;
    uint typeAndValid; // bits[3:0] = type, bit[4] = valid flag
    float3 end;
    uint _pad;
};

struct LightConstants {
    ROW_MAJOR float4x4 viewProj;
    float4 position;
    float4 color;
    float4 direction;
    float intensity;
    float selectionPDF;
    uint padding[2];
};

// alphaMode values (shared CPU/GPU — must match cgltf alpha mode enum)
// 0 = Opaque, 1 = Mask (alpha-test), 2 = Blend (alpha-blended; rejected by meshlet pipeline)
#define ALPHA_MODE_OPAQUE 0u
#define ALPHA_MODE_MASK   1u
#define ALPHA_MODE_BLEND  2u

struct MaterialConstants {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureIndex;
    int normalTextureIndex;
    int metallicRoughnessTextureIndex;
    int alphaMode;      // ALPHA_MODE_OPAQUE / ALPHA_MODE_MASK / ALPHA_MODE_BLEND
    float alphaCutoff;
    uint _pad;
};

struct DrawNodeData {
    ROW_MAJOR float4x4 world;
    uint vertexOffset;
    uint indexOffset;
    uint materialID;
    uint padding;
};

struct Reservoir {
    float3 hitPos;
    float3 hitNormal;
    float3 radiance;
    float w_sum;
    float W;
    float M;
    float firstBounceHitT;
    // Bit 31 of historyAge encodes lobe type: 0 = diffuse, 1 = specular.
    // Bits [0-30] store the actual age. MAX_HISTORY_AGE (12) fits comfortably.
    uint historyAge;
};

// Use these helpers (defined in CommonTracing.hlsl) to access historyAge safely.
#define RESERVOIR_SPECULAR_FLAG 0x80000000u

// Per-pixel DI reservoir — stores the winning local light index from ReSTIR DI.
// 32 bytes total (8 x uint32).
struct DIRreservoir {
    float w_sum;              // Running sum of RIS weights
    float W;                  // Normalized unbiased weight W = w_sum / (M * p_hat_selected)
    float M;                  // Effective sample count (history length)
    float targetPdf;          // Target PDF p_hat of the currently selected light
    uint  selectedLightIndex; // Index into LightsBuffer of the winning light (bits[0-15])
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

// Per-pixel diffuse candidate data written by RTDGI temporal, read by RTR temporal
struct DiffuseCandidate {
    float3 hitPos;
    float  hitT;
    float3 hitNormal;
    float  _pad0;
    float3 radiance;
    float  _pad1;
};

struct SharcBindlessIndices {
    uint HashEntriesBufIdx;
    uint AccumulationBufIdx;
    uint ResolvedBufIdx;
};

struct IrCacheBindlessIndices {
    uint MetaBufIdx;
    uint PoolBufIdx;
    uint GridMetaBufIdx;
    uint EntryCellBufIdx;
    uint IrradianceBufIdx;
    uint LifeBufIdx;
    uint IndirectionBufIdx;
    uint TraceArgsBufIdx;
    uint PosBufIdx;
    uint RepropBufIdx;
    uint ReproposalCountBufIdx;
};

// =============================================================================
// Meshlet Pipeline Types (shared CPU/GPU)
// =============================================================================

// Maximum meshlet sizes (compile-time constants, must match bin cache embedding)
static const uint MESHLET_MAX_VERTICES  = 64;
static const uint MESHLET_MAX_TRIANGLES = 124;

struct Meshlet {
    uint VertexOffset;      // Offset into MeshletVertexBuffer (uint32 indices)
    uint TriangleOffset;    // Offset into MeshletTriangleBuffer (packed triangles)
    uint VertexCount;       // Number of unique vertices (max 64)
    uint TriangleCount;     // Number of triangles (max 124)
};

struct MeshletTriangle {
    uint V0 : 10;
    uint V1 : 10;
    uint V2 : 10;
    uint    : 2;  // padding
};

// Bounding SPHERE (not AABB) — chosen over AABB because a sphere transforms
// trivially under any affine LocalToWorld (center = mul(center,M), radius scaled
// by the matrix's max axis scale), whereas an AABB must be fully recomputed from
// its 8 corners on every transform. Computed once at load time via meshoptimizer's
// meshopt_computeMeshletBounds (tight cluster bounding sphere), consumed by
// FrustumCullMeshletSphere/HZBCull in MeshletCommon.hlsli / MeshletTwoPassCull.hlsl.
struct MeshletBounds {
    float3 LocalCenter;
    float  LocalRadius;
};

// Replaces DrawNodeData for the meshlet path
// Each *Offset field is an element offset into the corresponding global StructuredBuffer<T>
struct MeshData {
    uint PositionOffset;        // Element offset into GlobalPositions[]  (float3)
    uint NormalOffset;          // Element offset into GlobalNormals[]    (uint, RGB10A2_SNORM)
    uint UVOffset;              // Element offset into GlobalUVs[]        (uint, RG16_FLOAT)
    uint MeshletOffset;         // Element offset into GlobalMeshlets[]   (Meshlet)
    uint MeshletVertexOffset;   // Element offset into GlobalMeshletVertices[] (uint)
    uint MeshletTriangleOffset; // Element offset into GlobalMeshletTriangles[] (MeshletTriangle)
    uint MeshletBoundsOffset;   // Element offset into GlobalMeshletBounds[]   (MeshletBounds)
    uint MeshletCount;
    uint MaterialIndex;
    uint GlobalMeshletStart;   // Global meshlet index where this entry's meshlets begin (CPU prefix-sum)
    uint AlphaMode;            // ALPHA_MODE_OPAQUE / ALPHA_MODE_MASK / ALPHA_MODE_BLEND
    uint _pad;
};

struct InstanceData {
    ROW_MAJOR float4x4 LocalToWorld;
    uint MeshDataIndex;         // Index into global MeshData[] buffer
    uint BoundsIndex;           // Index into InstanceBounds[] — currently equal to MeshDataIndex
    uint _pad1;
    uint _pad2;
};

// LOCAL-space bounding SPHERE of a SubMesh — the enclosing sphere of the union
// of that SubMesh's meshlet bounding spheres (meshopt_computeSphereBounds over
// prim.meshletBounds), computed once on CPU in Model::BuildMeshlets().
// One entry per MeshData, built once on CPU in Model::CreateMeshletResources()
// Pass 1 (primitive iteration) — bounds are a GEOMETRY property, not an
// instance property, so shared meshes share one entry.
// Cull shaders transform this by the instance's PER-FRAME LocalToWorld
// (D3D12_Research pattern: local bounds × current matrix), so culling
// follows animated/moving instances correctly.
struct InstanceBounds {
    float3 BoundsCenter;   // Local-space center
    float  BoundsRadius;   // Local-space bounding-sphere radius
};

// Culling token — the unit of work through the cull pipeline
struct MeshletCandidate {
    uint InstanceID;    // Index into InstanceData[]
    uint MeshletIndex;  // Per-mesh meshlet index
};

// =============================================================================
// Meshlet Dispatch / Rasterize Params (shared CPU/GPU, passed via root constants b1)
// =============================================================================

// IMPORTANT: cbuffer-bound (root-constants) structs must stay FLAT — scalar fields only,
// no nested structs, no arrays. HLSL cbuffer packing 16-byte-aligns nested struct
// members (and array elements), which silently misaligns a plain C++ struct.

// Passed to the Mesh Shader rasterize pass via root param 12 (b1).
// With no binning, the mesh shader indexes VisibleMeshlets[Phase's base offset + SV_GroupID].
// Phase 2's base offset is VisibleMeshletsCounter[COUNTER_PHASE1_VISIBLE_MESHLETS] (Phase 1's
// final count) — see the counter layout comment on TWO_PASS_PHASE_FIRST/SECOND below.
struct RasterParams {
    uint VisibleMeshletsIdx;         // SRV index of VisibleMeshlets[] (for rasterize MS)
    uint DispatchMeshArgsIdx;        // UAV index of DispatchMeshArgs (for BuildDispatchMeshArgsCS)
    uint VisibleMeshletsCounterIdx;  // SRV index of VisibleMeshletsCounter[2] (for BuildDispatchMeshArgsCS and rasterize MS's Phase-2 base offset)
    uint Phase;                      // TWO_PASS_PHASE_FIRST / TWO_PASS_PHASE_SECOND — also doubles as the VisibleMeshletsCounter slot index for this phase
    // Bindless SRV heap indices of the 9 global meshlet stream buffers.
    // Each stream is looked up individually via ResourceDescriptorHeap[idx], so the streams
    // need no adjacent heap slots.
    uint GlobalPositionsSRVIdx;        // StructuredBuffer<float3>
    uint GlobalNormalsSRVIdx;          // StructuredBuffer<uint> (RGB10A2_SNORM)
    uint GlobalUVsSRVIdx;              // StructuredBuffer<uint> (RG16_FLOAT)
    uint GlobalMeshletsSRVIdx;         // StructuredBuffer<Meshlet>
    uint GlobalMeshletVerticesSRVIdx;  // StructuredBuffer<uint> (vertex indirection)
    uint GlobalMeshletTrianglesSRVIdx; // StructuredBuffer<MeshletTriangle>
    uint GlobalMeshletBoundsSRVIdx;    // StructuredBuffer<MeshletBounds>
    uint MeshDataSRVIdx;               // StructuredBuffer<MeshData>
    uint InstanceDataSRVIdx;           // StructuredBuffer<InstanceData>
};

// Meshlet visibility-buffer debug overlay parameters — root constants at main-root-signature
// param 13 (b2), shared slot with IrCacheBindlessIndices / HZBDebugParams etc. (debug-only
// PSOs, never bound simultaneously). Consumed by VisibilityDebugView.hlsl.
struct MeshletDebugParams {
    uint Mode;             // 0 = Off, 1 = Instance, 2 = Meshlet, 3 = Primitive, 4 = HZB mip tint
    uint VisBufSRVIdx;     // Descriptor-heap SRV index for visibility buffer (R32_UINT texture)
    uint CandidatesSRVIdx; // Descriptor-heap SRV index for VisibleMeshlets StructuredBuffer
    uint OutputUAVIdx;     // Descriptor-heap UAV index for output color (R16G16B16A16_FLOAT texture)
    uint Width;
    uint Height;
    uint MipsSRVIdx;       // Descriptor-heap SRV index for VisibleMeshletMips (mode 4)
    uint HZBMipCount;      // HZB mip count (mode 4 color ramp normalization)
    // Bindless SRV heap indices of the 9 global meshlet stream buffers (task012 Phase 1c).
    uint GlobalPositionsSRVIdx;        // StructuredBuffer<float3>
    uint GlobalNormalsSRVIdx;          // StructuredBuffer<uint> (RGB10A2_SNORM)
    uint GlobalUVsSRVIdx;              // StructuredBuffer<uint> (RG16_FLOAT)
    uint GlobalMeshletsSRVIdx;         // StructuredBuffer<Meshlet>
    uint GlobalMeshletVerticesSRVIdx;  // StructuredBuffer<uint> (vertex indirection)
    uint GlobalMeshletTrianglesSRVIdx; // StructuredBuffer<MeshletTriangle>
    uint GlobalMeshletBoundsSRVIdx;    // StructuredBuffer<MeshletBounds>
    uint MeshDataSRVIdx;               // StructuredBuffer<MeshData>
    uint InstanceDataSRVIdx;           // StructuredBuffer<InstanceData>
};

// =============================================================================
// HZB (Hierarchical Z-Buffer) construction
// Built via AMD FidelityFX SPD (fetched via CMake FetchContent, see CMakeLists.txt). Bound as a root CBV
// (b0) on MeshletPass's dedicated m_HZBRootSignature. All texture/buffer access
// inside HZB.hlsl goes through ResourceDescriptorHeap[idx] bindless indices —
// no members here are arrays, to avoid HLSL cbuffer array-packing (16 bytes per
// element) mismatching this struct's plain C++ layout. Per-mip HZB UAV indices
// instead live in a small StructuredBuffer<uint> (MipIndicesSRVIdx), which packs
// tightly on both sides.
struct HZBConstants {
    uint  DepthSRVIdx;       // SRV index of the source depth buffer (GBuffer.depth)
    uint  MipIndicesSRVIdx;  // SRV index of StructuredBuffer<uint> — one HZB mip UAV index per element
    uint  SpdCounterUAVIdx;  // UAV index of the SPD global atomic counter (RWStructuredBuffer<uint>[1])
    uint  NumMips;           // Number of HZB mip levels (<= 12, SPD's max)
    uint  NumWorkGroups;     // SpdSetup() output — total thread groups dispatched (X*Y)
    uint  WorkGroupOffsetX;  // SpdSetup() output — always 0 (HZB has no sub-rect offset)
    uint  WorkGroupOffsetY;
    uint  Width;             // HZB mip 0 width, in texels
    uint  Height;            // HZB mip 0 height, in texels
    float DimensionsInvX;    // 1 / Width  — used by HZBInitCS to compute the source-depth UV
    float DimensionsInvY;    // 1 / Height
    uint  _pad0;
};

// Debug visualization of the raw HZB mip chain (Sources/Shaders/HZBDebugView.hlsl),
// bound at the same root param 13 (b2) slot VisibilityDebugView.hlsl's DebugParams
// uses — reused across debug-only CS PSOs, never bound simultaneously.
struct HZBDebugParams {
    uint HZBSRVIdx;    // SRV index of the HZB texture (all mips)
    uint MipLevel;     // Which mip to visualize
    uint OutputUAVIdx; // UAV index of FullScreenDebugTex (R16G16B16A16_FLOAT)
    uint Width;        // Output (FullScreenDebugTex) width
    uint Height;       // Output (FullScreenDebugTex) height
};

// -----------------------------------------------------------------------------
// Occluded-rect debug recording (task007 mode 1) — Sources/Shaders/OccludedRectDebug.hlsl
//
// Cull shaders (MeshletTwoPassCull.hlsl) append one record per instance/meshlet
// rejected by HZBCull; the draw pass rasterizes the recorded NDC rects over a
// dimmed scene-albedo background into FullScreenDebugTex.
// -----------------------------------------------------------------------------
#define MAX_OCCLUDED_RECT_DEBUG 16384

struct OccludedRectDebug {
    float2 RectMinNDC;   // Clamped NDC rect (HZBCull step 2/4 output)
    float2 RectMaxNDC;
    float  NearestDepth; // Object side of the step-7 compare (reverse-Z: larger = closer)
    float  HZBDepth;     // HZB side of the compare
    uint   Mip;          // HZB mip the test used (step 5)
    uint   Phase;        // TWO_PASS_PHASE_* of the cull pass that occluded it
    uint   Kind;         // 0 = meshlet, 1 = instance
    uint   _pad0;
    float2 SampleMinNDC; // NDC rect spanned by the 4 HZB sample texels (step 5)
    float2 SampleMaxNDC; // — compare against RectMin/MaxNDC for the texel:object ratio
};

// Meshlet debug view mode that tints surviving meshlets by the HZB mip their
// occlusion test used (task007 mode 3). Matches the combo in Application::RenderImGui
// and ViewMode in VisibilityDebugView.hlsl. 0=Off, 1=Instance, 2=Meshlet, 3=Primitive.
#define MESHLET_DEBUG_MIP_TINT 4

// Draw-pass parameters — root constants at main-root-signature param 13 (b2),
// same slot as HZBDebugParams (debug-only PSOs, never bound simultaneously).
struct OccludedRectDrawParams {
    uint RectsSRVIdx;      // Bindless SRV of the OccludedRectDebug buffer
    uint RectsCountSRVIdx; // Bindless SRV of the record counter
    uint OutputUAVIdx;     // UAV index of FullScreenDebugTex
    uint Width;            // Output (FullScreenDebugTex) width
    uint Height;           // Output height
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// =============================================================================
// GPU on-screen debug text/lines — Sources/Shaders/DebugTextRender.hlsli/.hlsl
//
// Any shader appends packed text/line instances to one RWByteAddressBuffer via
// the producer API; at end of frame a 1-thread CS builds two indirect draw args
// (resetting the counters) and two ExecuteIndirect draws rasterize glyphs+lines
// onto the backbuffer. Port of D3D12_Research ShaderDebugRenderer, bindless variant.
// =============================================================================
#define DEBUG_TEXT_MAX_CHARS 8192
#define DEBUG_TEXT_MAX_LINES 32768

// RWByteAddressBuffer layout offsets (bytes) — must match DebugTextRenderer::Data
#define DEBUG_TEXT_COUNTER_OFFSET   0
#define DEBUG_LINE_COUNTER_OFFSET   4
#define DEBUG_TEXT_COUNTERS_SIZE    16
#define DEBUG_TEXT_INSTANCES_OFFSET DEBUG_TEXT_COUNTERS_SIZE
#define DEBUG_LINE_INSTANCES_OFFSET (DEBUG_TEXT_INSTANCES_OFFSET + DEBUG_TEXT_MAX_CHARS * 32)

// One glyph of the debug font atlas (built from an ImFontAtlas)
struct DebugGlyph {
    float2 MinUV;
    float2 MaxUV;
    float2 Dimensions;   // Quad size in pixels at scale 1
    float2 Offset;       // Offset from cursor to quad top-left
    float  AdvanceX;     // Cursor advance in pixels
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

// Unpacked instances (32 bytes each) — pack later if bandwidth matters
struct DebugCharInstance {
    float2 Position;     // Pixels, top-left of glyph quad
    uint   Character;    // ASCII codepoint
    float  Scale;        // Size multiplier (1 = native font size)
    float4 Color;
};

struct DebugLineInstance {
    float3 A;
    uint   ColorA;       // RGBA8; LSB of ColorA = screen-space ([0,1]) flag
    float3 B;
    uint   ColorB;       // RGBA8; world-space endpoints need FrameCB in the line VS
};

// End-of-frame pass parameters — root constants b1 (main root signature param 12,
// shared by the args-builder CS and the glyph/line raster PSOs)
struct DebugTextRenderParams {
    uint  DataSRVIdx;      // ByteAddressBuffer SRV (instances)
    uint  DataUAVIdx;      // RWByteAddressBuffer UAV (args builder resets counters)
    uint  ArgsUAVIdx;      // RWStructuredBuffer<uint4> indirect draw args (2 entries)
    uint  GlyphSRVIdx;     // StructuredBuffer<DebugGlyph>
    uint  FontAtlasSRVIdx; // Texture2D<float4> font atlas
    float TargetWidth;     // Backbuffer width in pixels
    float TargetHeight;    // Backbuffer height in pixels
    uint  _pad0;
};

// Depth-readout producer (mode 5) parameters — root constants b2 (param 13)
struct DepthReadoutParams {
    uint  RectsSRVIdx;      // Bindless SRV of the OccludedRectDebug buffer
    uint  RectsCountSRVIdx; // Bindless SRV of the record counter
    uint  DataUAVIdx;       // Debug text render-data UAV
    uint  GlyphSRVIdx;      // StructuredBuffer<DebugGlyph>
    float FontSize;         // Native font size in pixels
    float BackbufferWidth;
    float BackbufferHeight;
    uint  MaxLabels;        // Cap on labels emitted per frame
};

// =============================================================================
// Two-Pass Occlusion Culling — constants shared by CullInstancesCS / CullMeshletsCS
// =============================================================================

// Phase selectors (passed as root CBV fields, not shader defines).
// These double as the slot index into VisibleMeshletsCounter[2] (RWStructuredBuffer<uint>[2]):
// slot 0 = Phase 1's own visible-meshlet count, slot 1 = Phase 2's own count. Only cleared
// once per frame (gated to Phase 1 in GPUCulling::CullTwoPass) — Phase 2 must NOT reset slot 0,
// since Phase 2 appends its own candidates into VisibleMeshlets STARTING at slot 0's value
// (elementOffset += VisibleMeshletsCounter[TWO_PASS_PHASE_FIRST]) rather than aliasing over
// Phase 1's already-rasterized range. A single shared counter corrupts the deferred
// Visibility Buffer resolve for Phase-1 pixels. See docs/bug_flyingworld_meshlet_flicker.md.
#define TWO_PASS_PHASE_FIRST   0u
#define TWO_PASS_PHASE_SECOND  1u

struct TwoPassCullConstants {
    uint  NumInstances;              // Total instances in InstanceData[]
    uint  NumMeshlets;               // Total meshlets across all instances
    uint  HZBSRVIdx;                 // Bindless SRV index of the HZB texture (all mips)
    uint  HZBMipCount;               // Number of HZB mip levels
    uint  HZBWidth;                  // HZB mip 0 width
    uint  HZBHeight;                 // HZB mip 0 height
    uint  CandidateMeshletsCounterIdx; // UAV bindless index of CandidateMeshletsCounter
    uint  CandidateMeshletsUAVIdx;     // UAV bindless index of CandidateMeshlets[]
    uint  OccludedInstancesCounterIdx; // UAV bindless index of OccludedInstancesCounter
    uint  OccludedInstancesUAVIdx;     // UAV bindless index of OccludedInstances[] (write)
    uint  OccludedInstancesSRVIdx;     // Bindless SRV index of OccludedInstances[] (read by Phase 2)
    uint  VisibleMeshletsUAVIdx;       // UAV bindless index of VisibleMeshlets[]
    uint  VisibleMeshletsCounterUAVIdx;// UAV bindless index of VisibleMeshletsCounter
    uint  Phase;                       // TWO_PASS_PHASE_FIRST / SECOND / FRUSTUM_ONLY
    uint  EnableOcclusion;             // 0 = frustum-only, 1 = occlusion culling active
    // Occluded-rect debug recording (task007 mode 1). When DebugRecordOccluded is 0
    // the cull shaders never touch the debug buffers (zero overhead).
    uint  DebugRecordOccluded;         // 1 = append OccludedRectDebug records for HZB-rejected candidates
    uint  OccludedRectsUAVIdx;         // UAV bindless index of OccludedRectDebug[]
    uint  OccludedRectsCounterUAVIdx;  // UAV bindless index of the record counter

    // Mip-selection tint (task007 mode 3): one mip value per visible-meshlet slot,
    // read back by the debug overlay via the vis token's candidateIndex.
    uint  DebugRecordMip;              // 1 = write the occlusion test's mip per surviving meshlet
    uint  VisibleMeshletMipsUAVIdx;    // UAV bindless index of VisibleMeshletMips[]

    // Fully-bindless resource indices — MeshletTwoPassCull.hlsl has no root SRV/UAV
    // descriptors at all (root signature is just b0+b1+a static sampler); every buffer
    // is looked up via ResourceDescriptorHeap[idx] instead. This also sidesteps having
    // to allocate a dedicated root SRV slot for OccludedInstances, which is UAV-written
    // in Phase 1 but SRV-read in Phase 2 — a role reversal one fixed root slot cannot
    // serve. OccludedInstancesSRVIdx above is that path.
    uint  InstanceDataSRVIdx;          // SRV bindless index of Model's InstanceData[]
    uint  InstanceBoundsSRVIdx;        // SRV bindless index of Model's InstanceBounds[]
    uint  MeshDataSRVIdx;              // SRV bindless index of Model's MeshData[]
    uint  MeshletBoundsSRVIdx;         // SRV bindless index of Model's GlobalMeshletBounds[]
    uint  MeshletCullArgsUAVIdx;       // UAV bindless index of MeshletCullArgs (uint3 dispatch args)
    uint  InstanceCullArgsUAVIdx;      // UAV bindless index of InstanceCullArgs (uint3 dispatch args)
    uint  VisibleInstancesCounterUAVIdx; // UAV bindless index of VisibleInstancesCounter
};

// =============================================================================
// CullStats — per-phase culling counters copied from the functional counters
// into a dedicated debug buffer after each phase completes. Consumed by
// CullStatsCS for on-screen overlay.
// =============================================================================

// CullStatsBuffer layout (RWStructuredBuffer<uint>[16]):
// All P2 slots are aligned to CopyCullStatsCS's P2 BaseSlot=4 write offsets:
// P2 writes candidate@[4], visible@[5], occluded@[6], instance-visible@[7].
#define CULL_STATS_P1_CANDIDATE_MESHLETS   0  // CandidateMeshletsCounter after Phase 1
#define CULL_STATS_P1_VISIBLE_MESHLETS     1  // VisibleMeshletsCounter after Phase 1
#define CULL_STATS_P1_OCCLUDED_INSTANCES   2  // OccludedInstancesCounter (deferred) after Phase 1
#define CULL_STATS_P1_VISIBLE_INSTANCES    3  // VisibleInstancesCounter after Phase 1 — instances that passed frustum+HZB
#define CULL_STATS_P2_CANDIDATE_MESHLETS   4  // CandidateMeshletsCounter after Phase 2
#define CULL_STATS_P2_VISIBLE_MESHLETS     5  // VisibleMeshletsCounter after Phase 2
#define CULL_STATS_P2_OCCLUDED_INSTANCES   6  // OccludedInstancesCounter (P1 value preserved into P2)
#define CULL_STATS_P2_VISIBLE_INSTANCES    7  // VisibleInstancesCounter after Phase 2 — instances newly visible in P2
#define CULL_STATS_COUNT                   8

// Passed to CopyCullStatsCS via root constants.
// For P1 (BaseSlot=0): copies Candidate[0]->Stats[0], Visible[TWO_PASS_PHASE_FIRST]->Stats[1],
//   Occluded[0]->Stats[2], InstanceVisible[0]->Stats[3].
// For P2 (BaseSlot=4): copies Candidate[0]->Stats[4], Visible[TWO_PASS_PHASE_SECOND]->Stats[5],
//   Occluded[0]->Stats[6], InstanceVisible[0]->Stats[7].
// VisibleCounterSRVIdx now points at the 2-slot VisibleMeshletsCounter (see
// TWO_PASS_PHASE_FIRST/SECOND above) — CopyCullStatsCS picks the phase-appropriate slot
// from BaseSlot instead of always reading slot 0, since slot 0 (Phase 1's count) is no
// longer reset/overwritten before Phase 2 runs.
struct CullStatsCopyParams {
    uint CandidateCounterSRVIdx;          // SRV index of CandidateMeshletsCounter
    uint VisibleCounterSRVIdx;            // SRV index of VisibleMeshletsCounter
    uint OccludedCounterSRVIdx;           // SRV index of OccludedInstancesCounter
    uint StatsBufferUAVIdx;               // UAV index of CullStatsBuffer
    uint BaseSlot;                        // 0 for Phase 1, 4 for Phase 2
    uint InstanceVisibleCounterSRVIdx;    // SRV index of VisibleInstancesCounter (per-phase instance count)
    uint _pad[2];
};

// Passed to CullStatsCS via root constants — reads the debug stats buffer and
// renders an on-screen table via the GPU debug-text system.
struct CullStatsParams {
    uint StatsBufferSRVIdx;          // SRV heap index of CullStatsBuffer
    uint DataUAVIdx;                 // Debug text render-data UAV index
    uint GlyphSRVIdx;                // Debug glyph atlas SRV index
    float FontSize;                  // Native font line height in pixels
    uint TotalInstances;             // CPU-provided: total instances in scene
    uint TotalMeshlets;              // CPU-provided: total meshlets across all instances
    uint BackbufferWidth;
    uint BackbufferHeight;
    float StartX;                    // Top-left pixel X
    float StartY;                    // Top-left pixel Y
};

#endif // SHARED_TYPES_H
