#pragma once

#include "Graphics/GraphicsTypes.h"

#include <vector>

class Model;

// =============================================================================
// BakedGI — probe-based baked GI: uniform probe grid + runtime transport bake.
//
// Exclusive indirect-GI source (either this or ReSTIR GI, never both). The
// bake computes, per probe and per delta direction (16 fibonacci-hemisphere
// dirs), the INDIRECT-ONLY directional-irradiance SH9 response to a unit
// delta light plus the direct-delta visibility V_i (multi-bounce via a
// series-expansion ping-pong with exact direct light at every trace hit —
// the retrace-sun discipline). Per frame, a cheap update pass folds the live
// sky (cubemap quadrature) and sun (direction-kernel interpolation) into a
// lit-probe buffer. The deferred pass fetches it with a DDGI-style weighted
// blend: trilinear x backface x Chebyshev occlusion from per-probe octahedral
// depth-moment maps (step 3), with the unified self-shadow bias. See
// docs/task017-baked-gi-probe-system-design.md.
// =============================================================================
class BakedGI
{
public:
    static constexpr uint32_t kDirections      = 16; // delta/sky-quadrature directions
    static constexpr uint32_t kResponseFloat4s = 10; // per (probe, dir): 9 SH9 RGB + V_i

    // ---- Equinox hour states: the sun direction set ----
    //
    // The sun is one number now: the hour of TRUE SOLAR TIME, where hour 12 is
    // solar noon. The date is fixed at the equinox, so the sun path is a great
    // circle and the arc above the horizon is exactly 180 degrees. Seventeen
    // states cover that arc at 0.75 h (11.25 degree) intervals.
    //
    // An ODD count is required. State 8 is noon. With an even count no state falls
    // at noon, so the scene's authored sun becomes a blend of two bakes instead of
    // one.
    //
    // MUST match BAKED_GI_SUN_STATES and the two spacing constants in
    // BakedGI.hlsli.
    static constexpr uint32_t kSunStates       = 17;
    static constexpr float    kSunStateHour0   = 6.0f;  // state 0 sits at sunrise
    static constexpr float    kSunStateStepHours = 0.75f; // 17 states over 12 hours

    // The scene's authored sun, VERTICAL COMPONENT. This is the source of truth
    // for the latitude, not a transcribed number: at the equinox the noon
    // elevation is 90 - latitude, so this cosine IS cos(latitude), and
    // SunDirectionAtHour derives the latitude from it with acos.
    //
    // Kept in DOUBLE on purpose. acos(0.906106471) = 25.0272791670 degrees, and
    // rounding that to the six decimals the design note prints (25.027279) moves
    // state 8 by 3.5e-9 — above the 1e-9 the authored sun is held to. The double
    // keeps the round trip exact: the state-8 direction is bit-identical to the
    // authored float3.
    static constexpr double   kEquinoxSunY     = 0.906106471;

    // The same latitude in degrees, for display and for the tooltips.
    // Informational only: no math reads this, because the rounded value is not
    // accurate enough to reproduce the authored sun. See kEquinoxSunY.
    static constexpr double   kEquinoxLatitudeDeg = 25.027279;

    static constexpr uint32_t kLitFloat4s     = 9;  // per probe: 9 SH9 RGB
    static constexpr uint32_t kBakeIterations  = 3;  // series-expansion bounces
    static constexpr uint32_t kMaxProbeCount   = 128 * 1024;
    static constexpr uint32_t kMaxDim          = 256; // per-axis grid cap
    static constexpr uint32_t kVisStride       = 18 * 18; // padded (M1, M2) moment texels per probe
    static constexpr uint32_t kVisDispatchX    = 4096;    // visibility-bake dispatch X tile
                                                           // (must match BAKED_GI_VIS_DISPATCH_X)

    // ---- Per-bake-term capture (debug view "Efb Contribution") ----
    // A response ball is a per-probe SH9, so it has the LIT buffer's shape and
    // reuses kLitFloat4s. The capture stores one ball for EACH term
    // (kBakeIterations) but only kBakeIterations - 1 extra transport tables: the
    // LAST term's table IS m_Response[m_FinalResponse], so it is never duplicated.
    static constexpr uint32_t kEfbBallFloat4s    = kLitFloat4s;
    static constexpr uint32_t kIterSnapshotCount = kBakeIterations - 1;
    // "No response ball" sentinel for bakedGIEfbIterSRVIndex and
    // bakedGIEfbBaseSRVIndex. 0 is a legal descriptor index, so it cannot mean
    // "none": the debug view early-outs on this value. The shader spells it
    // BAKED_GI_EFB_NO_SRV (BakedGI_Debug.hlsl) and its comment names this one.
    static constexpr uint32_t kEfbBallNoSRV = 0xFFFFFFFFu;
    // "Fold all 16 directions" for FrameConstants::bakedGIEfbDir. The view's
    // single-direction mode publishes an index in 0..kDirections-1 instead, so the
    // shader's test is "is this a direction index?" and every other value — this
    // one included — means the folded ball. Deliberately NOT 0: direction 0 is a
    // real direction, and a default of 0 would silently un-average it.
    static constexpr uint32_t kEfbDirAll = 0xFFFFFFFFu;

    // ---- RNG seed index (BakedGIBakeParams::seedDir) ----
    // The bake's RNG seed depends on an index, so two passes over the same probe
    // with different indices gather DIFFERENT directions. The sun pass now seeds
    // from the hour STATE, so two adjacent states never share rays: sharing them
    // would make the blend inherit a correlated error instead of averaging two
    // independent draws.
    //
    // This value means "use the thread's own direction", which is what the binned
    // sky transport uses. The shaders spell it with a literal, because they cannot
    // see this file.
    static constexpr uint32_t kSunSeedOwnDir = 0xFFFFFFFFu;

    // ---- Ray-debug tool (debug only, on demand) ----
    // Rays the BAKE traces for each direction. 64 -> 256 on 2026-10-26.
    //
    // Reason: at 64, one bin of probe 7669 held 74.7 percent of the sun red DC,
    // and that bin's whole direct term was ONE gather ray. A 64-ray estimate of
    // the full 4*PI sphere cannot resolve a distant source: the ray stands for
    // 4*PI/64 = 0.196 sr, and the cloth subtends far less. Measured: removing
    // that single ray dropped the probe's sun red from 148.9 to about 79.1.
    //
    // Cost is bake time only. The payload is the same size and the update pass
    // reads the same bytes, so neither memory nor frame time changes.
    // MUST match GATHER_RAYS in BakedGI_Bake.hlsl and BAKE_RAYS in
    // BakedGI_RayDebug.hlsl.
    static constexpr uint32_t kBakeGatherRays = 256;
    // Ray RECORD slots written for each direction, and the Ray Fan's instance
    // count. Deliberately NOT kBakeGatherRays: this value sizes the debug
    // buffer, and the view cannot show 256 boxes legibly anyway. The debug pass
    // replays all kBakeGatherRays and writes only the ones that matter.
    static constexpr uint32_t kDebugGatherRays = 64;
    // float4s for each ray record. 4 -> 6 on 2026-10-14: rec4 and rec5 add the
    // hit's IDENTITY (normal, world position, instance index, primitive index),
    // which this layout used to discard — so naming the surface behind a coloured
    // ray meant reproducing EvalResponseIndirect offline and inferring the normal
    // from NdotD alone. rec0..rec3 keep their meaning AND their offsets; only the
    // STRIDE changed. One stride, three files: this one, RAY_RECORD_F4
    // (BakedGI_RayDebug.hlsl, the writer) and BAKED_GI_RAYFAN_REC_F4
    // (BakedGI_Debug.hlsl, the viewer).
    static constexpr uint32_t kRayRecordF4   = 6;   // float4s for each ray record (96 B)
    // The re-accumulated table sits after the records in the same buffer, so one
    // UAV serves both outputs. Layout: [records][table], both float4.
    // The record stride is the DEBUG count: keeping the buffer at its old size
    // is what stops this change from adding 302,592 B of readback.
    static constexpr uint32_t kRayDebugTableBase = kDirections * kDebugGatherRays * kRayRecordF4;
    static constexpr uint32_t kRayDebugBufF4s    = kRayDebugTableBase + kDirections * kResponseFloat4s;
    static constexpr uint64_t kRayDebugBufBytes  = (uint64_t)kRayDebugBufF4s * 16;
    static constexpr uint64_t kRayDebugTableBytes = (uint64_t)kDirections * kResponseFloat4s * 16;
    static constexpr uint64_t kRayDebugStoredOff = kRayDebugBufBytes; // stored table copy follows the debug buffer
    static constexpr uint64_t kRayDebugReadbackBytes = kRayDebugBufBytes + kRayDebugTableBytes;

    // ---- Cage Trace (bakedGIDebugView == 9) ----
    // The view's geometry is a FIXED set of thin boxes, so the instance count is
    // a constant: 1 probe->hit segment + 8 hit->cage-probe segments + 8 cage
    // markers + 2 endpoint markers (probe, hit) + 12 cell edges. Must match
    // BAKED_GI_CAGE_INSTANCES (BakedGI_Debug.hlsl, which owns the layout).
    static constexpr uint32_t kCageInstances = 31u;
    // Vertices for each instance: kCubeTris in BakedGI_Debug.hlsl (12 triangles).
    static constexpr uint32_t kCageVertsPerInstance = 36u;

    void CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature);

    // Records the full bake (grid setup, buffer (re)creation, series
    // iterations) into cmdList. The caller closes/executes/waits — the
    // Renderer::BakeGIProbes wrapper follows the BuildAccelerationStructures
    // reset/record/ExecuteCommandList pattern, so buffers are never in flight
    // when released.
    bool RecordBake(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                    ID3D12RootSignature* rootSignature, Model* model,
                    D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                    D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                    float spacing,
                    // Method 3: also bake the sun's response for every hour state,
                    // into the per-probe sun table. The directions come from
                    // SunDirectionAtHourCPU, not from a caller argument: the sun
                    // is the hour, so a live direction must not reach the bake.
                    bool singleDirectionSun = false);

    // Per-frame: fold the live sun/sky state into the lit-probe buffer.
    // hour selects the two baked sun states that bracket it. The sky term reads
    // the 16-record quadrature table and does not depend on the hour.
    void RecordUpdate(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                      D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                      const DirectX::XMFLOAT3& sunIrradiance, float hour);

    bool              IsValid() const { return m_Valid; }
    float             GetSpacing() const { return m_Spacing; }
    uint32_t          GetProbeCount() const { return m_ProbeCount; }
    // Bumped every time RecordBake (re)establishes the grid — dims, spacing or
    // gridMin. The UI's nearest-probe list keys its cache on this, so a re-bake
    // with a different grid can never leave a list built against the old one.
    // A counter, not a float compare: the grid identity is discrete.
    uint32_t          GetGridRevision() const { return m_GridRevision; }
    DirectX::XMFLOAT3 GetGridMin() const { return m_GridMin; }
    uint32_t          GetDimX() const { return m_DimX; }
    uint32_t          GetDimY() const { return m_DimY; }
    uint32_t          GetDimZ() const { return m_DimZ; }
    uint32_t          GetLitSRVIndex() const { return (uint32_t)m_LitProbes.srvIndex; }
    uint32_t          GetLitSkySRVIndex() const { return (uint32_t)m_LitProbesSky.srvIndex; }
    uint32_t          GetLitSunSRVIndex() const { return (uint32_t)m_LitProbesSun.srvIndex; }
    uint32_t          GetMetaSRVIndex() const { return (uint32_t)m_Meta.srvIndex; }
    uint32_t          GetVisibilitySRVIndex() const { return (uint32_t)m_Visibility.srvIndex; }
    uint32_t          GetResponseSRVIndex() const { return (uint32_t)m_Response[m_FinalResponse].srvIndex; }

    // Per-probe SUN response table (method 3 only): kSunStates x kResponseFloat4s
    // float4 for each probe, indexed by hour state. Populated by RecordBake when
    // singleDirectionSun is set. Separate from the 16-bin table on purpose — see
    // BakedGIBakeParams::sunMode. Returns 0 when no sun table exists.
    uint32_t          GetSunResponseSRVIndex() const
    {
        return m_SunResponseValid ? (uint32_t)m_SunResponse[m_FinalSunResponse].srvIndex : 0u;
    }
    bool              HasSunResponse() const { return m_SunResponseValid; }
    // True only when the resident sun table can serve THIS hour.
    //
    // The table holds every hour state, so the only way it can fail to serve an
    // hour is that no bake has produced it. The direction test that used to live
    // here is gone: the state directions are fixed by the equinox formula and are
    // unit length by construction, so no live sun vector can invalidate them.
    //
    // Kept as a query rather than a bare HasSunResponse so the callers that ask
    // "is the sun term available for the frame I am about to draw" keep one
    // expression, and a future partial-coverage table has a place to say so.
    bool              HasSunResponseFor(float hour) const;
    // Ray-debug buffer (records + re-accumulated table). The Ray Fan view reads
    // the records. The buffer exists only after a dump.
    bool              HasRayDebugBuffer() const { return m_RayDebugBuf.resource != nullptr; }
    uint32_t          GetRayDebugSRVIndex() const { return (uint32_t)m_RayDebugBuf.srvIndex; }
    // The probe the last dump recorded, already clamped into the grid. The fan
    // takes its probe from here, not from the UI field, so the fan and the text
    // always describe one probe.
    uint32_t          GetRayDebugProbe() const { return m_RayDebugProbe; }
    // True only when the meta bit was READ and the dump's probe is invalid, so
    // the Ray Fan dims the L, Efb and regime channels. The verdict is bound to
    // the probe it describes, and RecordRayDebug clears it on every dump, so it
    // can never describe another probe or a stale verdict. A valid probe and an
    // unknown one both return false: dimming claims that the probe is degenerate.
    bool              GetRayDebugKnownInvalid() const
    {
        return (m_RayDebugValidProbe == m_RayDebugProbe) && !m_RayDebugValid;
    }
    // The dump leaves the buffer in UNORDERED_ACCESS, so the read side moves it
    // to a shader-read state before the fan draws. Call this before that draw.
    void              TransitionRayDebugForRead(ID3D12GraphicsCommandList* cmdList);

    // ---- Cage Trace gather-ray picker (bakedGIDebugView == 9) ----
    //
    // ONE entry for each gather ray of a direction whose DIRECT term loses to the
    // feedback at its hit — "direct < Efb". That set IS the pool the view's ray is
    // chosen from, and this list is what the UI combo enumerates and publishes
    // into FrameConstants::bakedGIRayDebugRay. Entry 0 of the score ranking is
    // also what the shader picks on its own.
    //
    // It exists on the CPU at all because a dump already copies the records to a
    // readback buffer, so the numbers the combo shows are the trace's own numbers
    // rather than a second, drifting computation. Nothing here is recomputed for
    // the picture: the shader reads the records straight from its SRV, and a
    // disagreement between the two would be a defect, not a rounding.
    //
    // The pool covers ALL kDirections, not just the direction of the moment, so
    // moving the sun after a dump cannot leave the combo describing a direction
    // the trace no longer draws.
    struct RayCandidate
    {
        uint32_t ray;       // the RECORD index inside that direction's fan (0..kDebugGatherRays-1)
        uint32_t sourceRay; // the source gather ray it was taken from (0..kBakeGatherRays-1)
        float    score;     // luminance(diffAlbedo * Efb) — the shader's own ranking key
        float    efbLum;    // the feedback term's luminance at the hit
        float    directLum; // the direct term's luminance at the hit, always < efbLum here
        float    albedoLum; // the hit's diffuse albedo luminance — the ray that carries a
                            // large Efb into a dark surface contributes almost nothing
    };

    // "No override" for FrameConstants::bakedGIRayDebugRay: the shader keeps its
    // own argmax. 0xFFFFFFFF, because a ray index of 0 is a real ray.
    static constexpr uint32_t kNoRayPick = 0xFFFFFFFFu;

    // The pool of one delta direction, or an empty list when no dump has been
    // read out yet (or the direction is out of range). Ascending by RAY INDEX,
    // not by score: the combo's rows must stay put while the user steps through
    // the pool, and a score order would re-sort under the cursor on the next
    // dump. The dump's printed table is the score-ordered one.
    const std::vector<RayCandidate>& GetRayCandidates(uint32_t dir) const;
    // 0 until a dump's readout has filled the pool. The UI reports "no dump" on
    // this rather than on an empty direction, which is a different fact: an empty
    // pool means the direction really has no feedback-dominated ray.
    uint32_t          GetRayCandidateDirs() const { return m_RayCandidateDirs; }

    // ---------------------------------------------------------------------
    // Per-bake-term response balls — the "Efb Contribution" view.
    //
    // The bake snapshots every term's transport table and folds each one into a
    // per-probe SH9 ball (BakedGI_IterFold.hlsl). The view shows
    // ball[selected] - ball[0]: since the gather is linear in Efb and Efb never
    // changes a ray direction, the shadow-ray gate or an alpha-mask draw, the
    // term-k table is exactly R^0 + G(Efb_k), so that difference IS the
    // contribution of the series-feedback term by term k. Term 0 has feedback
    // gated off (BakedGI_Bake.hlsl:236), so selecting it subtracts a ball from
    // itself and the view is black — by construction, not by failure.
    //
    // FRESHNESS. The balls are written by the bake and describe the grid it ran
    // on. The Probes/ cache path restores a single final table instead of
    // dispatching, so it never folds: HasEfbBalls() is then false and the view
    // declines to draw, rather than differencing two absent balls.
    // ---------------------------------------------------------------------
    bool     HasEfbBalls() const { return m_EfbBallsValid; }
    // term is 0 .. kBakeIterations-1 (the UI's "Iteration" 1..3 minus 1).
    // Returns kEfbBallNoSRV when no bake has folded the terms, or when term is
    // at or past kBakeIterations: the view then has nothing to read, and the
    // sentinel is how it knows.
    uint32_t GetEfbBallSRVIndex(uint32_t term) const
    {
        return (m_EfbBallsValid && term < kBakeIterations)
             ? (uint32_t)m_EfbBall[term].srvIndex : kEfbBallNoSRV;
    }
    // The fold leaves the balls in UNORDERED_ACCESS. Call this before the view
    // draws, so the vertex shader reads a shader-read state (the same placement
    // rule as TransitionRayDebugForRead: the read site owns its read barrier,
    // and GPUResource::Transition makes it a no-op on every later frame).
    // It also moves the two TRANSPORT tables the single-direction mode reads.
    void     TransitionEfbBallsForRead(ID3D12GraphicsCommandList* cmdList);

    // The transport TABLE of a bake term — the fold pass's own input, exposed so
    // the "un-averaged" mode of the Efb view can read ONE direction without
    // folding (a single direction cannot be recovered from a 16-direction sum).
    // term is 0 .. kBakeIterations-1 (the UI's "Iteration" 1..3 minus 1). The LAST
    // term's table IS the live response buffer; the earlier ones are the bake's
    // snapshots. Same sentinel and the same reason as GetEfbBallSRVIndex: no bake
    // has captured the terms, so the view must not index a descriptor that does
    // not exist. On the Probes/ cache path the snapshots are never created, and
    // m_EfbBallsValid stays false, so this returns the sentinel there too.
    uint32_t GetEfbTermSRVIndex(uint32_t term) const
    {
        if (!m_EfbBallsValid || term >= kBakeIterations)
            return kEfbBallNoSRV;
        return (term + 1u < kBakeIterations)
             ? (uint32_t)m_ResponseIter[term].srvIndex
             : (uint32_t)m_Response[m_FinalResponse].srvIndex;
    }

    ID3D12PipelineState* GetDebugPSO() const { return m_DebugPSO.Get(); }
    ID3D12PipelineState* GetLeakDebugPSO() const { return m_LeakDebugPSO.Get(); }

    // Post-bake stats logging. Scene bounds / grid dims are logged inside
    // RecordBake; these two cover the GPU-side probe validity:
    //   1. RecordMetaReadback — appends a meta -> readback copy to the bake
    //      command list (call after RecordBake, before executing).
    //   2. LogBakeStats — maps the readback (GPU idle) and logs the valid /
    //      invalid probe coverage.
    bool RecordMetaReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void LogBakeStats() const;

    // ---------------------------------------------------------------------
    // Ray-debug tool. DEBUG ONLY, on demand, off the normal frame path.
    //
    // RecordRayDebug re-traces the 64 gather rays of ONE probe, for all 16
    // directions, and appends a copy of that probe's stored table. It also
    // appends the two copies to a readback buffer. The caller executes the
    // list and waits (the BakeGIProbes pattern), then calls
    // LogRayDebugReadout on the idle GPU.
    //
    // LogRayDebugReadout ranks the 16 directions by stored DC luminance and the
    // 64 rays of that winner. Under method 3 it ranks the replayed sun state's
    // rays and names no bin winner, because the 16 bins carry no sun weight.
    // It also runs the self-check that compares the re-accumulated table with
    // the stored table.
    // ---------------------------------------------------------------------
    bool RecordRayDebug(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                        ID3D12RootSignature* rootSignature,
                        D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                        D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                        D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                        D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                        D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                        D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                        uint32_t probeIndex,
                        // Method 3: replay ONE sun hour state instead of the 16
                        // bins. sunState selects the state, and the direction and
                        // the RNG seed both follow from it, exactly as the bake
                        // derives them. A different state here replays a
                        // different bake and the self-check reports MISMATCH —
                        // which is a true report, not a defect to route around.
                        bool singleDirectionSun = false,
                        uint32_t sunState = 8u);
    // Names the sun state the fan draws. Pass nullptr for a scene with no sun.
    // Do not add a default value: a forgotten argument would print "no sun" while
    // the fan draws a real one, and the log and the picture would disagree.
    void LogRayDebugReadout(const DirectX::XMFLOAT3* sunToDir) const;

    // Ray Fan colour channel that swaps the geometry: it draws the 16 fibonacci
    // sky-quadrature directions of the fan's probe, plus one marker box for the
    // sun's direction at the current hour, instead of the 64 gather rays. It
    // reads no record, so it needs no dump.
    // Must match BAKED_GI_RAYFAN_CH_DELTADIRS (BakedGI_Debug.hlsl) and the
    // "DeltaDirections" entry of the Ray Fan Channel combo (Application.cpp).
    // One value with three consumers is why it is named here rather than written
    // as 7u three times.
    static constexpr uint32_t kRayFanChannelDeltaDirs = 7u;

    // Instances the DeltaDirections channel draws: kDirections bin boxes plus ONE
    // marker box for the real sun (FrameConstants::bakedGISunDir). The 17th
    // instance is a MARKER, not a 17th basis direction — kDirections stays 16,
    // because its value divides the fibonacci lattice, so raising it would move
    // every existing bin.
    //
    // BAKED_GI_RAYFAN_DELTADIRS_INSTANCES (BakedGI_Debug.hlsl, which owns the
    // instance layout) is the authority and this is its mirror. The renderer
    // takes its draw count from here, the shader takes its instance layout from
    // there, and both derive the count from the same basis count, so the draw
    // call cannot disagree with the geometry. Same shape as kCageInstances.
    static constexpr uint32_t kRayFanDeltaDirInstances = kDirections + 1u;

    // ---------------------------------------------------------------------
    // Sun fold: the ONE definition of the hour -> two-state blend.
    //
    // The shader owns the formula (BakedGISunDirectionAtHour, BakedGI.hlsli) and
    // BakedGI_Update.hlsl owns the fold. These two functions are the CPU mirror,
    // and every CPU consumer goes through them: the bake's state directions, the
    // panel's table, and the frame constants the debug views read.
    //
    // A drift between the shader and this mirror puts the bake and the fold on
    // different arcs, and nothing reports it: the image stays plausible. The
    // known-good reference values sit beside the definitions.
    // ---------------------------------------------------------------------

    // The equinox sun direction at an hour of TRUE SOLAR TIME. Unit length by
    // construction, so there is no normalize and no NaN case.
    // Reference, hour 12 at kEquinoxLatitudeDeg: (0, 0.906106471, 0.423049718).
    static DirectX::XMFLOAT3 SunDirectionAtHour(float hour);
    // The same call at the hour of a state index: hour = kSunStateHour0 +
    // kSunStateStepHours * state. This is the bake's direction for that state.
    static DirectX::XMFLOAT3 SunDirectionAtHourCPU(uint32_t state);

    // The two states that bracket an hour, and the linear blend weight between
    // them. i0 is clamped to 0..kSunStates-2, so i0 + 1 is always a real state.
    // t = 0 at state i0, 1 at state i0 + 1. Below hour 6 and above hour 18 this
    // returns a ZERO t at the end pair and sunE is already near zero, so the term
    // fades out without a branch.
    struct SunHourBlend
    {
        uint32_t i0 = 0u;   // first bracketing state
        float    t  = 0.0f; // blend weight toward i0 + 1
        float    u  = 0.0f; // unclamped state coordinate; < 0 before sunrise
        bool     inDaylight = false; // hour inside [6, 18]
    };
    static SunHourBlend SunBlendForHour(float hour);

    ID3D12PipelineState* GetRayDebugPSO() const { return m_RayDebugPSO.Get(); }

    // ---------------------------------------------------------------------
    // Sun hour-state panel (debug UI only)
    //
    // The question it answers: "which states does this hour interpolate, and by
    // how much". Every row is one of the 17 baked states, so the table is a
    // complete map of the fold rather than a subset of it.
    //
    // The weight is GLOBAL: it depends only on the hour, so every probe shares it.
    // Only the two bracketing states carry a non-zero weight, and the two always
    // sum to 1. There is no per-probe column and no argmax any more: the old
    // panel ranked a kernel that no longer exists, and its "highest DC" marker
    // named a bin the fold never read.
    // ---------------------------------------------------------------------
    struct SunStateRow
    {
        uint32_t          state    = 0u;    // 0..kSunStates-1
        float             hour     = 0.0f;  // kSunStateHour0 + kSunStateStepHours * state
        float             angleDeg = 0.0f;  // angle to the sun direction at the current hour
        float             weight   = 0.0f;  // 0 except for the two bracketing states
        bool              active   = false; // this row is one of the two brackets
        DirectX::XMFLOAT3 dir      = { 0.0f, 0.0f, 0.0f };
        float             elevationDeg = 0.0f;
        float             azimuthDeg   = 0.0f;
    };

    struct SunStatePanel
    {
        SunStateRow rows[kSunStates];
        float    hour          = 12.0f;
        uint32_t i0            = 0u;    // first bracketing state
        float    t             = 0.0f;  // weight toward i0 + 1
        bool     inDaylight    = false; // hour inside [6, 18]: the sun term is on
        float    elevationDeg  = 0.0f;  // of the hour's sun direction
        float    azimuthDeg    = 0.0f;
    };

    // Fills the panel from the hour alone. Pure CPU, no GPU read-back, no state.
    static void BuildSunStatePanel(float hour, SunStatePanel& out);

    // Why a probe's meta word could not be read. Two causes are common and they
    // need different messages and different fixes: no readback buffer (the bake
    // has not run), or an index at or past the probe count (the grid shrank).
    enum class ProbeMetaStatus { Ok, NoReadback, ProbeOutOfRange, MapFailed };

    // One row of FindNearestValidProbes below.
    struct NearestProbe
    {
        uint32_t          index  = 0; // probe index = x + y*dimX + z*dimX*dimY
        uint32_t          cellX  = 0;
        uint32_t          cellY  = 0;
        uint32_t          cellZ  = 0;
        DirectX::XMFLOAT3 center = { 0.0f, 0.0f, 0.0f }; // world position of the probe centre
        float             distance = 0.0f;               // metres from the query point
    };

    // The k VALID probes nearest to a world point, ASCENDING by distance. Writes
    // at most k rows to out and returns how many it wrote.
    //
    // This serves the UI's "probes near the camera" list, which answers "which
    // probe index am I standing in?" without the user having to derive an index
    // from coordinates.
    //
    // ONE Map of the meta readback covers the whole scan. ReadProbeMeta maps and
    // unmaps per call, so a per-probe loop over it would map and unmap up to
    // kMaxProbeCount (128K) times — that is the only reason this is a function
    // and not a loop at the call site.
    //
    // With no meta readback (the bake has not run) it writes 0 rows: an
    // unverified probe is not offered as valid.
    //
    // Valid means meta bit 0 (BAKED_GI_META_VALID), the same bit the placement
    // view and LogBakeStats read.
    //
    // The cell->world formula is the one the debug overlay uses
    // (BakedGIProbeCellPos, BakedGI_Debug.hlsl:120): gridMin + cell * spacing,
    // with cell (0,0,0) at gridMin. The CPU computes gridMin, dims and spacing at
    // bake time and passes them to the shaders, so the CPU is the authority and
    // the shader is the mirror — this is not a second source of truth.
    uint32_t FindNearestValidProbes(const DirectX::XMFLOAT3& point, uint32_t k,
                                    std::vector<NearestProbe>& out) const;

    // ---------------------------------------------------------------------
    // Probe disk cache switch. STILL OFF, and the reason is unchanged.
    //
    // While it is false:
    //   - RecordBake never calls TryLoadCache, so every RecordBake re-runs the
    //     dispatches instead of restoring a Probes/ file. That is what makes a
    //     bake-shader edit take effect without deleting a stale cache file.
    //   - Renderer::BakeGIProbes skips RecordCacheReadback and WriteCacheFile, so
    //     no Probes/ file is written and the sun-table hash line never prints.
    //
    // CONSEQUENCE FOR THE VERSION-2 FORMAT: the round-trip test cannot run while
    // this is false. Set it to true, bake twice, and compare the two
    // "[BakedGI] sun table ..." FNV1a64 lines. That switch is a separate decision
    // from the format itself, so it is left as the human set it.
    static constexpr bool kProbeCacheEnabled = false;

    // Disk cache (Probes/ — the ShaderCache pattern), format version 2.
    //
    // After a bake, FOUR outputs are read back on the same list
    // (RecordCacheReadback, before executing) and written to
    // <exeDir>/Probes/<stem>_s<spacing>.probebin after the sync (WriteCacheFile):
    //   1. the 16-record response table,
    //   2. the meta words,
    //   3. the visibility moments,
    //   4. the per-probe SUN table, all 17 hour states, appended.
    // One file for all 17 states, which is the human's decision. Section 4 costs
    // 17 x 10 float4 x 16 B = 2720 B for each probe.
    //
    // A later RecordBake with the same grid config, the same state count and the
    // same latitude restores the state with plain buffer copies instead of 51
    // dispatches — check LoadedFromCache() after RecordBake to know which ran.
    bool RecordCacheReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList);
    void WriteCacheFile(const std::string& scenePath);
    bool LoadedFromCache() const { return m_LoadedFromCache; }

private:
    // Shared by the bake and the cache-load paths: (re)creates the seven
    // probe buffers for the current grid (dims/spacing/probeCount members).
    // wantSunTable: also create the per-probe sun response pair. Off for a bake
    // that does not need the sun, so its payload is byte-for-byte what it was.
   bool CreateProbeBuffers(bool wantSunTable = false);
    // Debug-only outputs of the bake: the per-term transport snapshots
    // (m_ResponseIter, kIterSnapshotCount tables) and the folded per-probe
    // response balls (m_EfbBall, kBakeIterations). Created on the DISPATCH path
    // only — the cache-load path has one table and never folds. Returns false
    // without voiding the bake: the capture is a debug aid, so RecordBake logs
    // the failure, leaves m_EfbBallsValid false and finishes.
    bool CreateIterationBuffers();
    // Copies the term table just written by the bake dispatch at m_Response[src]
    // into the snapshot for that term. Adds the transition pair the
    // record-copy-record pattern needs, exactly as the ray-debug dump's stored
    // table copy does.
    void SnapshotResponseTable(ID3D12GraphicsCommandList* cmdList, uint32_t src, uint32_t snapshot);
    // Restores the bake outputs from the Probes/ cache (a matching header).
    // Called inside RecordBake after the grid is computed; on a hit RecordBake
    // returns true immediately.
    //
    // The cache holds FOUR sections now: the 16-record response table, the meta
    // words, the visibility moments, and — appended, one file for all 17 states —
    // the per-probe sun table. wantSunTable is the same flag RecordBake was given:
    // it decides whether the sun buffer exists to copy into, so a caller that did
    // not ask for the sun term does not pay 2720 B for each probe of VRAM.
    bool TryLoadCache(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                      const std::string& scenePath, bool wantSunTable);

    // Ping-pong transport table: probeCount x kDirections x kResponseFloat4s float4s.
    GPUBuffer m_Response[2];
    // Sun hour-state series. Ping-pong for the same reason as m_Response: each
    // iteration's feedback reads the previous one. Stride is kSunStates records
    // per probe, so 17 x 160 B = 2720 B per probe per buffer.
    GPUBuffer m_SunResponse[2];
    bool      m_SunResponseValid = false;
    uint32_t  m_FinalSunResponse = 0;
    GPUBuffer m_LitProbes;    // per-frame lit SH9 RGB, TOTAL (runtime fetch source)
    GPUBuffer m_LitProbesSky; // per-frame lit SH9 RGB, sky-sourced split (debug)
    GPUBuffer m_LitProbesSun; // per-frame lit SH9 RGB, sun-sourced split (debug)
    GPUBuffer m_Meta;      // per-probe validity (bit 0)
    GPUBuffer m_Visibility; // per-probe octahedral 16x16 depth moments (padded 18x18, float2 (M1, M2))

    // Per-bake-term capture, debug-only (see HasEfbBalls). m_ResponseIter holds
    // terms 0..kBakeIterations-2 — the LAST term's table is m_Response
    // [m_FinalResponse] and is not duplicated, so the pair together covers every
    // term. m_EfbBall holds the folded response ball of each term.
    GPUBuffer m_ResponseIter[kIterSnapshotCount];
    GPUBuffer m_EfbBall[kBakeIterations];
    bool      m_EfbBallsValid = false;

    uint32_t  m_FinalResponse = 0; // response buffer holding the completed bake
    bool      m_Valid = false;
    float     m_Spacing = 1.0f;
    uint32_t  m_DimX = 0, m_DimY = 0, m_DimZ = 0, m_ProbeCount = 0;
    DirectX::XMFLOAT3 m_GridMin = { 0.0f, 0.0f, 0.0f };
    uint32_t  m_GridRevision = 0; // bumped where RecordBake establishes the grid

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_BakePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_VisBakePSO; // step-3 depth-moment visibility bake
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_UpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DebugPSO; // RTV R8G8B8A8 (post-composite overlay)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_LeakDebugPSO; // fullscreen leak map (post-composite, alpha-blended)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RayDebugPSO;  // on-demand ray-debug re-trace (compute)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IterFoldPSO;  // per-term response-ball fold (compute, debug)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_MetaReadback; // per-bake validity stats readback
    Microsoft::WRL::ComPtr<ID3D12Resource> m_CacheReadback; // per-bake serialization readback (response|meta|vis)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_CacheUpload;   // cache-load staging — must outlive the recorded list
    Microsoft::WRL::ComPtr<ID3D12Resource> m_RayDebugReadback; // CPU: debug buffer | stored table
    GPUBuffer m_RayDebugBuf;                                   // GPU: records + re-accumulated table
    uint32_t m_RayDebugProbe = 0;
    // Which table the last dump replayed. The readout runs separately and later,
    // so it needs the method to compare against the right table.
    bool     m_RayDebugSunMode = false;
    // Which hour state the last dump replayed, so the readout compares against
    // the right record of the sun table.
    uint32_t m_RayDebugSunState = 0u;
    // Written by LogRayDebugReadout, which is const, so these members are
    // mutable. m_RayDebugValid is true when the meta bit was read and the probe
    // is valid, and also when there is no verdict. RecordRayDebug clears the
    // pairing on every dump, so a verdict exists only for a completed readout.
    mutable bool     m_RayDebugValid      = true;
    // The probe whose readout produced m_RayDebugValid. UINT32_MAX means "no
    // verdict for the current dump". The getter trusts the flag only when this
    // matches m_RayDebugProbe, so a failed Map on the debug readback, or a dump
    // that never reached the readout, resolves to "do not dim".
    mutable uint32_t m_RayDebugValidProbe = UINT32_MAX;
    // The Cage Trace picker's pool, one list for each delta direction. Written by
    // LogRayDebugReadout (const, hence mutable) and CLEARED by RecordRayDebug, so
    // the pool always describes the records that are in the buffer right now.
    // m_RayCandidateDirs is the number of directions the last readout filled, and
    // it is what tells the UI "no dump has been read out" apart from "this
    // direction has no feedback-dominated ray".
    mutable std::vector<RayCandidate> m_RayCandidates[kDirections];
    mutable uint32_t m_RayCandidateDirs = 0u;
    bool m_LoadedFromCache = false;

    // Reads one probe's meta word out of the bake's meta readback. Returns Ok
    // and writes metaWord on success. The two failures are named so the readout
    // can print them differently: NoReadback (the bake has not run) and
    // ProbeOutOfRange (the probe index is past the current probe count). The
    // ray-debug readout uses the meta bit to tell an invalid probe from a
    // drifted stream, and it must read the bit rather than test the stored row
    // for zero: a zero row is not a validity test (see LogRayDebugReadout).
    ProbeMetaStatus ReadProbeMeta(uint32_t probe, uint32_t& metaWord) const;
};
