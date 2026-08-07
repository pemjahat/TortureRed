// MeshletTwoPassCull.hlsl
// Two-phase GPU occlusion culling: instance-level + meshlet-level, with HZB occlusion.
//
// Entry points (compiled with OCCLUSION_CULL=1 for occlusion, 0 for frustum-only fallback):
//   CullInstancesCS   — one thread per instance: frustum/HZB test, enumerate meshlets.
//   CullMeshletsCS    — one thread per candidate meshlet: frustum/HZB test → VisibleMeshlets.
//   BuildMeshletCullIndirectArgsCS  — builds the indirect dispatch for CullMeshletsCS.
//   BuildInstanceCullIndirectArgsCS — builds the indirect dispatch for Phase 2 CullInstancesCS.
//
// Defines (set at compile time):
//   OCCLUSION_CULL=1  — include HZB occlusion testing.
//   SECOND_PHASE=1    — Phase 2 behavior: process OccludedInstances instead of all instances.
//
// Resource binding: fully bindless. Every buffer is looked up via
// ResourceDescriptorHeap[CullConst.*Idx] at the point of use — there are no root
// SRV/UAV descriptors at all (root signature is just b0 FrameCB + b1 TwoPassCullConstants
// + a static sampler).
//
// This used to be a mix: most buffers were root SRV/UAV descriptors (register(t0-t4)/
// register(u0-u8)), except OccludedInstances, which additionally had a bindless SRV path
// (CullConst.OccludedInstancesSRVIdx) because it is UAV-written in Phase 1 but SRV-read in
// Phase 2 — a role reversal a single fixed root descriptor slot can't represent (one root
// parameter is bound as one type for the whole dispatch). Going fully bindless removes the
// inconsistency and the extra root-signature complexity of carrying 14 root descriptors
// that only differ in which register they occupy.

#include "MeshletCommon.hlsli"

SamplerState PointClampSampler : register(s0);

// ---- Resources ----
// See the file-header comment above — every buffer below is fetched via
// ResourceDescriptorHeap[CullConst.*Idx] at its point of use, not declared here.
ConstantBuffer<FrameConstants>        FrameCB   : register(b0);
ConstantBuffer<TwoPassCullConstants>  CullConst : register(b1);

// Recover (P00, P11, zNear) from FrameCB.projectionInverseUnjittered — the exact inverse
// of Camera::GetProjMatrix()'s reverse-Z infinite-far layout (verified algebraically):
//   projInv[0][0] = 1/P00, projInv[1][1] = 1/P11, projInv[2][3] = 1/zNear
// Using the *unjittered* inverse (same rationale as MotionVectors.hlsl) means these three
// scalars are jitter-immune even though jitter perturbs other entries of the matrix.
// Feeds FrustumCullSphere/FrustumCullMeshletSphere (MeshletCommon.hlsli), which need these
// to recover Niagara-style view-space quantities directly from clip space (no separate
// view matrix is stored in FrameConstants).
float3 GetProjScaleAndNear(float4x4 projInvUnjittered)
{
    return float3(1.0f / projInvUnjittered[0][0],
                 1.0f / projInvUnjittered[1][1],
                 1.0f / projInvUnjittered[2][3]);
}

// Which camera basis does CullConst.Phase's occlusion test actually sample the HZB in?
//   Phase 1 (FIRST)  — previous frame's HZB (built at the end of LAST frame, before this
//                       frame's own rasterize ran) → must project with viewProjPrevious,
//                       or the rect lands in the wrong place whenever the camera moved.
//   Phase 2 (SECOND) — THIS frame's fresh HZB, rebuilt from Phase 1's own depth output
//                       earlier in the same frame → already aligned with viewProj.
// See the GPU markers in Application.cpp: "Phase 1 - CullInstances (vs prev HZB)" /
// "Phase 2 - CullInstances (vs fresh HZB)". Frustum-only mode (no occlusion) never reads
// this — its value doesn't matter, so it just falls through to viewProj.
float4x4 GetHZBViewProj()
{
    return (CullConst.Phase == TWO_PASS_PHASE_FIRST) ? FrameCB.viewProjPrevious : FrameCB.viewProj;
}

// =============================================================================
// HZB Occlusion Culling (reverse-Z)
//
// TortureRed uses reverse-Z: near=1.0, far=0.0, DepthFunc=GREATER_EQUAL.
// The HZB stores the FARTHEST depth per texel (min-reduce, since far=0 < near=1).
//
// Occlusion test:
//   1. Analytically project the object's world-space bounding SPHERE to an NDC rect
//      (Mara/McGuire 2013 — see FrustumCullSphere in MeshletCommon.hlsli)
//   2. Find the NDC bounding rect and the object's nearest (largest) depth
//   3. Map rect to HZB pixel coords and pick an appropriate mip level
//   4. Sample a 4x4 texel footprint at that mip (4 bilinear samples at corners)
//   5. Min-reduce the sampled depths → conservative farthest HZB depth
//   6. Object is occluded if its nearest depth < HZB's farthest depth
//      (i.e. the closest point of the object is farther than everything in the HZB)
// =============================================================================

// Debug data captured by HZBCull for the occluded-rect overlay (task007 mode 1).
struct HZBCullDebug {
    float2 RectMinNDC;   // Clamped NDC rect (end of step 4)
    float2 RectMaxNDC;
    float  NearestDepth; // Object's closest point (reverse-Z: larger = closer)
    float  HZBDepth;     // Farthest occluder sampled from the HZB
    uint   Mip;          // HZB mip the test used
    uint   Valid;        // 1 = a full occlusion test ran (0 on the near-plane/degenerate early-outs)
    uint2  _pad;
    float2 SampleMinNDC; // NDC rect spanned by the 4 HZB sample texels (step 5)
    float2 SampleMaxNDC;
};

bool HZBCull(FrustumCullData fc,
             uint hzbSRVIdx, uint hzbMipCount, uint hzbWidth, uint hzbHeight,
             out HZBCullDebug dbg)
{
    dbg = (HZBCullDebug)0;

    // --- Steps 1-3: projection, NDC rect, nearest depth, near-plane fallback ---
    // Done ONCE by FrustumCullSphere (Niagara-style analytic sphere projection, see
    // MeshletCommon.hlsli) — here we just consume its output.
    // Clamp NDC to [-1,1] range (RT-safe)
    float minNDCx = clamp(fc.RectMin.x, -1.0f, 1.0f);
    float maxNDCx = clamp(fc.RectMax.x, -1.0f, 1.0f);
    float minNDCy = clamp(fc.RectMin.y, -1.0f, 1.0f);
    float maxNDCy = clamp(fc.RectMax.y, -1.0f, 1.0f);
    float nearestDepth = saturate(fc.RectMax.z); // Reverse-Z: larger = closer

    // If the rect is invalid or covers the full screen, don't cull
    float rectWidth  = maxNDCx - minNDCx;
    float rectHeight = maxNDCy - minNDCy;
    if (rectWidth <= 0.0f || rectHeight <= 0.0f || rectWidth >= 2.0f || rectHeight >= 2.0f)
        return false;

    // --- Step 3: Map NDC to HZB pixel coords (mip 0) ---
    // NDC [-1,1] → [0,1] UV → pixel coords.
    // Y must be FLIPPED: NDC y=+1 is the top of the screen, but HZB texel row 0 is also the
    // top of the screen (HZBInitCS Gather()s the depth buffer with the same row convention as
    // rasterization), so pixelY = (-ndc.y*0.5+0.5) * height — matching every other NDC->UV
    // conversion in this codebase (MotionVectors.hlsl, RestirDI/GI_Temporal.hlsl,
    // NrdPrepareGuides.hlsl) and Adria/D3D12_Research's HZBCull (float2(0.5,-0.5) swizzle in
    // GpuDrivenRendering.hlsli). minNDCy (bottom of screen) maps to the LARGER pixel Y, so
    // min/max swap under the flip.
    float2 rectMinPixel = float2(
        (minNDCx * 0.5f + 0.5f) * (float)hzbWidth,
        (-maxNDCy * 0.5f + 0.5f) * (float)hzbHeight);
    float2 rectMaxPixel = float2(
        (maxNDCx * 0.5f + 0.5f) * (float)hzbWidth,
        (-minNDCy * 0.5f + 0.5f) * (float)hzbHeight);

    float2 rectSize = max(rectMaxPixel - rectMinPixel, 1.0f.xx);

    // --- Step 4: Pick HZB mip level ---
    // Choose mip where the rect footprint is about 4×4 HZB texels.
    // ceil(log2(max(rectSize))) gives a mip where 1 texel ≈ the footprint.
    // Add 1 for 2×2 oversample; clamp.
    float mipLevelF = ceil(log2(max(rectSize.x, rectSize.y)));
    uint  mipLevel  = min(uint(mipLevelF), hzbMipCount - 1);

    // Scale pixel coords to the chosen mip
    float mipScale = exp2(-float(mipLevel));
    float2 mipRectMin = rectMinPixel * mipScale;
    float2 mipRectMax = rectMaxPixel * mipScale;
    //int4 mipRect = int4(rectMinPixel >> mipLevel;
    //float2 mipRectMax = rectMaxPixel >> mipLevel;

    // --- Step 5: Sample the footprint at the chosen mip ---
    // Snap OUTWARD to the integer texel rect that fully encloses the object rect
    // ([floor(min), ceil(max)-1]) and tap its 4 corner texels. Sampling at
    // (mipRectMin + 0.5) instead would round the min side INWARD whenever the
    // rect starts in the upper half of a texel, leaving a strip of the object
    // uncovered by any depth sample (visible in the debug overlay as the object
    // rect sticking out of the sampled-texel rect).
    float2 mipTexelMin = floor(mipRectMin);
    float2 mipTexelMax = ceil(mipRectMax) - 1.0f;

    Texture2D<float> hzb = ResourceDescriptorHeap[hzbSRVIdx];
    float2 texelSize = 1.f / uint2(hzbWidth, hzbHeight) * (1u << mipLevel);
    float hzbSample0 = hzb.SampleLevel(PointClampSampler, (mipTexelMin + 0.5f) * texelSize, mipLevel);
    float hzbSample1 = hzb.SampleLevel(PointClampSampler, (float2(mipTexelMax.x, mipTexelMin.y) + 0.5f) * texelSize, mipLevel);
    float hzbSample2 = hzb.SampleLevel(PointClampSampler, (float2(mipTexelMin.x, mipTexelMax.y) + 0.5f) * texelSize, mipLevel);
    float hzbSample3 = hzb.SampleLevel(PointClampSampler, (mipTexelMax + 0.5f) * texelSize, mipLevel);

    // --- Step 6: Min-reduce (reverse-Z: farthest = smallest) ---
    float hzbDepth = min(min(hzbSample0, hzbSample1), min(hzbSample2, hzbSample3));

    // Debug record output — consumed when this test occludes the candidate
    dbg.RectMinNDC   = float2(minNDCx, minNDCy);
    dbg.RectMaxNDC   = float2(maxNDCx, maxNDCy);
    dbg.NearestDepth = nearestDepth;
    dbg.HZBDepth     = hzbDepth;
    dbg.Mip          = mipLevel;
    dbg.Valid        = 1;
    // NDC rect of the sampled HZB texel block — encloses RectMin/MaxNDC by
    // construction; comparing sizes shows the texel:object ratio of the test.
    // Inverts the Step 3 Y-flip: pixel space is flipped relative to NDC, so the smaller pixel
    // row (top of screen) maps to the LARGER NDC y and vice versa (min/max swap on Y only).
    float2 hzbDims = float2((float)hzbWidth, (float)hzbHeight);
    float2 sampleMinPixel = mipTexelMin * (float)(1u << mipLevel);
    float2 sampleMaxPixel = (mipTexelMax + 1.0f) * (float)(1u << mipLevel);
    dbg.SampleMinNDC = float2(sampleMinPixel.x / hzbDims.x * 2.0f - 1.0f,
                              1.0f - 2.0f * (sampleMaxPixel.y / hzbDims.y));
    dbg.SampleMaxNDC = float2(sampleMaxPixel.x / hzbDims.x * 2.0f - 1.0f,
                              1.0f - 2.0f * (sampleMinPixel.y / hzbDims.y));

    // --- Step 7: Occlusion test ---
    // Object is occluded if its nearest point is farther than everything in the HZB.
    // Reverse-Z: nearest = larger, farthest = smaller.
    // hzbDepth = smallest (farthest) of 4×4 footprint.
    // If nearestDepth < hzbDepth, the object's closest point is farther than the
    // farthest thing the HZB contains → object is behind all HZB geometry → occluded.
    return nearestDepth < hzbDepth;
}

// =============================================================================
// Occluded-rect debug recording (task007 mode 1)
// Appends an OccludedRectDebug record for a candidate rejected by HZBCull.
// No-op unless the CPU enabled recording via CullConst.DebugRecordOccluded.
// =============================================================================
void RecordOccludedRect(HZBCullDebug dbg, uint phase, uint kind)
{
    if (!CullConst.DebugRecordOccluded)
        return;

    RWStructuredBuffer<uint> counter = ResourceDescriptorHeap[CullConst.OccludedRectsCounterUAVIdx];
    uint slot;
    InterlockedAdd(counter[0], 1, slot);
    if (slot >= MAX_OCCLUDED_RECT_DEBUG)
        return;

    RWStructuredBuffer<OccludedRectDebug> rects = ResourceDescriptorHeap[CullConst.OccludedRectsUAVIdx];
    OccludedRectDebug rec;
    rec.RectMinNDC   = dbg.RectMinNDC;
    rec.RectMaxNDC   = dbg.RectMaxNDC;
    rec.NearestDepth = dbg.NearestDepth;
    rec.HZBDepth     = dbg.HZBDepth;
    rec.Mip          = dbg.Mip;
    rec.Phase        = phase;
    rec.Kind         = kind;
    rec._pad0        = 0;
    rec.SampleMinNDC = dbg.SampleMinNDC;
    rec.SampleMaxNDC = dbg.SampleMaxNDC;
    rects[slot] = rec;
}

// =============================================================================
// CullInstancesCS — Instance-level frustum + occlusion culling.
//
// Phase 0 (FIRST):  dispatchThreadID is the instance index into InstanceData[].
//                   Tests against LAST frame's HZB (see GetHZBViewProj above).
//                   On occlusion, defers to OccludedInstances[] for Phase 2.
// Phase 1 (SECOND): dispatchThreadID indexes into OccludedInstances[] (built by
//                   Phase 1). The real instance index is OccludedInstances[threadID].
//                   Tests against THIS frame's fresh HZB (rebuilt from Phase 1's own
//                   depth output earlier this frame — catches instances Phase 1 wrongly
//                   thought were still occluded). No further deferral — occluded
//                   instances are silently discarded.
// =============================================================================

[numthreads(64, 1, 1)]
void CullInstancesCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> occludedInstancesCounter = ResourceDescriptorHeap[CullConst.OccludedInstancesCounterIdx];

    // --- Phase 2: redirect thread index through the deferred-instance list ---
    uint instanceIdx = dispatchThreadID.x;
    if (CullConst.Phase == TWO_PASS_PHASE_SECOND)
    {
        // Dispatch is sized by OccludedInstancesCounter; guard against empty dispatch.
        if (instanceIdx >= occludedInstancesCounter[0])
            return;
        StructuredBuffer<uint> occludedList = ResourceDescriptorHeap[CullConst.OccludedInstancesSRVIdx];
        instanceIdx = occludedList[instanceIdx];
    }
    else
    {
        if (instanceIdx >= CullConst.NumInstances)
            return;
    }

    StructuredBuffer<InstanceData>   instanceDataBuf   = ResourceDescriptorHeap[CullConst.InstanceDataSRVIdx];
    StructuredBuffer<InstanceBounds> instanceBoundsBuf = ResourceDescriptorHeap[CullConst.InstanceBoundsSRVIdx];
    StructuredBuffer<MeshData>       meshDataBuf       = ResourceDescriptorHeap[CullConst.MeshDataSRVIdx];

    InstanceData inst    = instanceDataBuf[instanceIdx];
    InstanceBounds bound = instanceBoundsBuf[inst.BoundsIndex]; // LOCAL-space bounding sphere
    MeshData      md     = meshDataBuf[inst.MeshDataIndex];

    // Reject alpha-blended instances — never enter the meshlet pipeline.
    // ALPHA_MODE_BLEND=2; shared CPU/GPU constant in Shared/SharedTypes.h.
    if (md.AlphaMode == ALPHA_MODE_BLEND)
        return;

    // Transform local bounds by the PER-FRAME LocalToWorld (updated each frame by
    // Model::UpdateNodeBuffer) — culling follows animated/moving instances.
    // Same pattern as D3D12_Research: FrustumCull(LocalBounds, LocalToWorld, viewProj).
    float3 worldCenter; float worldRadius;
    TransformSphereToWorld(bound.BoundsCenter, bound.BoundsRadius, inst.LocalToWorld,
                           worldCenter, worldRadius);

    // Frustum cull (fresh world-space sphere) against the CURRENT camera. The HZB rect
    // uses a separate projection (GetHZBViewProj) since Phase 1 samples last frame's HZB.
    float3 projParams = GetProjScaleAndNear(FrameCB.projectionInverseUnjittered);
    FrustumCullData fc = FrustumCullSphere(worldCenter, worldRadius, FrameCB.viewProj, GetHZBViewProj(),
                                           projParams.x, projParams.y, projParams.z);
    if (!fc.IsVisible)
        return;

    // Occlusion cull (only if enabled and not frustum-only mode)
    bool occluded = false;
    if (CullConst.EnableOcclusion)    
    {
        HZBCullDebug dbg;
        occluded = HZBCull(fc, CullConst.HZBSRVIdx, CullConst.HZBMipCount,
                           CullConst.HZBWidth, CullConst.HZBHeight, dbg);

        if (occluded)
            RecordOccludedRect(dbg, CullConst.Phase, 1); // kind 1 = instance
    }

    if (occluded)
    {
        // Phase 1: defer to Phase 2 for retest against fresh HZB.
        if (CullConst.Phase == TWO_PASS_PHASE_FIRST)
        {
            RWStructuredBuffer<uint> occludedInstances = ResourceDescriptorHeap[CullConst.OccludedInstancesUAVIdx];
            uint slot;
            InterlockedAdd(occludedInstancesCounter[0], 1, slot);
            occludedInstances[slot] = instanceIdx;
        }
        return;
    }

    // Instance passed frustum + HZB culling — count it once.
    {
        RWStructuredBuffer<uint> visibleInstancesCounter = ResourceDescriptorHeap[CullConst.VisibleInstancesCounterUAVIdx];
        uint instanceSlot;
        InterlockedAdd(visibleInstancesCounter[CullConst.Phase], 1, instanceSlot);
    }

    // Instance passed — enumerate all its meshlets into CandidateMeshlets.
    RWStructuredBuffer<uint> candidateMeshletsCounter = ResourceDescriptorHeap[CullConst.CandidateMeshletsCounterIdx];
    RWStructuredBuffer<MeshletCandidate> candidateMeshlets = ResourceDescriptorHeap[CullConst.CandidateMeshletsUAVIdx];
    for (uint localIdx = 0; localIdx < md.MeshletCount; localIdx++)
    {
        uint slot;
        InterlockedAdd(candidateMeshletsCounter[0], 1, slot);
        MeshletCandidate cand;
        cand.InstanceID   = instanceIdx;
        cand.MeshletIndex = localIdx;
        candidateMeshlets[slot] = cand;
    }
}

// =============================================================================
// CullMeshletsCS — Meshlet-level frustum + occlusion culling
// =============================================================================

[numthreads(64, 1, 1)]
void CullMeshletsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> candidateMeshletsCounter = ResourceDescriptorHeap[CullConst.CandidateMeshletsCounterIdx];
    uint candidateIdx = dispatchThreadID.x;
    uint count = candidateMeshletsCounter[0]; // read from the UAV Counter

    if (candidateIdx >= count)
        return;

    RWStructuredBuffer<MeshletCandidate> candidateMeshlets = ResourceDescriptorHeap[CullConst.CandidateMeshletsUAVIdx];
    MeshletCandidate cand = candidateMeshlets[candidateIdx]; // same buffer CullInstancesCS wrote

    StructuredBuffer<InstanceData> instanceDataBuf = ResourceDescriptorHeap[CullConst.InstanceDataSRVIdx];
    StructuredBuffer<MeshData>     meshDataBuf     = ResourceDescriptorHeap[CullConst.MeshDataSRVIdx];
    InstanceData     inst  = instanceDataBuf[cand.InstanceID];
    MeshData         md    = meshDataBuf[inst.MeshDataIndex];

    // Load meshlet bounds (bounding sphere)
    StructuredBuffer<MeshletBounds> meshletBoundsBuf = ResourceDescriptorHeap[CullConst.MeshletBoundsSRVIdx];
    MeshletBounds bounds = meshletBoundsBuf[md.MeshletBoundsOffset + cand.MeshletIndex];

    // transformToWorld + frustum cull against the CURRENT camera (radius is scaled by the
    // transform's max axis scale inside TransformSphereToWorld, so rotated/scaled instances
    // stay conservative). The HZB rect uses a separate projection (GetHZBViewProj) since
    // Phase 1 samples last frame's HZB.
    float3 projParams = GetProjScaleAndNear(FrameCB.projectionInverseUnjittered);
    FrustumCullData fc = FrustumCullMeshletSphere(bounds, inst.LocalToWorld, FrameCB.viewProj, GetHZBViewProj(),
                                                  projParams.x, projParams.y, projParams.z);
    if (!fc.IsVisible)
        return;

    HZBCullDebug dbg = (HZBCullDebug)0;
    if (CullConst.EnableOcclusion)
    {
        if (HZBCull(fc, CullConst.HZBSRVIdx, CullConst.HZBMipCount,
                    CullConst.HZBWidth, CullConst.HZBHeight, dbg))
        {
            RecordOccludedRect(dbg, CullConst.Phase, 0); // kind 0 = meshlet
            return;
        }
    }

    // VisibleMeshletsCounter[2]: slot [TWO_PASS_PHASE_FIRST] is Phase 1's own count,
    // slot [TWO_PASS_PHASE_SECOND] is Phase 2's own count. Phase 2 appends AFTER Phase 1's
    // range (base offset = Phase 1's FINAL count) instead of restarting at 0 — the counter
    // is only ever cleared once per frame (GPUCulling::CullTwoPass, gated to Phase 1), so
    // Phase 1's range in VisibleMeshlets stays intact and resolvable for the whole frame.
    RWStructuredBuffer<uint> visibleMeshletsCounter = ResourceDescriptorHeap[CullConst.VisibleMeshletsCounterUAVIdx];
    RWStructuredBuffer<MeshletCandidate> visibleMeshlets = ResourceDescriptorHeap[CullConst.VisibleMeshletsUAVIdx];

    uint slot;
    if (CullConst.Phase == TWO_PASS_PHASE_SECOND)
    {
        InterlockedAdd(visibleMeshletsCounter[TWO_PASS_PHASE_SECOND], 1, slot);
        slot += visibleMeshletsCounter[TWO_PASS_PHASE_FIRST];
    }
    else
    {
        InterlockedAdd(visibleMeshletsCounter[TWO_PASS_PHASE_FIRST], 1, slot);
    }
    visibleMeshlets[slot] = cand;

    // Mip-selection tint sideband : record the mip this meshlet's
    // occlusion test used; 0xFF = no test ran (frustum-only / near-plane fallback).
    if (CullConst.DebugRecordMip)
    {
        RWStructuredBuffer<uint> mips = ResourceDescriptorHeap[CullConst.VisibleMeshletMipsUAVIdx];
        mips[slot] = dbg.Valid ? dbg.Mip : 0xFFu;
    }
}

// =============================================================================
// BuildMeshletCullIndirectArgsCS
// Reads CandidateMeshletsCounter → builds indirect dispatch for CullMeshletsCS.
// =============================================================================

[numthreads(1, 1, 1)]
void BuildMeshletCullIndirectArgsCS()
{
    RWStructuredBuffer<uint> candidateMeshletsCounter = ResourceDescriptorHeap[CullConst.CandidateMeshletsCounterIdx];
    RWStructuredBuffer<uint> meshletCullArgs           = ResourceDescriptorHeap[CullConst.MeshletCullArgsUAVIdx];

    uint count = candidateMeshletsCounter[0];
    meshletCullArgs[0] = (count + 63) / 64; // ThreadGroupCountX
    meshletCullArgs[1] = 1;
    meshletCullArgs[2] = 1;
}

// =============================================================================
// BuildInstanceCullIndirectArgsCS
// Reads OccludedInstancesCounter[0] — the P1-deferred count, preserved across phases
// because OccludedInstancesCounter is only cleared in Phase 1.
// =============================================================================

[numthreads(1, 1, 1)]
void BuildInstanceCullIndirectArgsCS()
{
    RWStructuredBuffer<uint> occludedInstancesCounter = ResourceDescriptorHeap[CullConst.OccludedInstancesCounterIdx];
    RWStructuredBuffer<uint> instanceCullArgs          = ResourceDescriptorHeap[CullConst.InstanceCullArgsUAVIdx];

    uint count = occludedInstancesCounter[0];
    instanceCullArgs[0] = (count + 63) / 64; // ThreadGroupCountX
    instanceCullArgs[1] = 1;
    instanceCullArgs[2] = 1;
}
