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

#include "MeshletCommon.hlsli"

SamplerState PointClampSampler : register(s0);

// ---- Resources ----

ConstantBuffer<FrameConstants>        FrameCB   : register(b0);
ConstantBuffer<TwoPassCullConstants>  CullConst : register(b1);

// Instance-level resources
StructuredBuffer<InstanceData>    InstanceDataBuf    : register(t0);
StructuredBuffer<InstanceBounds>  InstanceBoundsBuf  : register(t1);
StructuredBuffer<MeshData>        MeshDataBuf        : register(t2);

// CullInstancesCS outputs
RWStructuredBuffer<MeshletCandidate> CandidateMeshlets      : register(u0);
RWStructuredBuffer<uint>             CandidateMeshletsCounter : register(u1);
RWStructuredBuffer<uint>             OccludedInstances        : register(u2);
RWStructuredBuffer<uint>             OccludedInstancesCounter : register(u3);

// CullMeshletsCS input: reads the SAME buffer CullInstancesCS wrote via the UAV
// above (CandidateMeshlets, u0). Deliberately NOT re-declared as a separate SRV —
// binding the same resource as both SRV and UAV in one root signature would
// require it to be in two mutually-exclusive resource states at once.
StructuredBuffer<MeshletBounds>    MeshletBoundsBuf    : register(t3);

// CullMeshletsCS outputs
RWStructuredBuffer<MeshletCandidate> VisibleMeshlets        : register(u4);
RWStructuredBuffer<uint>             VisibleMeshletsCounter : register(u5);

// Indirect dispatch args buffers
RWStructuredBuffer<uint> MeshletCullArgs     : register(u6);  // uint3 dispatch args
RWStructuredBuffer<uint> InstanceCullArgs    : register(u7);  // uint3 dispatch args

// =============================================================================
// HZB Occlusion Culling (reverse-Z)
//
// TortureRed uses reverse-Z: near=1.0, far=0.0, DepthFunc=GREATER_EQUAL.
// The HZB stores the FARTHEST depth per texel (min-reduce, since far=0 < near=1).
//
// Occlusion test:
//   1. Project the object's world-space AABB to NDC
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
    // Done ONCE by FrustumCullAABB (Adria-style shared projection, see
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
    // min/max swap under the flip. See docs/bug_hzbcull_ndcflip.md.
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
//                   On occlusion, defers to OccludedInstances[] for Phase 2.
// Phase 1 (SECOND): dispatchThreadID indexes into OccludedInstances[] (built by
//                   Phase 1). The real instance index is OccludedInstances[threadID].
//                   No further deferral — occluded instances are silently discarded.
//                   Uses the same previous-frame HZB as Phase 1 (conservative
//                   retest — catches boundary cases from Phase 1 thread-wave divergence).
// =============================================================================

[numthreads(64, 1, 1)]
void CullInstancesCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // --- Phase 2: redirect thread index through the deferred-instance list ---
    uint instanceIdx = dispatchThreadID.x;
    if (CullConst.Phase == TWO_PASS_PHASE_SECOND)
    {
        // Dispatch is sized by OccludedInstancesCounter; guard against empty dispatch.
        if (instanceIdx >= OccludedInstancesCounter[0])
            return;
        StructuredBuffer<uint> occludedList = ResourceDescriptorHeap[CullConst.OccludedInstancesSRVIdx];
        instanceIdx = occludedList[instanceIdx];
    }
    else
    {
        if (instanceIdx >= CullConst.NumInstances)
            return;
    }

    InstanceData inst    = InstanceDataBuf[instanceIdx];
    InstanceBounds bound = InstanceBoundsBuf[inst.BoundsIndex]; // LOCAL-space AABB
    MeshData      md     = MeshDataBuf[inst.MeshDataIndex];

    // Transform local bounds by the PER-FRAME LocalToWorld (updated each frame by
    // Model::UpdateNodeBuffer) — culling follows animated/moving instances.
    // Same pattern as D3D12_Research: FrustumCull(LocalBounds, LocalToWorld, viewProj).
    float3 worldCenter, worldExtents;
    TransformAABBToWorld(bound.BoundsCenter, bound.BoundsExtents, inst.LocalToWorld,
                         worldCenter, worldExtents);

    // Frustum cull (fresh world-space AABB) — shared projection; the NDC rect
    // is reused by HZBCull below instead of re-projecting the bounds.
    FrustumCullData fc = FrustumCullAABB(worldCenter, worldExtents, FrameCB.viewProj);
    if (!fc.IsVisible)
        return;

    // Occlusion cull (only if enabled and not frustum-only mode)
    bool occluded = false;
    if (CullConst.EnableOcclusion)    
    {
        HZBCullDebug dbg;
        occluded = HZBCull(fc, CullConst.HZBSRVIdx, CullConst.HZBMipCount,
                           CullConst.HZBWidth, CullConst.HZBHeight, dbg);

        // if (!occluded)
        if (occluded)
            RecordOccludedRect(dbg, CullConst.Phase, 1); // kind 1 = instance
    }

    if (occluded)
    {
        // Phase 1: defer to Phase 2 for retest against fresh HZB.
        // Phase 2: final pass — silently discard, no further deferral.
        if (CullConst.Phase == TWO_PASS_PHASE_FIRST)
        {
            uint slot;
            InterlockedAdd(OccludedInstancesCounter[0], 1, slot);
            OccludedInstances[slot] = instanceIdx;
        }
        return;
    }

    // Instance passed — enumerate all its meshlets into CandidateMeshlets.
    for (uint localIdx = 0; localIdx < md.MeshletCount; localIdx++)
    {
        uint slot;
        InterlockedAdd(CandidateMeshletsCounter[0], 1, slot);
        MeshletCandidate cand;
        cand.InstanceID   = instanceIdx;
        cand.MeshletIndex = localIdx;
        CandidateMeshlets[slot] = cand;
    }
}

// =============================================================================
// CullMeshletsCS — Meshlet-level frustum + occlusion culling
// =============================================================================

[numthreads(64, 1, 1)]
void CullMeshletsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint candidateIdx = dispatchThreadID.x;
    uint count = CandidateMeshletsCounter[0]; // read from the UAV Counter

    if (candidateIdx >= count)
        return;

    MeshletCandidate cand  = CandidateMeshlets[candidateIdx]; // read via the same UAV CullInstancesCS wrote
    InstanceData     inst  = InstanceDataBuf[cand.InstanceID];
    MeshData         md    = MeshDataBuf[inst.MeshDataIndex];

    // Load meshlet bounds
    MeshletBounds bounds = MeshletBoundsBuf[md.MeshletBoundsOffset + cand.MeshletIndex];

    // transformToWorld + frustum cull — shared projection; the NDC rect is
    // reused by HZBCull below (extents are expanded by abs(linear part) inside
    // TransformAABBToWorld, so rotated/scaled instances stay conservative).
    FrustumCullData fc = FrustumCullMeshlet(bounds, inst.LocalToWorld, FrameCB.viewProj);
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

    uint slot;
    InterlockedAdd(VisibleMeshletsCounter[0], 1, slot);
    VisibleMeshlets[slot] = cand;

    // Mip-selection tint sideband (task007 mode 3): record the mip this meshlet's
    // occlusion test used; 0xFF = no test ran (frustum-only / near-plane fallback).
    if (CullConst.DebugRecordMip)
    {
        RWStructuredBuffer<uint> mips = ResourceDescriptorHeap[CullConst.VisibleMeshletMipsUAVIdx];
        mips[slot] = dbg.Valid ? dbg.Mip : 0xFFu;
    }
}

// =============================================================================
// BuildMeshletCullIndirectArgsCS
// Reads CandidateMeshletsCounter → builds indirect dispatch for CullMeshletsCS
// =============================================================================

[numthreads(1, 1, 1)]
void BuildMeshletCullIndirectArgsCS()
{
    uint count = CandidateMeshletsCounter[0];
    MeshletCullArgs[0] = (count + 63) / 64; // ThreadGroupCountX
    MeshletCullArgs[1] = 1;
    MeshletCullArgs[2] = 1;
}

// =============================================================================
// BuildInstanceCullIndirectArgsCS
// Reads OccludedInstancesCounter → builds indirect dispatch for Phase 2 CullInstancesCS
// =============================================================================

[numthreads(1, 1, 1)]
void BuildInstanceCullIndirectArgsCS()
{
    uint count = OccludedInstancesCounter[0];
    InstanceCullArgs[0] = (count + 63) / 64; // ThreadGroupCountX
    InstanceCullArgs[1] = 1;
    InstanceCullArgs[2] = 1;
}
