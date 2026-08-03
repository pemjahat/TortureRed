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

bool HZBCull(float3 worldCenter, float3 worldExtents, float4x4 viewProj,
             uint hzbSRVIdx, uint hzbMipCount, uint hzbWidth, uint hzbHeight)
{
    // --- Step 1: Project all 8 AABB corners to NDC ---
    float3 corners[8];
    corners[0] = worldCenter + float3(-worldExtents.x, -worldExtents.y, -worldExtents.z);
    corners[1] = worldCenter + float3( worldExtents.x, -worldExtents.y, -worldExtents.z);
    corners[2] = worldCenter + float3(-worldExtents.x,  worldExtents.y, -worldExtents.z);
    corners[3] = worldCenter + float3( worldExtents.x,  worldExtents.y, -worldExtents.z);
    corners[4] = worldCenter + float3(-worldExtents.x, -worldExtents.y,  worldExtents.z);
    corners[5] = worldCenter + float3( worldExtents.x, -worldExtents.y,  worldExtents.z);
    corners[6] = worldCenter + float3(-worldExtents.x,  worldExtents.y,  worldExtents.z);
    corners[7] = worldCenter + float3( worldExtents.x,  worldExtents.y,  worldExtents.z);

    // --- Step 2: Compute NDC bounding rect ---
    float minNDCx =  1.0f, minNDCy =  1.0f;
    float maxNDCx = -1.0f, maxNDCy = -1.0f;
    float nearestDepth = 0.0f; // Reverse-Z: nearest = largest (we'll max())

    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float4 clip = mul(float4(corners[i], 1.0f), viewProj);
        if (clip.w <= 0.0f)
        {
            // Corner is behind the near plane or at infinity — conservative: bounding rect
            // extends to the full screen edge, so occlusion test cannot falsely cull.
            minNDCx = -1.0f; maxNDCx = 1.0f;
            minNDCy = -1.0f; maxNDCy = 1.0f;
            // Don't skip depth — use the corner's expected depth after clamping
            continue;
        }
        float3 ndc = clip.xyz / clip.w;
        minNDCx = min(minNDCx, ndc.x);
        maxNDCx = max(maxNDCx, ndc.x);
        minNDCy = min(minNDCy, ndc.y);
        maxNDCy = max(maxNDCy, ndc.y);
        nearestDepth = max(nearestDepth, ndc.z); // Reverse-Z: larger = closer
    }

    // Clamp NDC to [-1,1] range (RT-safe)
    minNDCx = clamp(minNDCx, -1.0f, 1.0f);
    maxNDCx = clamp(maxNDCx, -1.0f, 1.0f);
    minNDCy = clamp(minNDCy, -1.0f, 1.0f);
    maxNDCy = clamp(maxNDCy, -1.0f, 1.0f);
    nearestDepth = saturate(nearestDepth);

    // If the rect is invalid or covers the full screen, don't cull
    float rectWidth  = maxNDCx - minNDCx;
    float rectHeight = maxNDCy - minNDCy;
    if (rectWidth <= 0.0f || rectHeight <= 0.0f || rectWidth >= 2.0f || rectHeight >= 2.0f)
        return false;

    // --- Step 3: Map NDC to HZB pixel coords (mip 0) ---
    // NDC [-1,1] → [0,1] UV → pixel coords
    float2 rectMinPixel = float2(
        (minNDCx * 0.5f + 0.5f) * (float)hzbWidth,
        (minNDCy * 0.5f + 0.5f) * (float)hzbHeight);
    float2 rectMaxPixel = float2(
        (maxNDCx * 0.5f + 0.5f) * (float)hzbWidth,
        (maxNDCy * 0.5f + 0.5f) * (float)hzbHeight);

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

    // --- Step 5: Sample 4×4 footprint at the chosen mip ---
    // 4 bilinear samples at the corners cover a 4×4 texel area.
    Texture2D<float> hzb = ResourceDescriptorHeap[hzbSRVIdx];
    float hzbSample0 = hzb.SampleLevel(PointClampSampler, (mipRectMin + 0.5f) / float2(hzbWidth >> mipLevel, hzbHeight >> mipLevel), mipLevel);
    float hzbSample1 = hzb.SampleLevel(PointClampSampler, (float2(mipRectMax.x, mipRectMin.y) + 0.5f) / float2(hzbWidth >> mipLevel, hzbHeight >> mipLevel), mipLevel);
    float hzbSample2 = hzb.SampleLevel(PointClampSampler, (float2(mipRectMin.x, mipRectMax.y) + 0.5f) / float2(hzbWidth >> mipLevel, hzbHeight >> mipLevel), mipLevel);
    float hzbSample3 = hzb.SampleLevel(PointClampSampler, (mipRectMax + 0.5f) / float2(hzbWidth >> mipLevel, hzbHeight >> mipLevel), mipLevel);

    // --- Step 6: Min-reduce (reverse-Z: farthest = smallest) ---
    float hzbDepth = min(min(hzbSample0, hzbSample1), min(hzbSample2, hzbSample3));

    // --- Step 7: Occlusion test ---
    // Object is occluded if its nearest point is farther than everything in the HZB.
    // Reverse-Z: nearest = larger, farthest = smaller.
    // hzbDepth = smallest (farthest) of 4×4 footprint.
    // If nearestDepth < hzbDepth, the object's closest point is farther than the
    // farthest thing the HZB contains → object is behind all HZB geometry → occluded.
    return nearestDepth < hzbDepth;
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

    // Frustum cull (fresh world-space AABB)
    if (!FrustumCullAABB(worldCenter, worldExtents, FrameCB.viewProj))
        return;

    // Occlusion cull (only if enabled and not frustum-only mode)
    bool occluded = false;
    if (CullConst.EnableOcclusion)    
    {
        occluded = HZBCull(worldCenter, worldExtents, FrameCB.viewProj,
                           CullConst.HZBSRVIdx, CullConst.HZBMipCount,
                           CullConst.HZBWidth, CullConst.HZBHeight);
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

    // transformToWorld
    if (!FrustumCullMeshlet(bounds, inst.LocalToWorld, FrameCB.viewProj))
        return;

    // Occlusion cull — transform local bounds by per-frame LocalToWorld
    // (extents must be expanded by abs(linear part), not passed through raw,
    // otherwise rotated/scaled instances get an underestimated AABB).
    if (CullConst.EnableOcclusion)
    {
        float3 worldCenter, worldExtents;
        TransformAABBToWorld(bounds.LocalCenter, bounds.LocalExtents, inst.LocalToWorld,
                             worldCenter, worldExtents);
        if (HZBCull(worldCenter, worldExtents, FrameCB.viewProj,
                    CullConst.HZBSRVIdx, CullConst.HZBMipCount,
                    CullConst.HZBWidth, CullConst.HZBHeight))
        {
            return;
        }
    }

    uint slot;
    InterlockedAdd(VisibleMeshletsCounter[0], 1, slot);
    VisibleMeshlets[slot] = cand;
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
