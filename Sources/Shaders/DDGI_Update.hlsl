// =============================================================================
// DDGI_Update.hlsl — the resolve passes, plus the setup clears.
//
// Four entry points:
//   UpdateIrradianceCS    the ray buffer -> the irradiance atlas
//   UpdateDepthCS         the ray buffer -> the depth-moment atlas
//   UpdateProbeStatesCS   the ray buffer -> the probe offsets and the states
//   ClearIrradianceCS     setup only: zero one irradiance atlas
//   ClearDepthCS          setup only: zero one depth atlas
//   ClearProbeBuffersCS   setup only: zero the probe offsets and the states
//
// The two resolves and the state resolve each dispatch ONE THREAD GROUP FOR
// EACH PROBE. The probe index spreads over both dispatch axes so a grid past
// 65,535 probes still fits the per-axis dispatch cap.
//
// Two rules hold in all three resolves:
//   1. The early exits are UNIFORM across the group (the probe index and the
//      probe state come from groupId, not from the thread id). A
//      GroupMemoryBarrierWithGroupSync after a non-uniform return would
//      deadlock; this layout cannot produce one.
//   2. Only the ROTATING rays (index >= DDGI_NUM_STABLE_RAYS) contribute. The
//      32 stable rays exist so that probe-state classification is deterministic
//      across frames, and they stay out of the images.
//
// REQUIRES in scope: DDGIParams (b2). The clear entries read the same block:
// they take the atlas UAV from irradianceTargetUAVIdx / depthTargetUAVIdx, the
// probe buffer UAVs from probeOffsetUAVIdx / probeStateUAVIdx, and the atlas
// size from the probe count.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<DDGIParams>     DDGI    : register(b2);

// DDGI.hlsli includes BakedGI.hlsli, which references FrameCB.
#include "DDGI.hlsli"

uint DDGIResolveProbeIndex(uint2 groupId)
{
    // groupId.x walks DDGI_RESOLVE_DISPATCH_X probe tiles, groupId.y the rows.
    return groupId.y * DDGI_RESOLVE_DISPATCH_X + groupId.x;
}

// =============================================================================
// Irradiance resolve
// =============================================================================
#define DDGI_IRR_GS_SIZE (DDGI_PROBE_IRRADIANCE_TEXELS * DDGI_PROBE_IRRADIANCE_TEXELS)
groupshared float3 gsIrradiance[DDGI_IRR_GS_SIZE];
groupshared float3 gsIrradianceDirection[DDGI_IRR_GS_SIZE];

[numthreads(DDGI_PROBE_IRRADIANCE_TEXELS, DDGI_PROBE_IRRADIANCE_TEXELS, 1)]
void UpdateIrradianceCS(uint3 gtid : SV_GroupThreadID, uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    const uint3 dims       = uint3(DDGI.dimX, DDGI.dimY, DDGI.dimZ);
    const uint  probeCount = dims.x * dims.y * dims.z;
    const uint  probeIdx   = DDGIResolveProbeIndex(groupId.xy);
    if (probeIdx >= probeCount)
        return; // uniform: groupId is shared by every thread in the group

    StructuredBuffer<uint> states = ResourceDescriptorHeap[DDGI.probeStateSRVIdx];
    if (states[probeIdx] != 0u)
        return; // uniform: same probe for the whole group

    const uint probesPerRow = DDGIProbesPerRow(probeCount);
    const uint2 texel       = DDGIGetProbeTexel(probeIdx, probesPerRow, DDGI_PROBE_IRRADIANCE_TEXELS) + gtid.xy;
    const uint2 tileCorner  = DDGIGetProbeTexel(probeIdx, probesPerRow, DDGI_PROBE_IRRADIANCE_TEXELS) - 1u;

    Texture2D<float4>   history  = ResourceDescriptorHeap[DDGI.irradianceHistorySRVIdx];
    StructuredBuffer<float4> rayHits = ResourceDescriptorHeap[DDGI.rayHitSRVIdx];
    RWTexture2D<float4> target   = ResourceDescriptorHeap[DDGI.irradianceTargetUAVIdx];

    const float3 previous = history[texel].rgb;
    const float3 probeDir = BakedGIDecodeOcta(((float2)gtid.xy + 0.5f) / (float)DDGI_PROBE_IRRADIANCE_TEXELS);
    const float3x3 randomRotation = DDGIAngleAxis3x3(DDGI.randomAngle,
                                        float3(DDGI.randomVectorX, DDGI.randomVectorY, DDGI.randomVectorZ));

    float  weightSum = 0.0f;
    float3 sum       = float3(0.0f, 0.0f, 0.0f);

    uint rayIndex = DDGI_NUM_STABLE_RAYS;
    while (rayIndex < DDGI.numRaysPerProbe)
    {
        const uint rayCount = min(DDGI_IRR_GS_SIZE, DDGI.numRaysPerProbe - rayIndex);
        if (groupIndex < rayCount)
        {
            gsIrradiance[groupIndex] = rayHits[probeIdx * DDGI.maxRaysPerProbe + rayIndex + groupIndex].rgb;
            gsIrradianceDirection[groupIndex] = DDGIGetRayDirection(rayIndex + groupIndex, DDGI.numRaysPerProbe, randomRotation);
        }
        GroupMemoryBarrierWithGroupSync();

        [loop]
        for (uint i = 0; i < rayCount; ++i)
        {
            const float weight = saturate(dot(probeDir, gsIrradianceDirection[i]));
            sum += gsIrradiance[i] * weight;
            weightSum += weight;
        }

        // The cache is about to be overwritten by the next chunk.
        GroupMemoryBarrierWithGroupSync();
        rayIndex += DDGI_IRR_GS_SIZE;
    }

    // Divide by 2 * weightSum: the mean of saturate(dot(.)) over the hemisphere
    // is 0.5, so the factor 2 restores the hemisphere integral.
    const float epsilon = 1e-9f * (float)DDGI.numRaysPerProbe;
    sum *= 1.0f / max(2.0f * weightSum, epsilon);

    sum = pow(max(sum, 0.0f.xxx), 1.0f / DDGI_PROBE_GAMMA); // encode
    sum = lerp(previous, sum, saturate(1.0f - DDGI.historyBlendWeight));
    target[texel] = float4(sum, 1.0f);

    GroupMemoryBarrierWithGroupSync();

    // --- Border ring: mirror the interior ring across each edge so a bilinear
    // tap that falls half a texel outside the interior reads the correct
    // spherical neighbour instead of a different probe's tile. ---
    const uint extent    = DDGITexelExtent(DDGI_PROBE_IRRADIANCE_TEXELS);
    const uint tileTexels = extent * extent;
    [loop]
    for (uint t = groupIndex; t < tileTexels; t += DDGI_IRR_GS_SIZE)
    {
        const int2 c = int2(t % extent, t / extent);
        if (c.x > 0 && c.x < (int)extent - 1 && c.y > 0 && c.y < (int)extent - 1)
            continue; // interior — already written
        const uint2 src = DDGIBorderSource(c, DDGI_PROBE_IRRADIANCE_TEXELS);
        target[tileCorner + (uint2)c] = target[tileCorner + src];
    }
}

// =============================================================================
// Depth-moment resolve
// =============================================================================
#define DDGI_DEPTH_GS_SIZE (DDGI_PROBE_DEPTH_TEXELS * DDGI_PROBE_DEPTH_TEXELS)
groupshared float  gsDepth[DDGI_DEPTH_GS_SIZE];
groupshared float3 gsDepthDirection[DDGI_DEPTH_GS_SIZE];

#define DDGI_MIN_WEIGHT_THRESHOLD 0.0001f

[numthreads(DDGI_PROBE_DEPTH_TEXELS, DDGI_PROBE_DEPTH_TEXELS, 1)]
void UpdateDepthCS(uint3 gtid : SV_GroupThreadID, uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    const uint3 dims       = uint3(DDGI.dimX, DDGI.dimY, DDGI.dimZ);
    const uint  probeCount = dims.x * dims.y * dims.z;
    const uint  probeIdx   = DDGIResolveProbeIndex(groupId.xy);
    if (probeIdx >= probeCount)
        return; // uniform

    StructuredBuffer<uint> states = ResourceDescriptorHeap[DDGI.probeStateSRVIdx];
    if (states[probeIdx] != 0u)
        return; // uniform

    const uint probesPerRow = DDGIProbesPerRow(probeCount);
    const uint2 texel      = DDGIGetProbeTexel(probeIdx, probesPerRow, DDGI_PROBE_DEPTH_TEXELS) + gtid.xy;
    const uint2 tileCorner = DDGIGetProbeTexel(probeIdx, probesPerRow, DDGI_PROBE_DEPTH_TEXELS) - 1u;

    Texture2D<float2>   history  = ResourceDescriptorHeap[DDGI.depthHistorySRVIdx];
    StructuredBuffer<float4> rayHits = ResourceDescriptorHeap[DDGI.rayHitSRVIdx];
    RWTexture2D<float2> target   = ResourceDescriptorHeap[DDGI.depthTargetUAVIdx];

    const float2 previous = history[texel];
    const float3 probeDir = BakedGIDecodeOcta(((float2)gtid.xy + 0.5f) / (float)DDGI_PROBE_DEPTH_TEXELS);
    const float3x3 randomRotation = DDGIAngleAxis3x3(DDGI.randomAngle,
                                        float3(DDGI.randomVectorX, DDGI.randomVectorY, DDGI.randomVectorZ));

    float  weightSum = 0.0f;
    float2 sum       = float2(0.0f, 0.0f);

    uint rayIndex = DDGI_NUM_STABLE_RAYS;
    while (rayIndex < DDGI.numRaysPerProbe)
    {
        const uint rayCount = min(DDGI_DEPTH_GS_SIZE, DDGI.numRaysPerProbe - rayIndex);
        if (groupIndex < rayCount)
        {
            gsDepth[groupIndex] = rayHits[probeIdx * DDGI.maxRaysPerProbe + rayIndex + groupIndex].a;
            gsDepthDirection[groupIndex] = DDGIGetRayDirection(rayIndex + groupIndex, DDGI.numRaysPerProbe, randomRotation);
        }
        GroupMemoryBarrierWithGroupSync();

        [loop]
        for (uint i = 0; i < rayCount; ++i)
        {
            // A much sharper lobe than the irradiance resolve: 64 vs 1.
            const float weight = pow(saturate(dot(probeDir, gsDepthDirection[i])), 64.0f);
            if (weight > DDGI_MIN_WEIGHT_THRESHOLD)
            {
                const float depth = gsDepth[i];
                weightSum += weight;
                sum += float2(abs(depth), depth * depth) * weight;
            }
        }

        GroupMemoryBarrierWithGroupSync();
        rayIndex += DDGI_DEPTH_GS_SIZE;
    }

    if (weightSum > DDGI_MIN_WEIGHT_THRESHOLD)
        sum /= weightSum;

    sum = lerp(previous, sum, saturate(1.0f - DDGI.historyBlendWeight));
    target[texel] = sum;

    GroupMemoryBarrierWithGroupSync();

    const uint extent     = DDGITexelExtent(DDGI_PROBE_DEPTH_TEXELS);
    const uint tileTexels = extent * extent;
    [loop]
    for (uint t = groupIndex; t < tileTexels; t += DDGI_DEPTH_GS_SIZE)
    {
        const int2 c = int2(t % extent, t / extent);
        if (c.x > 0 && c.x < (int)extent - 1 && c.y > 0 && c.y < (int)extent - 1)
            continue;
        const uint2 src = DDGIBorderSource(c, DDGI_PROBE_DEPTH_TEXELS);
        target[tileCorner + (uint2)c] = target[tileCorner + src];
    }
}

// =============================================================================
// Probe-state and relocation resolve
//
// Reads the 32 STABLE rays of each probe:
//   - count the backface hits (depth < 0)
//   - find the nearest backface and the nearest / farthest frontface
//   - move the probe offset through the nearest backface when the probe is
//     inside geometry, otherwise away from a near frontface, otherwise back
//     toward its base position
//   - classify the probe: active when few enough stable rays hit a backface
//     and some frontface lies inside three probe sizes
// =============================================================================
[numthreads(32, 1, 1)]
void UpdateProbeStatesCS(uint3 dtid : SV_DispatchThreadID)
{
    const uint3 dims       = uint3(DDGI.dimX, DDGI.dimY, DDGI.dimZ);
    const uint  probeCount = dims.x * dims.y * dims.z;
    const uint  probeIdx   = dtid.x;
    if (probeIdx >= probeCount)
        return;

    StructuredBuffer<float4> rayHits = ResourceDescriptorHeap[DDGI.rayHitSRVIdx];
    RWStructuredBuffer<float4> offsets = ResourceDescriptorHeap[DDGI.probeOffsetUAVIdx];
    RWStructuredBuffer<uint>   states  = ResourceDescriptorHeap[DDGI.probeStateUAVIdx];

    const float3 previousOffset = offsets[probeIdx].xyz;

    const uint numStableRays   = min(DDGI.numRaysPerProbe, DDGI_NUM_STABLE_RAYS);
    const uint maxNumBackfaces = (uint)((float)numStableRays * DDGI_MAX_BACKFACE_FRACTION);
    uint numBackfaces = 0;

    int   nearestBackfaceIndex     = -1;
    float nearestBackfaceDistance  = 3.4e38f; // FLT_MAX
    int   nearestFrontfaceIndex    = -1;
    float nearestFrontfaceDistance = 3.4e38f;
    int   farthestFrontfaceIndex   = -1;
    float farthestFrontfaceDistance = 0.0f;

    [loop]
    for (uint rayIndex = 0; rayIndex < numStableRays; ++rayIndex)
    {
        float depth = rayHits[probeIdx * DDGI.maxRaysPerProbe + rayIndex].a;
        if (depth < 0.0f)
        {
            ++numBackfaces;
            depth /= DDGI_BACKFACE_DEPTH_MULTIPLIER;
            if (depth < nearestBackfaceDistance)
            {
                nearestBackfaceDistance = depth;
                nearestBackfaceIndex    = (int)rayIndex;
            }
        }
        else if (depth < nearestFrontfaceDistance)
        {
            nearestFrontfaceDistance = depth;
            nearestFrontfaceIndex    = (int)rayIndex;
        }
        else if (depth > farthestFrontfaceDistance)
        {
            farthestFrontfaceDistance = depth;
            farthestFrontfaceIndex    = (int)rayIndex;
        }
    }

    const float minFrontFaceDistanceThreshold = 0.2f;
    const float offsetScale = 0.5f;

    // Only the three branches below produce a new offset. The guard keeps the
    // previous offset when none of them fires.
    float3 newOffset      = previousOffset;
    bool   newOffsetValid = false;

    if (nearestBackfaceIndex != -1 && (float)numBackfaces / (float)numStableRays > DDGI_MAX_BACKFACE_FRACTION)
    {
        // The probe is inside geometry: move it through the nearest backface.
        const float3 direction = DDGIGetStableRayDirection((uint)nearestBackfaceIndex, DDGI.numRaysPerProbe);
        newOffset = previousOffset + direction * (nearestBackfaceDistance * (offsetScale + 1.0f));
        newOffsetValid = true;
    }
    else if (nearestFrontfaceDistance < minFrontFaceDistanceThreshold)
    {
        const float3 nearest  = DDGIGetStableRayDirection((uint)nearestFrontfaceIndex, DDGI.numRaysPerProbe);
        const float3 farthest = DDGIGetStableRayDirection((uint)farthestFrontfaceIndex, DDGI.numRaysPerProbe);
        // Only move away from the nearest frontface when that does not also
        // bring the probe closer to the nearest one.
        if (dot(nearest, farthest) <= 0.0f)
        {
            newOffset = previousOffset + farthest * min(farthestFrontfaceDistance, 1.0f) * offsetScale;
            newOffsetValid = true;
        }
    }
    else if (nearestFrontfaceDistance > minFrontFaceDistanceThreshold + offsetScale)
    {
        // Nothing is close: walk back toward the base position.
        const float pushBack = min(nearestFrontfaceDistance - minFrontFaceDistanceThreshold, length(previousOffset));
        if (pushBack > 0.0f)
        {
            newOffset = previousOffset - pushBack * normalize(previousOffset);
            newOffsetValid = true;
        }
    }

    // Accept a new offset only when it stays inside half a probe size. The
    // comparison is in normalised units, so it is scale-independent.
    float3 appliedOffset = previousOffset;
    if (newOffsetValid)
    {
        const float3 normalised = newOffset / DDGI.spacing;
        if (dot(normalised, normalised) < 0.25f)
            appliedOffset = newOffset;
    }

    offsets[probeIdx] = float4(appliedOffset, 0.0f);

    const float3 maxProbeDepth = DDGI.spacing * 3.0f;
    const bool isActive = numBackfaces < maxNumBackfaces && any(nearestFrontfaceDistance <= maxProbeDepth);
    states[probeIdx] = isActive ? 0u : 1u;
}

// =============================================================================
// Setup clears
//
// ClearUnorderedAccessViewFloat needs a descriptor from a non-shader-visible
// heap, and GPUTexture holds none. A compute clear is the substitute: the
// setup path dispatches these on a synced command list, so the atlases and the
// probe buffers are defined before the first frame. An uninitialized
// irradiance atlas would otherwise reach the trace pass's feedback term.
//
// The atlas size follows from the probe count alone (the DDGI.hlsli packing),
// so these entries need no extra parameters.
// =============================================================================

[numthreads(8, 8, 1)]
void ClearIrradianceCS(uint3 dtid : SV_DispatchThreadID)
{
    const uint probeCount = DDGI.dimX * DDGI.dimY * DDGI.dimZ;
    const float2 texSize  = DDGITexSize(probeCount, DDGI_PROBE_IRRADIANCE_TEXELS);
    if ((float)dtid.x >= texSize.x || (float)dtid.y >= texSize.y)
        return;
    RWTexture2D<float4> atlas = ResourceDescriptorHeap[DDGI.irradianceTargetUAVIdx];
    atlas[dtid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

[numthreads(8, 8, 1)]
void ClearDepthCS(uint3 dtid : SV_DispatchThreadID)
{
    const uint probeCount = DDGI.dimX * DDGI.dimY * DDGI.dimZ;
    const float2 texSize  = DDGITexSize(probeCount, DDGI_PROBE_DEPTH_TEXELS);
    if ((float)dtid.x >= texSize.x || (float)dtid.y >= texSize.y)
        return;
    RWTexture2D<float2> atlas = ResourceDescriptorHeap[DDGI.depthTargetUAVIdx];
    atlas[dtid.xy] = float2(0.0f, 0.0f);
}

[numthreads(64, 1, 1)]
void ClearProbeBuffersCS(uint3 dtid : SV_DispatchThreadID)
{
    const uint probeCount = DDGI.dimX * DDGI.dimY * DDGI.dimZ;
    if (dtid.x >= probeCount)
        return;
    RWStructuredBuffer<float4> offsets = ResourceDescriptorHeap[DDGI.probeOffsetUAVIdx];
    RWStructuredBuffer<uint>   states  = ResourceDescriptorHeap[DDGI.probeStateUAVIdx];
    offsets[dtid.x] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    states[dtid.x]  = 0u; // 0 = active: the first frame classifies every probe
}
