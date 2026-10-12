#ifndef DDGI_HLSLI
#define DDGI_HLSLI

// =============================================================================
// DDGI.hlsli — dynamic diffuse GI: octahedral probe atlas, addressing, fetch.
//
// Indirect method 3. 
// Each probe traces rays every frame and stores two atlases:
//
//   irradiance — RGBA16F, 6x6 interior texels + a 1-texel border ring.
//                Per direction: tone-curved (gamma 5) irradiance.
//   depth      — RG16F, 14x14 interior texels + a 1-texel border ring.
//                Per direction: (M1, M2) hit-distance moments.
//
// The border ring holds the MIRROR of the interior ring across each edge. A
// bilinear tap that falls half a texel outside the interior therefore reads
// the correct spherical neighbour of the octahedral fold instead of a
// different probe's data. The rule is in DDGIBorderSource below.
//
// ATLAS LAYOUT. The reference packs a probe at (x + y * dimX, z), so the atlas
// width is tileExtent * dimX * dimY. A 128 x 64 grid needs 65,536 texels of
// width, past the 16,384-texel device limit. This implementation packs the
// FLATTENED probe index instead, with probesPerRow = ceil(sqrt(probeCount))
// tiles in a row. The atlas is only ever read at a single probe's tile, never
// across two probes, so the layout change costs no quality.
//
// OCTAHEDRAL CONVENTION. The pack and the unpack are BakedGI's Y-up mapping
// (BakedGIEncodeOcta / BakedGIDecodeOcta), shared deliberately — one source of
// truth for the engine. The mapping and the ray direction set are independent
// bijections, so the convention only has to agree with itself. The ray set
// uses the z pole (BakedGIClassifyDirection, the reference SphericalFibonacci).
//
// REQUIRES in scope at include time:
//   - BakedGIEncodeOcta / BakedGIDecodeOcta / BakedGIClassifyDirection
//     (BakedGI.hlsli, included below)
//   - g_LinearSampler (CommonTracing.hlsl) for the atlas fetch
//   - ConstantBuffer<FrameConstants> FrameCB : register(b0) for the two
//     convenience wrappers that read the volume from FrameConstants
//
// The core functions take a DDGIVolumeGpu by value and read no globals, so the
// trace shader can build the volume from its root constants instead.
// =============================================================================

// BakedGI.hlsli supplies the shared octahedral mapping and the direction set.
// It references FrameCB, so every including shader declares FrameCB first.
#include "BakedGI.hlsli"

// --- Atlas dimensions (must match the C++ constants in DDGI.h) ---
#define DDGI_PROBE_IRRADIANCE_TEXELS 6
#define DDGI_PROBE_DEPTH_TEXELS      14

// Rays 0..31 stay unrotated so probe-state classification cannot flicker;
// rays 32.. rotate by a per-frame random axis-angle (the reference split).
#define DDGI_NUM_STABLE_RAYS         32

// Tone curve on the stored irradiance (encoding), and the depth multiplier
// that turns a backface hit into a negative depth for the relocation pass.
#define DDGI_PROBE_GAMMA             5.0f
#define DDGI_BACKFACE_DEPTH_MULTIPLIER (-0.2f)

// Floor on the cubed Chebyshev weight. The reference ships 0.05 and this port
// keeps it: DDGI re-traces the moments every frame, so the floor is the
// anti-flicker insurance for a moment that is momentarily wrong. BakedGI's
// zero floor (BAKED_GI_CHEB_MIN) applies to STATIC moments and must not be
// reused here — see BakedGI.hlsli's note on the corner-band leak.
#define DDGI_CHEB_MIN                0.05f

// Probe-state classification: a probe dies when this fraction of the stable
// rays hit a backface (it sits inside geometry).
#define DDGI_MAX_BACKFACE_FRACTION   0.25f

// Dispatch X tile for the two resolve passes: one group for each probe, with
// the probe index spread over X * Y to respect the 65535-per-axis dispatch cap.
#define DDGI_RESOLVE_DISPATCH_X      4096u

// The probe geometry a pass needs. The trace and the resolve passes build it
// from DDGIParams; the deferred fetch and the debug overlay build it from
// FrameConstants.
struct DDGIVolumeGpu
{
    float3 gridMin;
    float  spacing;
    uint3  dims;
    uint   probeCount;
    uint   probesPerRow;
    uint   numRays;
    uint   maxRays;
    uint   irradianceSRVIdx;
    uint   depthSRVIdx;
    uint   probeStateSRVIdx;
    uint   probeOffsetSRVIdx;
    float2 irradianceTexSize;
    float2 depthTexSize;
};

// ---------------------------------------------------------------------------
// Atlas addressing
// ---------------------------------------------------------------------------

uint DDGIFlatten3D(uint3 c, uint3 dims)
{
    return (c.z * dims.y + c.y) * dims.x + c.x;
}

uint3 DDGIUnflatten3D(uint index, uint3 dims)
{
    return uint3(index % dims.x, (index / dims.x) % dims.y, index / (dims.x * dims.y));
}

uint DDGITexelExtent(uint interiorTexels)
{
    return 1u + interiorTexels + 1u; // one border texel on each side
}

// Tile row length in tiles, from the probe count alone.
uint DDGIProbesPerRow(uint probeCount)
{
    uint row = (uint)ceil(sqrt((float)probeCount));
    return max(1u, row);
}

// Atlas size in texels, from the probe count alone.
float2 DDGITexSize(uint probeCount, uint interiorTexels)
{
    const uint extent = DDGITexelExtent(interiorTexels);
    const uint row    = DDGIProbesPerRow(probeCount);
    const uint rows   = (probeCount + row - 1u) / row;
    return float2((float)(extent * row), (float)(extent * rows));
}

// Top-left interior texel of a probe's tile.
uint2 DDGIGetProbeTexel(uint probeIdx, uint probesPerRow, uint interiorTexels)
{
    const uint  extent = DDGITexelExtent(interiorTexels);
    const uint2 tile   = uint2(probeIdx % probesPerRow, probeIdx / probesPerRow);
    return tile * extent + 1u;
}

// Normalised UV of the octahedral texel that holds `dir` for one probe.
float2 DDGIGetProbeUV(uint probeIdx, uint probesPerRow, uint interiorTexels, float2 texSize, float3 dir)
{
    float2 pixel = (float2)DDGIGetProbeTexel(probeIdx, probesPerRow, interiorTexels);
    pixel += BakedGIEncodeOcta(normalize(dir)) * (float)interiorTexels;
    return pixel / texSize;
}

// Source (interior) tile coordinate for a BORDER texel of a tile.
//
// Crossing an edge of the octahedral map mirrors the coordinate on the other
// axis (BakedGIWrapVisTexel uses the same rule for its manual bilinear). This
// one rule reproduces every entry of the reference's 28-entry and 60-entry
// border tables:
//   corner (2.5 sides, 2.5 sides) -> both axes pinned to the opposite interior edge
//   x is a border, y is not      -> src.y mirrors about the interior centre
//   y is a border, x is not      -> src.x mirrors about the interior centre
// `c` is a tile coordinate in [0, interiorTexels + 1].
uint2 DDGIBorderSource(int2 c, uint interiorTexels)
{
    const int lo = 0;
    const int hi = (int)interiorTexels + 1; // last tile index (border)
    const int xIsBorder = (c.x == lo || c.x == hi) ? 1 : 0;
    const int yIsBorder = (c.y == lo || c.y == hi) ? 1 : 0;

    int2 s = c;
    if (xIsBorder) s.x = (c.x == lo) ? (int)interiorTexels : 1;
    if (yIsBorder) s.y = (c.y == lo) ? (int)interiorTexels : 1;
    if (xIsBorder && !yIsBorder) s.y = (int)interiorTexels + 1 - c.y;
    if (yIsBorder && !xIsBorder) s.x = (int)interiorTexels + 1 - c.x;
    return (uint2)s;
}

// ---------------------------------------------------------------------------
// Ray directions
//
// The reference splits the set: a fixed full-sphere fibonacci set for the
// stable rays, and the same set rotated by a per-frame axis-angle for the
// rest. Only the rotating rays feed the irradiance and the depth resolves —
// the stable rays exist so that probe-state classification is deterministic
// across frames.
// ---------------------------------------------------------------------------
float3x3 DDGIAngleAxis3x3(float angle, float3 axis)
{
    const float s = sin(angle);
    const float c = cos(angle);
    const float t = 1.0f - c;
    const float3 a = normalize(axis);
    return float3x3(
        t * a.x * a.x + c,        t * a.x * a.y - s * a.z,  t * a.x * a.z + s * a.y,
        t * a.x * a.y + s * a.z,  t * a.y * a.y + c,        t * a.y * a.z - s * a.x,
        t * a.x * a.z - s * a.y,  t * a.y * a.z + s * a.x,  t * a.z * a.z + c);
}

// The fixed full-sphere set the stable rays use. Probe-state classification
// reads the same directions, so it must be this helper in both places.
float3 DDGIGetStableRayDirection(uint rayIndex, uint numRays)
{
    return BakedGIClassifyDirection(rayIndex, min(DDGI_NUM_STABLE_RAYS, numRays));
}

float3 DDGIGetRayDirection(uint rayIndex, uint numRays, float3x3 randomRotation)
{
    if (rayIndex < DDGI_NUM_STABLE_RAYS)
        return DDGIGetStableRayDirection(rayIndex, numRays);
    return mul(BakedGIClassifyDirection(rayIndex - DDGI_NUM_STABLE_RAYS,
                                        numRays - DDGI_NUM_STABLE_RAYS), randomRotation);
}

// ---------------------------------------------------------------------------
// Probe position, with the runtime relocation offset applied.
// ---------------------------------------------------------------------------
float3 DDGIProbePosition(DDGIVolumeGpu vol, uint3 c)
{
    float3 p = vol.gridMin + (float3)c * vol.spacing;
    StructuredBuffer<float4> offsets = ResourceDescriptorHeap[vol.probeOffsetSRVIdx];
    p += offsets[DDGIFlatten3D(c, vol.dims)].xyz;
    return p;
}

bool DDGIIsProbeActive(DDGIVolumeGpu vol, uint probeIdx)
{
    StructuredBuffer<uint> states = ResourceDescriptorHeap[vol.probeStateSRVIdx];
    return states[probeIdx] == 0u;
}

// ---------------------------------------------------------------------------
// Self-shadow bias (DDGI 2021 unified form). The query point slides off the
// receiver's own surface so the probe's stored geometry — which includes that
// surface — cannot occlude the receiver itself.
// ---------------------------------------------------------------------------
float3 DDGIComputeBias(float probeSize, float3 normal, float3 viewDir, float b)
{
    return (normal * 0.2f + viewDir * 0.8f) * 0.75f * probeSize * b;
}

// ---------------------------------------------------------------------------
// The fetch: DDGI's weighted blend of the 8 cage probes around a receiver.
//
//   w = trilinear * backface * chebyshev(cubed, floor 0.05) * crush
//
// Returns (irradiance.rgb, volumeWeight). The volume weight fades to zero at
// the grid border so a second volume could blend in later.
//
// The optional fallback ladder matters here as much as in BakedGI: if every
// cage probe is inactive or rejected, the result is black, not "no GI".
// ---------------------------------------------------------------------------
float4 DDGIQueryVolume(DDGIVolumeGpu vol, float3 position, float3 normal, float3 viewDir)
{
    if (vol.probeCount < 2u || vol.spacing <= 0.0f)
        return float4(0.0f, 0.0f, 0.0f, 0.0f);

    // --- Volume weight: fades to zero in the last cell on every axis. ---
    float3 rel = (position - vol.gridMin) / vol.spacing;
    float  volumeWeight = 1.0f;
    [unroll]
    for (uint i = 0; i < 3; ++i)
    {
        volumeWeight *= saturate(rel[i]);
        if (rel[i] > (float)vol.dims[i] - 2.0f)
            volumeWeight *= 1.0f - saturate(rel[i] - (float)vol.dims[i] + 2.0f);
    }
    if (volumeWeight <= 0.0f)
        return float4(0.0f, 0.0f, 0.0f, 0.0f);

    position += DDGIComputeBias(vol.spacing, normal, viewDir, 0.2f);

    const uint3 base = (uint3)clamp(floor(rel), float3(0.0f, 0.0f, 0.0f), (float3)(vol.dims - 1u));
    const float3 baseProbePos = DDGIProbePosition(vol, base);
    const float3 alpha = saturate((position - baseProbePos) / vol.spacing);

    Texture2D<float4> irradiance = ResourceDescriptorHeap[vol.irradianceSRVIdx];
    Texture2D<float2> depthMap   = ResourceDescriptorHeap[vol.depthSRVIdx];
    StructuredBuffer<uint> states = ResourceDescriptorHeap[vol.probeStateSRVIdx];

    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float  weightSum = 0.0f;

    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        const uint3 offset = uint3(k, k >> 1u, k >> 2u) & 1u;
        const uint3 c      = clamp(base + offset, uint3(0u, 0u, 0u), vol.dims - 1u);
        const uint  pIdx   = DDGIFlatten3D(c, vol.dims);
        if (states[pIdx] != 0u)
            continue;

        const float3 probePos = DDGIProbePosition(vol, c);
        const float3 toProbe  = probePos - position;      // receiver -> probe
        const float  dist     = length(toProbe);

        const float3 trilinear = max(0.001f.xxx, lerp(1.0f.xxx - alpha, alpha, (float3)offset));
        const float  trilinearWeight = trilinear.x * trilinear.y * trilinear.z;

        const float2 irrUV = DDGIGetProbeUV(pIdx, vol.probesPerRow, DDGI_PROBE_IRRADIANCE_TEXELS, vol.irradianceTexSize, normal);

        if (dist < 1e-5f)
        {
            // Receiver sits on the probe: no occlusion and no backface to test.
            sum += pow(irradiance.SampleLevel(g_LinearSampler, irrUV, 0).rgb, DDGI_PROBE_GAMMA * 0.5f) * trilinearWeight;
            weightSum += trilinearWeight;
            continue;
        }

        const float3 dirToProbe = toProbe / dist;

        // Backface: a cage member behind the receiver's tangent plane
        // (through-floor / through-wall) contributes nothing.
        float weight = saturate(dot(dirToProbe, normal));
        if (weight <= 0.0f)
            continue;

        // Chebyshev occlusion. The map stores rays fired FROM the probe, so the
        // lookup direction is probe -> receiver — the negation of dirToProbe.
        // Sampling at +dirToProbe would read the probe's far hemisphere (open
        // sky, M1 = sky distance -> weight 1) and never reject a walled-off
        // probe. BakedGI's fetch negates for the same reason.
        const float2 moments = depthMap.SampleLevel(g_LinearSampler,
                                   DDGIGetProbeUV(pIdx, vol.probesPerRow, DDGI_PROBE_DEPTH_TEXELS, vol.depthTexSize, -dirToProbe), 0).xy;
        float chebyshev = 1.0f;
        if (dist > moments.x)
        {
            const float variance = abs(moments.x * moments.x - moments.y);
            const float mD       = moments.x - dist;
            chebyshev = variance / (variance + mD * mD);
            chebyshev = chebyshev * chebyshev * chebyshev; // sharpen the rejection
        }
        weight *= max(chebyshev, DDGI_CHEB_MIN);
        weight  = max(1e-6f, weight);

        // Crush small weights so a barely-visible probe cannot tint the result.
        const float crushThreshold = 0.2f;
        if (weight < crushThreshold)
            weight *= weight * weight * (1.0f / (crushThreshold * crushThreshold));

        weight *= trilinearWeight;

        // Remove the tone curve and accumulate in linear space.
        sum += pow(irradiance.SampleLevel(g_LinearSampler, irrUV, 0).rgb, DDGI_PROBE_GAMMA * 0.5f) * weight;
        weightSum += weight;
    }

    if (weightSum <= 1e-7f)
        return float4(0.0f, 0.0f, 0.0f, volumeWeight);

    sum *= 1.0f / weightSum;
    sum *= sum;             // undo the second half of the gamma-5 curve
    sum *= 2.0f * PI;       // the resolve stores a hemisphere average
    return float4(max(sum, 0.0f.xxx), volumeWeight);
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

// Volume from the per-dispatch root constants (trace and resolve passes).
DDGIVolumeGpu DDGIGetVolumeFromParams(float3 gridMin, float spacing, uint3 dims,
                                      uint numRays, uint maxRays,
                                      uint irradianceSRVIdx, uint depthSRVIdx,
                                      uint probeStateSRVIdx, uint probeOffsetSRVIdx)
{
    DDGIVolumeGpu vol;
    vol.gridMin            = gridMin;
    vol.spacing            = spacing;
    vol.dims               = dims;
    vol.probeCount         = dims.x * dims.y * dims.z;
    vol.probesPerRow       = DDGIProbesPerRow(vol.probeCount);
    vol.numRays            = numRays;
    vol.maxRays            = maxRays;
    vol.irradianceSRVIdx   = irradianceSRVIdx;
    vol.depthSRVIdx        = depthSRVIdx;
    vol.probeStateSRVIdx   = probeStateSRVIdx;
    vol.probeOffsetSRVIdx  = probeOffsetSRVIdx;
    vol.irradianceTexSize  = DDGITexSize(vol.probeCount, DDGI_PROBE_IRRADIANCE_TEXELS);
    vol.depthTexSize       = DDGITexSize(vol.probeCount, DDGI_PROBE_DEPTH_TEXELS);
    return vol;
}

// Volume from FrameConstants (deferred consumer and debug overlay).
DDGIVolumeGpu DDGIGetVolumeFromFrame()
{
    DDGIVolumeGpu vol;
    vol.gridMin            = float3(FrameCB.ddgiGridMinX, FrameCB.ddgiGridMinY, FrameCB.ddgiGridMinZ);
    vol.spacing            = FrameCB.ddgiSpacing;
    vol.dims               = uint3(FrameCB.ddgiDimX, FrameCB.ddgiDimY, FrameCB.ddgiDimZ);
    vol.probeCount         = vol.dims.x * vol.dims.y * vol.dims.z;
    vol.probesPerRow       = max(1u, FrameCB.ddgiProbesPerRow);
    vol.numRays            = FrameCB.ddgiNumRaysPerProbe;
    vol.maxRays            = FrameCB.ddgiMaxRaysPerProbe;
    vol.irradianceSRVIdx   = FrameCB.ddgiIrradianceSRVIndex;
    vol.depthSRVIdx        = FrameCB.ddgiDepthSRVIndex;
    vol.probeStateSRVIdx   = FrameCB.ddgiProbeStateSRVIndex;
    vol.probeOffsetSRVIdx  = FrameCB.ddgiProbeOffsetSRVIndex;
    vol.irradianceTexSize  = DDGITexSize(vol.probeCount, DDGI_PROBE_IRRADIANCE_TEXELS);
    vol.depthTexSize       = DDGITexSize(vol.probeCount, DDGI_PROBE_DEPTH_TEXELS);
    return vol;
}

// Indirect diffuse irradiance at a receiver (the deferred consumer entry).
float3 DDGISampleIrradiance(float3 position, float3 normal, float3 viewDir)
{
    if (FrameCB.ddgiMode == 0u || FrameCB.ddgiValid == 0u)
        return float3(0.0f, 0.0f, 0.0f);
    return DDGIQueryVolume(DDGIGetVolumeFromFrame(), position, normal, viewDir).rgb;
}

#endif // DDGI_HLSLI
