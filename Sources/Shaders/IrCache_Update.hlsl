// IrCache_Update.hlsl
// Indirect compute dispatch — one group per live probe, IRCACHE_RAYS_PER_PROBE threads/group.
// Trace rays from each probe, accumulate radiance with groupshared reduction,
// EMA-blend into the irradiance buffer.

#include "IrCache_Common.hlsl"
#include "IrCache_Lookup.hlsl"
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>         g_Frame   : register(b0);
ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

groupshared float3 gs_Radiance[IRCACHE_RAYS_PER_PROBE];

[numthreads(IRCACHE_RAYS_PER_PROBE, 1, 1)]
void main(uint3 GroupId   : SV_GroupID,
          uint  GroupIdx  : SV_GroupIndex)
{
    // ---- identify probe ----
    RWByteAddressBuffer  meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];
    uint tracingCount = meta.Load(IRCACHE_META_TRACING_ALLOC_COUNT);
    if (GroupId.x >= tracingCount)
    {
        gs_Radiance[GroupIdx] = float3(0, 0, 0);
        return;
    }

    RWStructuredBuffer<uint> indirection = ResourceDescriptorHeap[g_IrCache.IndirectionBufIdx];
    uint entryIdx = indirection[GroupId.x];

    RWStructuredBuffer<uint> entryCell = ResourceDescriptorHeap[g_IrCache.EntryCellBufIdx];
    uint cellIdx = entryCell[entryIdx];

    IrcacheCoord coord    = ircache_cell_idx_to_coord(cellIdx);
    float3       probePos = ircache_coord_to_world_center(coord, g_Frame.irCacheCameraPosition.xyz);

    // ---- trace one ray (this thread = ray GroupIdx) ----
    RNG rng;
    seed_rng(rng, uint2(entryIdx, GroupIdx), g_Frame.frameIndex);

    // Uniform sphere direction
    float2 u = float2(next_float(rng), next_float(rng));
    float  z = 1.0f - 2.0f * u.x;
    float  r = sqrt(max(0.0f, 1.0f - z * z));
    float  phi = 6.28318530f * u.y;
    float3 rayDir = float3(r * cos(phi), r * sin(phi), z);

    RayDesc ray;
    ray.Origin    = probePos;
    ray.Direction = rayDir;
    ray.TMin      = 0.01f;
    ray.TMax      = 1000.0f;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) { PROCESS_ALPHA_MASK(q, rng); }

    float3 sampleRadiance = float3(0, 0, 0);

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint instanceIdx = q.CommittedInstanceID();
        uint triIdx      = q.CommittedPrimitiveIndex();
        float2 barys     = q.CommittedTriangleBarycentrics();
        DrawNodeData      nodeData = g_DrawNodeBuffer[instanceIdx];
        MaterialConstants mat     = g_Materials[nodeData.materialID];

        uint i0 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 0];
        uint i1 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 1];
        uint i2 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 2];
        GLTFVertex v0 = g_GlobalVertices[nodeData.vertexOffset + i0];
        GLTFVertex v1 = g_GlobalVertices[nodeData.vertexOffset + i1];
        GLTFVertex v2 = g_GlobalVertices[nodeData.vertexOffset + i2];

        float  bary0    = 1.0f - barys.x - barys.y;
        float2 hitUv    = v0.texCoord * bary0 + v1.texCoord * barys.x + v2.texCoord * barys.y;
        float3 hitNorm  = normalize(mul(
            v0.normal * bary0 + v1.normal * barys.x + v2.normal * barys.y,
            (float3x3)nodeData.world));

        float4 albedo    = mat.baseColorFactor;
        float  metallic  = mat.metallicFactor;
        float  roughness = mat.roughnessFactor;
        if (mat.baseColorTextureIndex >= 0)
            albedo *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
        if (mat.metallicRoughnessTextureIndex >= 0)
        {
            float4 mr = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
            roughness *= mr.g;
            metallic  *= mr.b;
        }

        float3 hitPos  = ray.Origin + ray.Direction * q.CommittedRayT();
        float3 viewDir = -ray.Direction;

        float3 direct   = GetDirectLightingHybrid(hitPos, hitNorm, viewDir, albedo.rgb,
            metallic, roughness, g_Scene, g_Lights, g_Frame.numLights, g_Frame, true, rng);
        // SampleIrCache returns pure incident irradiance (no albedo baked in).
        // Apply albedo only to the indirect term; direct is already exitant radiance
        // (BSDF * cos evaluated by GetDirectLightingHybrid).
        float3 indirect = SampleIrCache(hitPos, g_IrCache, g_Frame.irCacheCameraPosition.xyz, hitNorm);
        IrCacheMaybeAllocate(hitPos, g_IrCache, g_Frame.irCacheCameraPosition.xyz, hitNorm);

        sampleRadiance = direct + indirect * albedo.rgb;
    }

    gs_Radiance[GroupIdx] = sampleRadiance;
    GroupMemoryBarrierWithGroupSync();

    // ---- thread 0 reduces + writes irradiance ----
    if (GroupIdx == 0)
    {
        float3 total = float3(0, 0, 0);
        [unroll]
        for (int k = 0; k < IRCACHE_RAYS_PER_PROBE; ++k)
            total += gs_Radiance[k];
        float3 newIrr = total / (float)IRCACHE_RAYS_PER_PROBE;

        RWStructuredBuffer<float4> irradiance = ResourceDescriptorHeap[g_IrCache.IrradianceBufIdx];
        float4 prev = irradiance[entryIdx];

        RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[g_IrCache.GridMetaBufIdx];
        uint flags      = ircache_cell_flags(gridMeta.Load(cellIdx * 4));
        bool firstTrace = !(flags & IRCACHE_ENTRY_META_TRACED);
        float blend = (firstTrace || g_Frame.frameIndex == 0) ? 1.0f : 0.05f;

        if (firstTrace)
        {
            // Mark irradiance as valid; SampleIrCache will now return real data.
            // InterlockedOr only touches bit 2 (TRACED), leaving entryIdx in bits [31:3] intact.
            uint dummy;
            gridMeta.InterlockedOr(cellIdx * 4, IRCACHE_ENTRY_META_TRACED, dummy);
        }

        irradiance[entryIdx] = float4(lerp(prev.rgb, newIrr, blend), 1.0f);

        // Reset life so the aging pass keeps this probe alive
        RWByteAddressBuffer life = ResourceDescriptorHeap[g_IrCache.LifeBufIdx];
        life.Store(entryIdx * 4, 0u);
    }
}
