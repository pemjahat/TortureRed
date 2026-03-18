// IrCache_Update.hlsl
// Indirect compute dispatch — one group per live probe, IRCACHE_RAYS_PER_PROBE threads/group.
// Trace rays from each probe and build a single reservoir per probe.

#include "IrCache_Common.hlsl"
#include "IrCache_Lookup.hlsl"
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>         g_Frame   : register(b0);
ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

groupshared Reservoir gs_Candidates[IRCACHE_RAYS_PER_PROBE];

[numthreads(IRCACHE_RAYS_PER_PROBE, 1, 1)]
void main(uint3 GroupId   : SV_GroupID,
          uint  GroupIdx  : SV_GroupIndex)
{
    // ---- identify probe ----
    RWByteAddressBuffer  meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];
    uint tracingCount = meta.Load(IRCACHE_META_TRACING_ALLOC_COUNT);
    if (GroupId.x >= tracingCount)
    {
        gs_Candidates[GroupIdx] = (Reservoir)0;
        return;
    }

    RWStructuredBuffer<uint> indirection = ResourceDescriptorHeap[g_IrCache.IndirectionBufIdx];
    uint entryIdx = indirection[GroupId.x];

    RWStructuredBuffer<uint> entryCell = ResourceDescriptorHeap[g_IrCache.EntryCellBufIdx];
    uint cellIdx = entryCell[entryIdx];

    IrcacheCoord coord    = ircache_cell_idx_to_coord(cellIdx);

#if IRCACHE_USE_POSITION_VOTING
    // Use the voted position from the previous frame; fall back to the cell
    // centre on the very first trace (w == 0 means not yet voted upon).
    RWStructuredBuffer<float4> g_PosBuffer = ResourceDescriptorHeap[g_IrCache.PosBufIdx];
    float4 storedPos = g_PosBuffer[entryIdx];
    float3 probePos  = (storedPos.w != 0.0f)
        ? storedPos.xyz
        : ircache_coord_to_world_center(coord, g_Frame.irCacheCameraPosition.xyz);
#else
    float3 probePos = ircache_coord_to_world_center(coord, g_Frame.irCacheCameraPosition.xyz);
#endif

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

    Reservoir localR = (Reservoir)0;

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        Surface hitSurf;
        ResolveHitSurface(ray, q.CommittedRayT(), q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), hitSurf);

        float3 direct   = GetDirectLightingHybrid(hitSurf.worldPos, hitSurf.normal, hitSurf.viewDir, hitSurf.albedo,
            hitSurf.metallic, hitSurf.roughness, g_Scene, g_Lights, g_Frame.numLights, g_Frame, true, rng);
        // SampleIrCache returns pure incident irradiance (no albedo baked in).
        // Apply albedo only to the indirect term; direct is already exitant radiance
        // (BSDF * cos evaluated by GetDirectLightingHybrid).
        float3 indirect = SampleIrCache(hitSurf.worldPos, g_IrCache, g_Frame.irCacheCameraPosition.xyz, hitSurf.normal);
        IrCacheMaybeAllocate(hitSurf.worldPos, g_IrCache, g_Frame.irCacheCameraPosition.xyz, hitSurf.normal);

        float3 sampleRadiance = direct + indirect * hitSurf.albedo;
        float sampleWeight = max(1e-5f, Luminance(sampleRadiance));
        if (sampleWeight > 0.0f)
        {
            // One candidate per thread, merged later into the final probe reservoir.
            updateReservoir(localR, hitSurf.worldPos, hitSurf.normal, sampleRadiance, sampleWeight, 0.0f);
            localR.W = 1.0f;
        }

#if IRCACHE_USE_POSITION_VOTING
        // ---- Cast a position vote for the probe covering hitSurf.worldPos ----
        // The probe at hitSurf.worldPos should migrate toward the visible surface.
        {
            IrcacheCoord hitCoord  = ws_pos_to_ircache_coord(hitSurf.worldPos, g_Frame.irCacheCameraPosition.xyz, hitSurf.normal);
            uint         hitCell   = hitCoord.cell_idx();
            RWByteAddressBuffer hitGrid = ResourceDescriptorHeap[g_IrCache.GridMetaBufIdx];
            uint hitPacked = hitGrid.Load(hitCell * 4);
            uint hitFlags  = ircache_cell_flags(hitPacked);

            if ((hitFlags & (IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_ALLOCATED))
                    == (IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_ALLOCATED))
            {
                uint hitEntry = ircache_cell_entry(hitPacked);

                // Read the probe's current applied position (may be cell centre on first trace)
                float4 hitStoredPos = g_PosBuffer[hitEntry];
                float3 hitProbePos  = (hitStoredPos.w != 0.0f)
                    ? hitStoredPos.xyz
                    : ircache_coord_to_world_center(hitCoord, g_Frame.irCacheCameraPosition.xyz);

                float  cellDiam  = ircache_cascade_cell_diameter(hitCoord.cascade);
                float3 proposal  = ircache_proposal_pos(hitProbePos, hitSurf.worldPos, cellDiam);

                // Uniform reservoir sampling: accept with probability 1/(voteCount+1)
                RWByteAddressBuffer countBuf = ResourceDescriptorHeap[g_IrCache.ReproposalCountBufIdx];
                uint prevCount;
                countBuf.InterlockedAdd(hitEntry * 4, 1u, prevCount);

#if IRCACHE_USE_UNIFORM_VOTING
                if (next_float(rng) <= 1.0f / (float)(prevCount + 1u))
#endif
                {
                    RWStructuredBuffer<float4> repropBuf = ResourceDescriptorHeap[g_IrCache.RepropBufIdx];
                    repropBuf[hitEntry] = float4(proposal, 1.0f);
                }
            }
        }
#endif
    }

    gs_Candidates[GroupIdx] = localR;
    GroupMemoryBarrierWithGroupSync();

    // ---- thread 0 combines candidates + writes probe reservoir ----
    if (GroupIdx == 0)
    {
        Reservoir outR = (Reservoir)0;
        float selectedPDF = 0.0f;
        RNG reduceRng;
        seed_rng(reduceRng, uint2(entryIdx, 131u), g_Frame.frameIndex);

        [unroll]
        for (int k = 0; k < IRCACHE_RAYS_PER_PROBE; ++k)
        {
            Reservoir candidate = gs_Candidates[k];
            if (candidate.M > 0.0f)
            {
                float shiftedTarget = max(1e-5f, Luminance(candidate.radiance));
                if (mergeReservoirs(outR, candidate, shiftedTarget, next_float(reduceRng)))
                    selectedPDF = shiftedTarget;
            }
        }

        if (outR.M > 0.0f && selectedPDF > 0.0f)
            outR.W = outR.w_sum / (outR.M * selectedPDF);
        else
            outR.W = 0.0f;

        RWStructuredBuffer<Reservoir> probeReservoirs = ResourceDescriptorHeap[g_IrCache.IrradianceBufIdx];

        RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[g_IrCache.GridMetaBufIdx];
        uint flags      = ircache_cell_flags(gridMeta.Load(cellIdx * 4));
        bool firstTrace = !(flags & IRCACHE_ENTRY_META_TRACED);
        if (firstTrace)
        {
            // Mark probe payload as valid so SampleIrCache can read this entry.
            // InterlockedOr only touches bit 2 (TRACED), leaving entryIdx in bits [31:3] intact.
            uint dummy;
            gridMeta.InterlockedOr(cellIdx * 4, IRCACHE_ENTRY_META_TRACED, dummy);

#if IRCACHE_USE_POSITION_VOTING
            // Seed reprop_buf with the first hit so the probe immediately has a
            // plausible surface position for the next frame's Age pass to apply.
            if (outR.M > 0.0f)
            {
                RWStructuredBuffer<float4> repropBuf = ResourceDescriptorHeap[g_IrCache.RepropBufIdx];
                repropBuf[entryIdx] = float4(outR.hitPos, 1.0f);
            }
#endif
        }

        if (!firstTrace)
        {
            Reservoir prev = probeReservoirs[entryIdx];
            if (prev.M > 0.0f)
            {
                float shiftedTarget = max(1e-5f, Luminance(prev.radiance));
                if (mergeReservoirs(outR, prev, shiftedTarget, next_float(reduceRng)))
                    selectedPDF = shiftedTarget;
                if (outR.M > 0.0f && selectedPDF > 0.0f)
                    outR.W = outR.w_sum / (outR.M * selectedPDF);
            }
        }

        probeReservoirs[entryIdx] = outR;

        // Reset life so the aging pass keeps this probe alive
        RWByteAddressBuffer life = ResourceDescriptorHeap[g_IrCache.LifeBufIdx];
        life.Store(entryIdx * 4, 0u);
    }
}
