#ifndef IRCACHE_LOOKUP_HLSL
#define IRCACHE_LOOKUP_HLSL

#include "IrCache_Common.hlsl"

// ---------------------------------------------------------------------------
// SampleIrCache
//   Read the stored reservoir contribution for the probe covering `worldPos`.
//   Returns float3(0) unless all three flags are set:
//     OCCUPIED  — cell is claimed
//     ALLOCATED — entryIdx field is valid
//     TRACED    — IrCache_Update has written at least one irradiance sample
//   The probe payload stores a reservoir. We expose a conservative estimate of
//   incident radiance as selected radiance scaled by reservoir normalization.
//   Life management is owned exclusively by IrCache_Update; this function is read-only.
// ---------------------------------------------------------------------------
static const uint IRCACHE_FLAGS_READY =
    IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_ALLOCATED | IRCACHE_ENTRY_META_TRACED;

float ircache_shift_jacobian(float3 queryPos, float3 probePos, float3 sampleHitPos, float3 sampleHitNormal)
{
    float3 dQuery = sampleHitPos - queryPos;
    float3 dProbe = sampleHitPos - probePos;
    float distSqQuery = max(1e-4f, dot(dQuery, dQuery));
    float distSqProbe = max(1e-4f, dot(dProbe, dProbe));
    float cosQuery = max(1e-4f, abs(dot(sampleHitNormal, dQuery * rsqrt(distSqQuery))));
    float cosProbe = max(1e-4f, abs(dot(sampleHitNormal, dProbe * rsqrt(distSqProbe))));
    return (cosQuery * distSqProbe) / max(1e-5f, cosProbe * distSqQuery);
}

float3 SampleIrCache(float3 worldPos, IrCacheBindlessIndices ircache, float3 cameraPos, float3 normal)
{
    IrcacheCoord coord   = ws_pos_to_ircache_coord(worldPos, cameraPos, normal);
    uint         cellIdx = coord.cell_idx();

    RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[ircache.GridMetaBufIdx];
    uint packed = gridMeta.Load(cellIdx * 4);

    // Require all three positive assertions before touching entryIdx
    if ((ircache_cell_flags(packed) & IRCACHE_FLAGS_READY) != IRCACHE_FLAGS_READY)
        return float3(0.0f, 0.0f, 0.0f);

    uint entryIdx = ircache_cell_entry(packed);

    RWStructuredBuffer<Reservoir> probeReservoirs = ResourceDescriptorHeap[ircache.IrradianceBufIdx];
    Reservoir r = probeReservoirs[entryIdx];
    if (r.M <= 0.0f || r.W <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);

    //float3 probePos = ircache_coord_to_world_center(coord, cameraPos);
    //float jacobian = ircache_shift_jacobian(worldPos, probePos, r.hitPos, r.hitNormal);
    //jacobian = clamp(jacobian, 0.1f, 10.0f);

    //return r.radiance * (r.W * jacobian);
    return r.radiance * r.W;
}

// ---------------------------------------------------------------------------
// IrCacheMaybeAllocate
//   If the cell at `worldPos` is empty, atomically claim it and pop a free
//   entry from the pool, writing the back-links into grid metadata.
//   Safe to call from many threads simultaneously; only one wins per cell.
// ---------------------------------------------------------------------------
void IrCacheMaybeAllocate(float3 worldPos, IrCacheBindlessIndices ircache, float3 cameraPos, float3 normal)
{
    IrcacheCoord coord   = ws_pos_to_ircache_coord(worldPos, cameraPos, normal);
    uint         cellIdx = coord.cell_idx();

    RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[ircache.GridMetaBufIdx];
    uint packed = gridMeta.Load(cellIdx * 4);

    // Already occupied by another probe — nothing to do
    if (ircache_cell_flags(packed) & IRCACHE_ENTRY_META_OCCUPIED)
        return;

    // Race to own this cell (one winner per cell per frame).
    // CAS from 0 → OCCUPIED only succeeds if the whole word is still zero,
    // ensuring no stale entryIdx bits from a previous probe survive.
    uint prev;
    gridMeta.InterlockedCompareExchange(cellIdx * 4, 0u, IRCACHE_ENTRY_META_OCCUPIED, prev);
    if (prev != 0u)
        return; // Another thread won

    // Pop a free entry from the pool
    RWByteAddressBuffer meta = ResourceDescriptorHeap[ircache.MetaBufIdx];
    uint oldAllocCount;
    meta.InterlockedAdd(IRCACHE_META_ALLOC_COUNT, 1u, oldAllocCount);

    if (oldAllocCount >= (uint)IRCACHE_MAX_ENTRIES)
    {
        // Pool exhausted — revert the alloc count and release the cell claim.
        // CAS back to 0: only OCCUPIED was written, so the word must still equal OCCUPIED.
        uint dummy;
        meta.InterlockedAdd(IRCACHE_META_ALLOC_COUNT, uint(-1), dummy);
        gridMeta.InterlockedCompareExchange(cellIdx * 4, IRCACHE_ENTRY_META_OCCUPIED, 0u, dummy);
        return;
    }

    RWStructuredBuffer<uint> pool = ResourceDescriptorHeap[ircache.PoolBufIdx];
    uint entryIdx = pool[oldAllocCount]; // pop from top of free stack

    // Record entry's owning cell
    RWStructuredBuffer<uint> entryCell = ResourceDescriptorHeap[ircache.EntryCellBufIdx];
    entryCell[entryIdx] = cellIdx;

    // Mark the entry as alive (life = 0 = brand new)
    RWByteAddressBuffer life = ResourceDescriptorHeap[ircache.LifeBufIdx];
    life.Store(entryIdx * 4, 0u);

#if IRCACHE_USE_POSITION_VOTING
    // Seed both position buffers with the allocation world position so the probe
    // starts near the requesting surface immediately (w=1 signals a valid position).
    RWStructuredBuffer<float4> posBuf   = ResourceDescriptorHeap[ircache.PosBufIdx];
    RWStructuredBuffer<float4> repropBuf = ResourceDescriptorHeap[ircache.RepropBufIdx];
    posBuf[entryIdx]    = float4(worldPos, 1.0f);
    repropBuf[entryIdx] = float4(worldPos, 1.0f);
#endif

    // Atomically publish entryIdx + OCCUPIED + ALLOCATED in a single word.
    // Any reader observing ALLOCATED=1 reads the correct entryIdx from this
    // same atomic — no DeviceMemoryBarrier required.
    // TRACED is still absent: SampleIrCache returns float3(0) until Update traces.
    uint dummy;
    gridMeta.InterlockedExchange(cellIdx * 4,
        ircache_pack_cell(entryIdx, IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_ALLOCATED),
        dummy);

    // Extend entry_count high-watermark so the Age pass sweeps far enough
    meta.InterlockedMax(IRCACHE_META_ENTRY_COUNT, oldAllocCount + 1u, dummy);
}

// ---------------------------------------------------------------------------
// DebugIrCacheLife
//   Green (fresh) → yellow → red (about to expire) heat ramp.
//   Unoccupied cells return dark grey.
// ---------------------------------------------------------------------------
float3 DebugIrCacheLife(float3 worldPos, IrCacheBindlessIndices ircache, float3 cameraPos)
{
    IrcacheCoord coord   = ws_pos_to_ircache_coord(worldPos, cameraPos);
    uint         cellIdx = coord.cell_idx();

    RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[ircache.GridMetaBufIdx];
    uint packed = gridMeta.Load(cellIdx * 4);
    uint flags  = ircache_cell_flags(packed);
    // Require at least OCCUPIED|ALLOCATED before reading entryIdx
    if ((flags & (IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_ALLOCATED))
            != (IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_ALLOCATED))
        return float3(0.05f, 0.05f, 0.05f);    // unoccupied or mid-allocation — dark grey

    uint entryIdx = ircache_cell_entry(packed);

    RWByteAddressBuffer life = ResourceDescriptorHeap[ircache.LifeBufIdx];
    uint lifeVal = life.Load(entryIdx * 4);

    // t=0 → fresh (green), t=1 → about to expire (red)
    float  t      = saturate((float)lifeVal / (float)(IRCACHE_ENTRY_LIFE_MAX - 1));
    float3 fresh  = float3(0.0f, 1.0f, 0.0f);   // green
    float3 mid    = float3(1.0f, 1.0f, 0.0f);   // yellow
    float3 expiry = float3(1.0f, 0.0f, 0.0f);   // red
    return t < 0.5f ? lerp(fresh, mid, t * 2.0f) : lerp(mid, expiry, (t - 0.5f) * 2.0f);
}

// ---------------------------------------------------------------------------
// DebugIrCacheCascade
//   Each cascade gets a distinct false colour.
//   Occupied cell → full brightness. Unoccupied → 20% (shows coverage extent).
// ---------------------------------------------------------------------------
float3 DebugIrCacheCascade(float3 worldPos, IrCacheBindlessIndices ircache, float3 cameraPos)
{
    static const float3 cascadeColors[8] =
    {
        float3(1.00f, 0.20f, 0.20f),   // 0 — red
        float3(1.00f, 0.60f, 0.10f),   // 1 — orange
        float3(1.00f, 1.00f, 0.10f),   // 2 — yellow
        float3(0.20f, 1.00f, 0.20f),   // 3 — green
        float3(0.10f, 1.00f, 1.00f),   // 4 — cyan
        float3(0.20f, 0.40f, 1.00f),   // 5 — blue
        float3(0.70f, 0.20f, 1.00f),   // 6 — violet
        float3(1.00f, 0.20f, 0.80f),   // 7 — magenta
    };

    IrcacheCoord coord   = ws_pos_to_ircache_coord(worldPos, cameraPos);
    uint         cellIdx = coord.cell_idx();

    RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[ircache.GridMetaBufIdx];
    uint packed   = gridMeta.Load(cellIdx * 4);
    bool occupied = (ircache_cell_flags(packed) & IRCACHE_ENTRY_META_OCCUPIED) != 0;

    float3 baseColor = cascadeColors[coord.cascade];
    return occupied ? baseColor : baseColor * 0.2f;
}

#endif // IRCACHE_LOOKUP_HLSL
