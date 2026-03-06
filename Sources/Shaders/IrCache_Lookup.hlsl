#ifndef IRCACHE_LOOKUP_HLSL
#define IRCACHE_LOOKUP_HLSL

#include "IrCache_Common.hlsl"

// ---------------------------------------------------------------------------
// SampleIrCache
//   Read the stored mean irradiance for the probe that covers `worldPos`.
//   Returns float3(0) if the cell has no allocated probe yet.
//   Also resets the probe's life counter so the aging pass keeps it alive.
// ---------------------------------------------------------------------------
float3 SampleIrCache(float3 worldPos, IrCacheBindlessIndices ircache, float3 cameraPos)
{
    IrcacheCoord coord   = ws_pos_to_ircache_coord(worldPos, cameraPos);
    uint         cellIdx = coord.cell_idx();

    RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[ircache.GridMetaBufIdx];
    uint2 cellData = gridMeta.Load2(cellIdx * 8);
    uint  flags    = cellData.y;

    if (!(flags & IRCACHE_ENTRY_META_OCCUPIED))
        return float3(0.0f, 0.0f, 0.0f);

    uint entryIdx = cellData.x;

    // Refresh life so the aging pass does not expire this probe
    RWByteAddressBuffer life = ResourceDescriptorHeap[ircache.LifeBufIdx];
    life.Store(entryIdx * 4, 0u);

    RWStructuredBuffer<float4> irradiance = ResourceDescriptorHeap[ircache.IrradianceBufIdx];
    return irradiance[entryIdx].rgb;
}

// ---------------------------------------------------------------------------
// IrCacheMaybeAllocate
//   If the cell at `worldPos` is empty, atomically claim it and pop a free
//   entry from the pool, writing the back-links into grid metadata.
//   Safe to call from many threads simultaneously; only one wins per cell.
// ---------------------------------------------------------------------------
void IrCacheMaybeAllocate(float3 worldPos, IrCacheBindlessIndices ircache, float3 cameraPos)
{
    IrcacheCoord coord   = ws_pos_to_ircache_coord(worldPos, cameraPos);
    uint         cellIdx = coord.cell_idx();

    RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[ircache.GridMetaBufIdx];
    uint2 cellData = gridMeta.Load2(cellIdx * 8);
    uint  flags    = cellData.y;

    // Already occupied by another probe — nothing to do
    if (flags & IRCACHE_ENTRY_META_OCCUPIED)
        return;

    // Race to own this cell (one winner per cell per frame)
    uint prevFlags;
    gridMeta.InterlockedOr(cellIdx * 8 + 4, IRCACHE_ENTRY_META_OCCUPIED, prevFlags);
    if (prevFlags & IRCACHE_ENTRY_META_OCCUPIED)
        return; // Another thread won

    // Pop a free entry from the pool
    RWByteAddressBuffer meta = ResourceDescriptorHeap[ircache.MetaBufIdx];
    uint oldAllocCount;
    meta.InterlockedAdd(IRCACHE_META_ALLOC_COUNT, 1u, oldAllocCount);

    if (oldAllocCount >= (uint)IRCACHE_MAX_ENTRIES)
    {
        // Pool exhausted — revert the alloc count and release the cell claim
        uint dummy;
        meta.InterlockedAdd(IRCACHE_META_ALLOC_COUNT, uint(-1), dummy);
        gridMeta.InterlockedAnd(cellIdx * 8 + 4, ~IRCACHE_ENTRY_META_OCCUPIED, prevFlags);
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

    // Write entry index + JUST_ALLOCATED flag into cell (overwrite the bare OCCUPIED we set above)
    gridMeta.Store2(cellIdx * 8, uint2(entryIdx,
        IRCACHE_ENTRY_META_OCCUPIED | IRCACHE_ENTRY_META_JUST_ALLOCATED));

    // Extend entry_count high-watermark so the Age pass sweeps far enough
    uint dummy;
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
    uint2 cellData = gridMeta.Load2(cellIdx * 8);
    if (!(cellData.y & IRCACHE_ENTRY_META_OCCUPIED))
        return float3(0.05f, 0.05f, 0.05f);    // unoccupied — dark grey

    uint entryIdx = cellData.x;

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
    uint2 cellData = gridMeta.Load2(cellIdx * 8);
    bool  occupied = (cellData.y & IRCACHE_ENTRY_META_OCCUPIED) != 0;

    float3 baseColor = cascadeColors[coord.cascade];
    return occupied ? baseColor : baseColor * 0.2f;
}

#endif // IRCACHE_LOOKUP_HLSL
