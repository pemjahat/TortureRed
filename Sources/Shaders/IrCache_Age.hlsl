// IrCache_Age.hlsl
// Fixed dispatch: Dispatch( (IRCACHE_MAX_ENTRIES + 63) / 64, 1, 1 )
// No indirect dispatch needed here — we always sweep the full pool range.
//
// Per entry:
//   * RECYCLED  → skip
//   * valid     → increment life
//               → if life >= IRCACHE_ENTRY_LIFE_MAX: expire (return to pool, clear cell)
//               → else: atomically grab a slot in the compact indirection buffer

#include "IrCache_Common.hlsl"

ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint entryIdx = DTid.x;
    if (entryIdx >= (uint)IRCACHE_MAX_ENTRIES)
        return;

    RWByteAddressBuffer life = ResourceDescriptorHeap[g_IrCache.LifeBufIdx];
    uint curLife = life.Load(entryIdx * 4);

    // Skip entries that are already back in the free pool
    if (curLife == IRCACHE_ENTRY_LIFE_RECYCLED)
        return;

    uint newLife = curLife + 1u;

    if (newLife >= (uint)IRCACHE_ENTRY_LIFE_MAX)
    {
        // -------- expire --------
        // 1. Mark as recycled
        life.Store(entryIdx * 4, IRCACHE_ENTRY_LIFE_RECYCLED);

        // 2. Clear the cell (packed entryIdx + flags in one uint)
        RWStructuredBuffer<uint> entryCell = ResourceDescriptorHeap[g_IrCache.EntryCellBufIdx];
        uint cellIdx = entryCell[entryIdx];
        RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[g_IrCache.GridMetaBufIdx];
        gridMeta.Store(cellIdx * 4, 0u);

        // 3. Clear stored irradiance
        RWStructuredBuffer<float4> irradiance = ResourceDescriptorHeap[g_IrCache.IrradianceBufIdx];
        irradiance[entryIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);

        // 4. Return entry to the free pool
        //    Decrement alloc_count first, then write entry_idx into that slot.
        RWByteAddressBuffer meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];
        uint dummy;
        meta.InterlockedAdd(IRCACHE_META_ALLOC_COUNT, uint(-1), dummy);
        // dummy-1 is the new top; write our freed entry there
        if (dummy > 0u)
        {
            RWStructuredBuffer<uint> pool = ResourceDescriptorHeap[g_IrCache.PoolBufIdx];
            pool[dummy - 1u] = entryIdx;
        }
    }
    else
    {
        // -------- still alive --------
        life.Store(entryIdx * 4, newLife);

        // Add this entry to the compact indirection list for this frame's Update pass
        RWByteAddressBuffer meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];
        uint slot;
        meta.InterlockedAdd(IRCACHE_META_COMPACT_WRITE_IDX, 1u, slot);

        RWStructuredBuffer<uint> indirection = ResourceDescriptorHeap[g_IrCache.IndirectionBufIdx];
        indirection[slot] = entryIdx;
    }
}
