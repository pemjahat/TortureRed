// IrCache_Pool_Init.hlsl
// One-shot initialisation.
// Dispatched once at startup: Dispatch( (IRCACHE_TOTAL_CELLS + 63) / 64, 1, 1 )
// i.e. 4096 groups = 262144 threads.
//
// Each thread sweeps:
//   * grid_meta cell  : always (idx < TOTAL_CELLS = 262144)
//   * pool + life entry: only when idx < MAX_ENTRIES = 32768
//   * meta counters   : only thread 0

#include "IrCache_Common.hlsl"

ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;

    // Clear the four meta counters (alloc, live, tracing, pool idx)
    if (idx == 0)
    {
        RWByteAddressBuffer meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];
        meta.Store4(0, uint4(0, 0, 0, 0));
    }

    // Clear every grid cell (single packed uint: entryIdx<<3 | flags)
    if (idx < (uint)IRCACHE_TOTAL_CELLS)
    {
        RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[g_IrCache.GridMetaBufIdx];
        gridMeta.Store(idx * 4, 0u);
    }

    // Initialise free pool (available entry index) and mark entries (life value each entry)
    if (idx < (uint)IRCACHE_MAX_ENTRIES)
    {
        RWStructuredBuffer<uint> pool = ResourceDescriptorHeap[g_IrCache.PoolBufIdx];
        pool[idx] = idx;

        RWByteAddressBuffer life = ResourceDescriptorHeap[g_IrCache.LifeBufIdx];
        life.Store(idx * 4, IRCACHE_ENTRY_LIFE_RECYCLED);
    }
}
