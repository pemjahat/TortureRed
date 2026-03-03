// IrCache_Prepare_Age.hlsl
// Single-thread dispatch (1,1,1).
// Resets the compaction write index to 0 before the Age pass runs,
// so the Age pass can use it as a safe atomic slot counter.

#include "IrCache_Common.hlsl"

ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

[numthreads(1, 1, 1)]
void main()
{
    RWByteAddressBuffer meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];

    // Reset the per-frame compaction slot counter
    meta.Store(IRCACHE_META_COMPACT_WRITE_IDX, 0u);
}
