// IrCache_Prepare_Trace.hlsl
// Single-thread dispatch (1,1,1).
// Reads the live-entry count from the Age pass result, stores it as the
// tracing snapshot, and writes indirect dispatch args for IrCache_Update:
//   groups = live_count   (one group per probe, IRCACHE_RAYS_PER_PROBE threads/group)

#include "IrCache_Common.hlsl"

ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

[numthreads(1, 1, 1)]
void main()
{
    RWByteAddressBuffer meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];

    // Snapshot the live count written by the Age pass
    uint liveCount = meta.Load(IRCACHE_META_COMPACT_WRITE_IDX);
    meta.Store(IRCACHE_META_TRACING_ALLOC_COUNT, liveCount);

    // Write indirect dispatch args: (liveCount, 1, 1)
    // Each group handles one probe with IRCACHE_RAYS_PER_PROBE threads.
    RWByteAddressBuffer traceArgs = ResourceDescriptorHeap[g_IrCache.TraceArgsBufIdx];
    traceArgs.Store3(0, uint3(liveCount, 1u, 1u));
}
