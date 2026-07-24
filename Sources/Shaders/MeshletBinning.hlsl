#include "MeshletCommon.hlsli"

/*
    Meshlet Binning — 4-pass GPU sort
    Classifies visible meshlets by PSO bin (Opaque / AlphaMasked) and builds
    a sorted indirection list so each bin can be drawn with a single DispatchMesh.

    Pass 1 — PrepareArgsCS:
        Zero MeshletCounts[], GlobalMeshletCounter.
        Build ClassifyDispatchArgs from VisibleMeshletsCounter.

    Pass 2 — ClassifyMeshletsCS (indirect):
        For each visible meshlet, look up material.RasterBin and increment that bin's counter.

    Pass 3 — AllocateBinRangesCS:
        Prefix-sum on MeshletCounts → writes MeshletOffsetAndCounts[bin] = uint4(0, 1, 1, offset).

    Pass 4 — WriteBinsCS (indirect):
        For each visible meshlet, write its index into BinnedMeshlets[] at its bin's offset.

    All params are passed via root constants (b1) as BinningParams.
    Bindless access via ResourceDescriptorHeap[].
*/

ConstantBuffer<BinningParams> gParams : register(b1);

// Indirect dispatch args struct (mirrors D3D12_DISPATCH_ARGUMENTS)
struct DispatchArgs
{
    uint3 ThreadGroupCount;
};

// Helper: get the number of visible meshlets
uint GetNumMeshlets()
{
    StructuredBuffer<uint> counter = ResourceDescriptorHeap[gParams.VisibleMeshletsCounterIdx];
    return counter[0];
}

// Helper: get the PSO bin for a given visible meshlet index
// For now, all meshlets go to Opaque bin (RASTER_BIN_OPAQUE = 0).
// TODO: look up material.RasterBin via InstanceData → MeshData → MaterialBuffer.
uint GetBin(uint meshletIndex)
{
    (void)meshletIndex;
    return RASTER_BIN_OPAQUE;
}

// ---- Pass 1: PrepareArgsCS ----
[numthreads(1, 1, 1)]
void PrepareArgsCS()
{
    RWStructuredBuffer<uint>        meshletCounts = ResourceDescriptorHeap[gParams.RWMeshletCountsIdx];
    RWStructuredBuffer<uint>        globalCounter = ResourceDescriptorHeap[gParams.RWGlobalMeshletCounterIdx];
    RWStructuredBuffer<DispatchArgs> dispatchArgs = ResourceDescriptorHeap[gParams.RWDispatchArgumentsIdx];

    // Zero per-bin counts
    for (uint i = 0; i < gParams.NumBins; ++i)
        meshletCounts[i] = 0;
    globalCounter[0] = 0;

    // Build indirect dispatch args for Classify/Write passes
    uint numMeshlets = GetNumMeshlets();
    DispatchArgs args;
    args.ThreadGroupCount = uint3((numMeshlets + 63) / 64, 1, 1);
    dispatchArgs[0] = args;
}

// ---- Pass 2: ClassifyMeshletsCS ----
[numthreads(64, 1, 1)]
void ClassifyMeshletsCS(uint threadID : SV_DispatchThreadID)
{
    if (threadID >= GetNumMeshlets())
        return;

    RWStructuredBuffer<uint> meshletCounts = ResourceDescriptorHeap[gParams.RWMeshletCountsIdx];

    uint bin = GetBin(threadID);

    // Wave-ops optimized: accumulate counts for threads with the same bin
    bool finished = false;
    while (WaveActiveAnyTrue(!finished))
    {
        if (!finished)
        {
            const uint firstBin = WaveReadLaneFirst(bin);
            if (firstBin == bin)
            {
                uint count = WaveActiveCountBits(true);
                uint originalValue;
                if (WaveIsFirstLane())
                    InterlockedAdd(meshletCounts[firstBin], count, originalValue);
                finished = true;
            }
        }
    }
}

// ---- Pass 3: AllocateBinRangesCS ----
[numthreads(64, 1, 1)]
void AllocateBinRangesCS(uint threadID : SV_DispatchThreadID)
{
    if (threadID >= gParams.NumBins)
        return;

    StructuredBuffer<uint>    meshletCounts          = ResourceDescriptorHeap[gParams.MeshletCountsIdx];
    RWStructuredBuffer<uint4> meshletOffsetAndCounts = ResourceDescriptorHeap[gParams.RWMeshletOffsetAndCountsIdx];
    RWStructuredBuffer<uint>  globalCounter          = ResourceDescriptorHeap[gParams.RWGlobalMeshletCounterIdx];

    uint numMeshlets = meshletCounts[threadID];

    // Prefix sum within wave
    uint offset = WavePrefixSum(numMeshlets);
    uint globalOffset = 0;
    if (WaveIsFirstLane())
    {
        uint waveTotal = WaveActiveSum(numMeshlets);
        InterlockedAdd(globalCounter[0], waveTotal, globalOffset);
    }
    offset += WaveReadLaneFirst(globalOffset);

    // uint4(count=0, 1, 1, offset) — count is filled by WriteBinsCS; (1,1) are Y/Z for DispatchMesh
    meshletOffsetAndCounts[threadID] = uint4(0, 1, 1, offset);
}

// ---- Pass 4: WriteBinsCS ----
[numthreads(64, 1, 1)]
void WriteBinsCS(uint threadID : SV_DispatchThreadID)
{
    if (threadID >= GetNumMeshlets())
        return;

    RWStructuredBuffer<uint4> meshletOffsetAndCounts = ResourceDescriptorHeap[gParams.RWMeshletOffsetAndCountsIdx];
    RWStructuredBuffer<uint>  binnedMeshlets         = ResourceDescriptorHeap[gParams.RWBinnedMeshletsIdx];

    uint bin = GetBin(threadID);
    uint offset = meshletOffsetAndCounts[bin].w;
    uint meshletOffset = 0;

    // Wave-ops optimized: batch-write indices for threads with the same bin
    bool finished = false;
    while (WaveActiveAnyTrue(!finished))
    {
        if (!finished)
        {
            const uint firstBin = WaveReadLaneFirst(bin);
            if (firstBin == bin)
            {
                uint count = WaveActiveCountBits(true);
                uint originalValue;
                if (WaveIsFirstLane())
                    InterlockedAdd(meshletOffsetAndCounts[firstBin].x, count, originalValue);
                meshletOffset = WaveReadLaneFirst(originalValue) + WavePrefixCountBits(true);
                finished = true;
            }
        }
    }

    binnedMeshlets[offset + meshletOffset] = threadID;
}
