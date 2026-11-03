// =============================================================================
// BakedGI_Dilate.hlsl — bake-time six-neighbor transport dilation.
//
// One thread handles one (probe, record) pair. VALID and FILLED source records
// copy directly. Other records average every in-grid axial donor with data.
// The source and destination tables are always different resources. Iteration
// zero reads the raw final tables, and later iterations read only dilation pairs.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>        FrameCB : register(b0);
ConstantBuffer<BakedGIDilationParams> Dilate  : register(b2);

#include "BakedGI.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint probe  = dtid.x;
    const uint record = dtid.y;
    if (probe >= Dilate.probeCount || record >= Dilate.recordCount)
        return;

    StructuredBuffer<float4>   sourceTable = ResourceDescriptorHeap[Dilate.tableSRVIdx];
    RWStructuredBuffer<float4> destTable   = ResourceDescriptorHeap[Dilate.tableUAVIdx];
    StructuredBuffer<uint>     sourceMeta  = ResourceDescriptorHeap[Dilate.metaSRVIdx];
    RWStructuredBuffer<uint>   destMeta    = ResourceDescriptorHeap[Dilate.metaUAVIdx];

    const uint sourceWord = sourceMeta[probe];
    const uint base = (probe * Dilate.recordCount + record) * BAKED_GI_RESPONSE_F4;

    // Direct assignment preserves every bit of a valid or earlier filled record.
    if (BakedGIProbeHasData(sourceWord))
    {
        [unroll]
        for (uint f = 0; f < BAKED_GI_RESPONSE_F4; ++f)
            destTable[base + f] = sourceTable[base + f];

        if (Dilate.writeMeta != 0u && record == 0u)
            destMeta[probe] = sourceWord;
        return;
    }

    const uint3 dims = uint3(Dilate.dimX, Dilate.dimY, Dilate.dimZ);
    const uint3 cell = uint3(probe % dims.x,
                             (probe / dims.x) % dims.y,
                             probe / (dims.x * dims.y));
    // Fixed order is part of the donor rule. All records and both table types
    // execute this same sequence, so sky and sun use identical donor weights.
    const int3 offsets[6] = {
        int3(-1, 0, 0), int3(1, 0, 0),
        int3(0, -1, 0), int3(0, 1, 0),
        int3(0, 0, -1), int3(0, 0, 1)
    };

    float4 sum[BAKED_GI_RESPONSE_F4] = (float4[BAKED_GI_RESPONSE_F4])0;
    uint donorCount = 0;
    [unroll]
    for (uint n = 0; n < 6; ++n)
    {
        const int3 neighbor = int3(cell) + offsets[n];
        if (any(neighbor < int3(0, 0, 0)) || any(neighbor >= int3(dims)))
            continue;

        const uint donorProbe = BakedGIFlatten(uint3(neighbor), dims);
        if (!BakedGIProbeHasData(sourceMeta[donorProbe]))
            continue;

        ++donorCount;
        const uint donorBase = (donorProbe * Dilate.recordCount + record) * BAKED_GI_RESPONSE_F4;
        [unroll]
        for (uint f = 0; f < BAKED_GI_RESPONSE_F4; ++f)
            sum[f] += sourceTable[donorBase + f];
    }

    [unroll]
    for (uint f = 0; f < BAKED_GI_RESPONSE_F4; ++f)
    {
        // A no-donor destination stays zero. The reciprocal runs only with a
        // positive donor count, so no zero denominator is possible.
        destTable[base + f] = (donorCount != 0u) ? sum[f] / (float)donorCount
                                                  : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (Dilate.writeMeta != 0u && record == 0u)
    {
        uint outputWord = sourceWord & ~BAKED_GI_META_FILLED;
        if (donorCount != 0u)
            outputWord |= BAKED_GI_META_FILLED;
        destMeta[probe] = outputWord;
    }
}
