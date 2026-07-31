// HZB.hlsl
// Hierarchical Z-Buffer (HZB) construction for two-phase occlusion culling.

//
// Two passes, dispatched back-to-back by MeshletPass::BuildHZB():
//   HZBInitCS   — Gather()s a 2x2 depth footprint from the source depth buffer and writes the
//                 conservative value into HZB mip 0.
//   HZBCreateCS — AMD FidelityFX SPD (Single Pass Downsampler, fetched via CMake FetchContent
//                 into ThirdParty/FidelityFX-SPD, see CMakeLists.txt) generates the remaining
//                 mip chain (mip 1..NumMips-1) in a single dispatch.
//
// Reverse-Z convention: TortureRed uses reverse-Z depth (1.0 = near, 0.0 = far, DepthFunc = GREATER_EQUAL,
// clear = 0.0). A conservative occlusion HZB must store the FARTHEST depth per texel (the
// depth of the most-distant occluder in that footprint). In reverse-Z, the farthest depth
// is the SMALLEST value (near=1.0 > far=0.0), so both the init pass and SPD's reduction
// function below use min(), not max() — the opposite of the standard-Z convention.
//
// Resource access: everything (source depth, HZB mip UAVs, SPD atomic counter) is accessed via
// bindless ResourceDescriptorHeap[idx] indices carried in HZBConstants (Shared/SharedTypes.h),
// bound as a root CBV on MeshletPass's dedicated m_HZBRootSignature. Per-mip HZB UAV indices live
// in a small StructuredBuffer<uint> (MipIndicesSRVIdx) rather than an array field on HZBConstants,
// to avoid HLSL cbuffer array packing (16 bytes/element) mismatching the plain C++ struct layout.

#include "Shared/SharedTypes.h"

ConstantBuffer<HZBConstants> HZBCB : register(b0);
SamplerState PointClampSampler : register(s0);

// -----------------------------------------------------------------------------
// HZBInitCS — builds HZB mip 0 from the source depth buffer.
// -----------------------------------------------------------------------------
[numthreads(16, 16, 1)]
void HZBInitCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= HZBCB.Width || dispatchThreadID.y >= HZBCB.Height)
        return;

    Texture2D<float> depthTex = ResourceDescriptorHeap[HZBCB.DepthSRVIdx];
    StructuredBuffer<uint> mipUAVs = ResourceDescriptorHeap[HZBCB.MipIndicesSRVIdx];
    RWTexture2D<float> hzbMip0 = ResourceDescriptorHeap[mipUAVs[0]];

    float2 uv = (float2(dispatchThreadID.xy) + 0.5f) * float2(HZBCB.DimensionsInvX, HZBCB.DimensionsInvY);
    float4 depths = depthTex.Gather(PointClampSampler, uv);

    // Reverse-Z: keep the FARTHEST of the 2x2 footprint (min, because far=0 < near=1) — see file header.
    float minDepth = min(min(depths.x, depths.y), min(depths.z, depths.w));
    hzbMip0[dispatchThreadID.xy] = minDepth;
}

// -----------------------------------------------------------------------------
// HZBCreateCS — SPD mip-chain generation (mips 1..NumMips-1).
// -----------------------------------------------------------------------------
#define A_GPU 1
#define A_HLSL 1
#include "ffx_a.h"

// LDS intermediates required by ffx_spd.h.
groupshared AF1 spdIntermediate[16][16];
groupshared AU1 spdCounter;

// mip 0 was already written by HZBInitCS above — SPD treats it as its source image.
AF4 SpdLoadSourceImage(ASU2 tex, AU1 slice)
{
    StructuredBuffer<uint> mipUAVs = ResourceDescriptorHeap[HZBCB.MipIndicesSRVIdx];
    RWTexture2D<float> mip0 = ResourceDescriptorHeap[mipUAVs[0]];
    return mip0[tex];
}

// Re-loads mip 6 for the final cross-workgroup reduction pass (mips 7..11).
AF4 SpdLoad(ASU2 tex, AU1 slice)
{
    StructuredBuffer<uint> mipUAVs = ResourceDescriptorHeap[HZBCB.MipIndicesSRVIdx];
    RWTexture2D<float> mip6 = ResourceDescriptorHeap[mipUAVs[6]];
    return mip6[tex];
}

// SPD's own "mip" parameter is 0-indexed relative to its first generated level, which is our
// HZB mip 1 (mip 0 already exists). So SPD mip M writes into HZB mip (M+1).
void SpdStore(ASU2 pix, AF4 value, AU1 mip, AU1 slice)
{
    StructuredBuffer<uint> mipUAVs = ResourceDescriptorHeap[HZBCB.MipIndicesSRVIdx];
    RWTexture2D<float> dst = ResourceDescriptorHeap[mipUAVs[mip + 1]];
    dst[pix] = value.x;
}

void SpdIncreaseAtomicCounter(AU1 slice)
{
    RWStructuredBuffer<uint> counter = ResourceDescriptorHeap[HZBCB.SpdCounterUAVIdx];
    InterlockedAdd(counter[0], 1, spdCounter);
}

AU1 SpdGetAtomicCounter() { return spdCounter; }

void SpdResetAtomicCounter(AU1 slice)
{
    RWStructuredBuffer<uint> counter = ResourceDescriptorHeap[HZBCB.SpdCounterUAVIdx];
    counter[0] = 0;
}

AF4 SpdLoadIntermediate(AU1 x, AU1 y) { return spdIntermediate[x][y]; }
void SpdStoreIntermediate(AU1 x, AU1 y, AF4 value) { spdIntermediate[x][y] = value.x; }

// Reverse-Z reduce: keep the FARTHEST of the four inputs (min, because far=0 < near=1).
AF4 SpdReduce4(AF4 v0, AF4 v1, AF4 v2, AF4 v3)
{
    return min(v0, min(v1, min(v2, v3)));
}

#include "ffx_spd.h"

[numthreads(256, 1, 1)]
void HZBCreateCS(uint3 WorkGroupId : SV_GroupID, uint LocalThreadIndex : SV_GroupIndex)
{
    SpdDownsample(
        AU2(WorkGroupId.xy),
        AU1(LocalThreadIndex),
        AU1(HZBCB.NumMips - 1), // SPD's own mip count excludes mip 0 (already written by HZBInitCS)
        AU1(HZBCB.NumWorkGroups),
        AU1(0));                // slice — HZB is a plain Texture2D, no array/cube slices
}
