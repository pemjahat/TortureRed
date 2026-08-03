// HZBDebugView.hlsl
// HZB mip-chain viewer — task007 mode 2 (docs/task007-hzbdebugmodes.md).
//
// Visualizes one HZB mip as grayscale into FullScreenDebugTex; the existing
// FullScreenDebugPS then blits that texture to screen (replacing the lighting
// pass when the mode is active). Reverse-Z: white = near geometry, black = far/sky.
//
// Nearest-neighbor via Load (no sampler state needed) so raw HZB texels are shown
// exactly as stored — filtering would hide min-reduce artifacts this view is meant
// to expose.

#include "Shared/SharedTypes.h"

ConstantBuffer<HZBDebugParams> Params : register(b2); // main root signature param 13

[numthreads(8, 8, 1)]
void HZBDebugViewCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= Params.Width || dispatchThreadID.y >= Params.Height)
        return;

    Texture2D<float>   hzb    = ResourceDescriptorHeap[Params.HZBSRVIdx];
    RWTexture2D<float4> outTex = ResourceDescriptorHeap[Params.OutputUAVIdx];

    uint mipW, mipH, mipCount;
    hzb.GetDimensions(Params.MipLevel, mipW, mipH, mipCount);

    // Map output pixel → HZB mip texel (nearest, aspect-preserving stretch)
    uint2 coord = uint2(
        min(dispatchThreadID.x * mipW / Params.Width,  mipW - 1),
        min(dispatchThreadID.y * mipH / Params.Height, mipH - 1));

    float d = hzb.Load(int3(coord, Params.MipLevel));

    outTex[dispatchThreadID.xy] = float4(d, d, d, 1.0f);
}
