// OccludedRectDebug.hlsl
// Occluded-rect overlay — task007 mode 1 (docs/task007-hzbdebugmodes.md).
//
// Draws the NDC screen rects of instances/meshlets that HZBCull rejected
// (recorded by MeshletTwoPassCull.hlsl into the OccludedRectDebug buffer)
// over a dimmed scene-albedo background in FullScreenDebugTex; the existing
// FullScreenDebugPS then blits that texture to screen.
//
// Two entry points, dispatched back-to-back by MeshletPass::DrawOccludedRects:
//   OccludedRectBackgroundCS — fullscreen dimmed-albedo background for context
//   OccludedRectsCS          — one thread per record, rasterizes the 4 rect edges
//
// Colors:  meshlet  phase 1 = red,    phase 2 = orange
//          instance phase 1 = purple, phase 2 = cyan

#include "Shared/SharedTypes.h"

ConstantBuffer<FrameConstants>         FrameCB    : register(b0);
ConstantBuffer<OccludedRectDrawParams> DrawParams : register(b2); // main root signature param 13

[numthreads(8, 8, 1)]
void OccludedRectBackgroundCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= DrawParams.Width || dispatchThreadID.y >= DrawParams.Height)
        return;

    Texture2D<float4>    albedo = ResourceDescriptorHeap[(uint)FrameCB.albedoIndex];
    RWTexture2D<float4>  outTex = ResourceDescriptorHeap[DrawParams.OutputUAVIdx];

    float3 scene = albedo.Load(int3(dispatchThreadID.xy, 0)).rgb;
    outTex[dispatchThreadID.xy] = float4(scene * 0.35f, 1.0f);
}

[numthreads(64, 1, 1)]
void OccludedRectsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    StructuredBuffer<uint> counter = ResourceDescriptorHeap[DrawParams.RectsCountSRVIdx];
    if (dispatchThreadID.x >= min(counter[0], (uint)MAX_OCCLUDED_RECT_DEBUG))
        return;

    StructuredBuffer<OccludedRectDebug> rects = ResourceDescriptorHeap[DrawParams.RectsSRVIdx];
    OccludedRectDebug rec = rects[dispatchThreadID.x];

    RWTexture2D<float4> outTex = ResourceDescriptorHeap[DrawParams.OutputUAVIdx];

    // NDC [-1,1] → output pixels (texture-space Y flip), clamped to the target
    float2 fmin = float2(rec.RectMinNDC.x, -rec.RectMaxNDC.y) * 0.5f + 0.5f;
    float2 fmax = float2(rec.RectMaxNDC.x, -rec.RectMinNDC.y) * 0.5f + 0.5f;
    int2 pmin = clamp((int2)(fmin * float2(DrawParams.Width, DrawParams.Height)), int2(0, 0), int2(DrawParams.Width - 1, DrawParams.Height - 1));
    int2 pmax = clamp((int2)(fmax * float2(DrawParams.Width, DrawParams.Height)), int2(0, 0), int2(DrawParams.Width - 1, DrawParams.Height - 1));

    float3 color;
    if (rec.Kind == 0)
        color = (rec.Phase == TWO_PASS_PHASE_SECOND) ? float3(1.0f, 0.55f, 0.1f) : float3(1.0f, 0.15f, 0.15f); // meshlet: orange / red
    else
        color = (rec.Phase == TWO_PASS_PHASE_SECOND) ? float3(0.2f, 0.8f, 1.0f) : float3(0.75f, 0.25f, 1.0f); // instance: cyan / purple

    // Rasterize the 4 edges (write conflicts between threads are benign)
    for (int x = pmin.x; x <= pmax.x; ++x)
    {
        outTex[int2(x, pmin.y)] = float4(color, 1.0f);
        outTex[int2(x, pmax.y)] = float4(color, 1.0f);
    }
    for (int y = pmin.y; y <= pmax.y; ++y)
    {
        outTex[int2(pmin.x, y)] = float4(color, 1.0f);
        outTex[int2(pmax.x, y)] = float4(color, 1.0f);
    }
}
