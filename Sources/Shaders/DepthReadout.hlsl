// DepthReadout.hlsl
// HZB depth-duel on-screen labels
//
// Producer for the GPU debug text system: reads the mode-1 OccludedRectDebug
// records and emits one small text label per record ("0.8123<0.7945 m6") anchored
// above the rect's top-left corner, color-matched to the rect overlay colors.
// The label shows the exact step-7 duel: nearestDepth < hzbDepth ⇒ occluded.

#include "Shared/SharedTypes.h"
#include "DebugTextRender.hlsli"

ConstantBuffer<DepthReadoutParams> RP : register(b2); // main root signature param 13

[numthreads(64, 1, 1)]
void DepthReadoutCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    StructuredBuffer<uint> counter = ResourceDescriptorHeap[RP.RectsCountSRVIdx];
    uint count = min(min(counter[0], (uint)MAX_OCCLUDED_RECT_DEBUG), RP.MaxLabels);
    if (dispatchThreadID.x >= count)
        return;

    StructuredBuffer<OccludedRectDebug> rects = ResourceDescriptorHeap[RP.RectsSRVIdx];
    OccludedRectDebug rec = rects[dispatchThreadID.x];

    // NDC → backbuffer pixels, label just above the rect's top-left corner
    float2 px = float2(rec.RectMinNDC.x * 0.5f + 0.5f, -rec.RectMinNDC.y * 0.5f + 0.5f)
                * float2(RP.BackbufferWidth, RP.BackbufferHeight);
    px.x = clamp(px.x, 0.0f, RP.BackbufferWidth - 180.0f);
    px.y = clamp(px.y - RP.FontSize * 1.5f, 0.0f, RP.BackbufferHeight - RP.FontSize * 1.5f);

    DebugRenderContext ctx;
    ctx.DataUAVIdx = RP.DataUAVIdx;
    ctx.GlyphSRVIdx = RP.GlyphSRVIdx;
    ctx.FontSize    = RP.FontSize;
    ctx._pad        = 0;

    // Same colors as the rect overlay (OccludedRectDebug.hlsl)
    float4 color = (rec.Kind == 0)
        ? ((rec.Phase == TWO_PASS_PHASE_SECOND) ? float4(1.0f, 0.55f, 0.1f, 1.0f) : float4(1.0f, 0.15f, 0.15f, 1.0f))
        : ((rec.Phase == TWO_PASS_PHASE_SECOND) ? float4(0.2f, 0.8f, 1.0f, 1.0f) : float4(0.75f, 0.25f, 1.0f, 1.0f));

    DebugTextWriter w = CreateDebugTextWriter(ctx, px, color, 1.25f);
    w.Int((int)rec.Mip);
    w.Char(' ');
    w.Float(rec.NearestDepth);
    w.Char('<');
    w.Float(rec.HZBDepth);
    w.Char('m');
}
