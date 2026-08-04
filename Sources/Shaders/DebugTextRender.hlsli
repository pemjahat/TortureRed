// DebugTextRender.hlsli
// GPU on-screen debug text/lines — producer API
//
// Bindless port of D3D12_Research's ShaderDebugRender.hlsli. Any shader that
// receives a DebugRenderContext (via its root constants) can append text
// characters and line instances to the shared render-data buffer; at end of
// frame DebugTextRender.hlsl builds indirect draw args (resetting the counters)
// and rasterizes everything onto the backbuffer.
//
// Usage:
//   DebugRenderContext ctx = { dataUAVIdx, glyphSRVIdx, fontSize, 0 };
//   DebugTextWriter w = CreateDebugTextWriter(ctx, float2(10, 10), float4(1,1,1,1), 1.25f);
//   w.Float(0.812f); w.Char('<'); w.Float(0.795f);
//   DrawDebugRect(ctx, rectMin01, rectMax01, float4(1,0,0,1)); // screen-space [0,1]

#ifndef DEBUG_TEXT_RENDER_HLSLI
#define DEBUG_TEXT_RENDER_HLSLI

#include "Shared/SharedTypes.h"

struct DebugRenderContext
{
    uint  DataUAVIdx;  // RWByteAddressBuffer (render data) bindless index
    uint  GlyphSRVIdx; // StructuredBuffer<DebugGlyph> bindless index
    float FontSize;    // Native font line height in pixels
    uint  _pad;
};

uint PackRGBA8(float4 c)
{
    uint4 u = (uint4)(saturate(c) * 255.0f + 0.5f);
    return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

float4 UnpackRGBA8(uint p)
{
    return float4(p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF, (p >> 24) & 0xFF) / 255.0f;
}

// --- Low-level appenders -----------------------------------------------------

void DebugAddCharacter(DebugRenderContext ctx, uint ch, float2 position, float4 color, float scale)
{
    RWByteAddressBuffer data = ResourceDescriptorHeap[ctx.DataUAVIdx];
    uint slot;
    data.InterlockedAdd(DEBUG_TEXT_COUNTER_OFFSET, 1, slot);
    if (slot >= DEBUG_TEXT_MAX_CHARS)
        return;

    StructuredBuffer<DebugGlyph> glyphs = ResourceDescriptorHeap[ctx.GlyphSRVIdx];
    DebugGlyph g = glyphs[ch];

    DebugCharInstance inst;
    inst.Position  = position + g.Offset * scale;
    inst.Character = ch;
    inst.Scale     = scale;
    inst.Color     = color;
    data.Store(DEBUG_TEXT_INSTANCES_OFFSET + slot * 32, inst);
}

void DebugAddLine(DebugRenderContext ctx, float3 a, float3 b, float4 color, bool screenSpace)
{
    RWByteAddressBuffer data = ResourceDescriptorHeap[ctx.DataUAVIdx];
    uint slot;
    data.InterlockedAdd(DEBUG_LINE_COUNTER_OFFSET, 1, slot);
    if (slot >= DEBUG_TEXT_MAX_LINES)
        return;

    DebugLineInstance inst;
    inst.A = a;
    inst.B = b;
    inst.ColorA = PackRGBA8(color);
    inst.ColorB = inst.ColorA;
    inst.ColorA = (inst.ColorA & 0xFFFFFFFEu) | (screenSpace ? 1u : 0u); // LSB = screen-space flag
    data.Store(DEBUG_LINE_INSTANCES_OFFSET + slot * 32, inst);
}

// --- Line helpers ------------------------------------------------------------

void DrawDebugLine(DebugRenderContext ctx, float2 a, float2 b, float4 color)
{
    DebugAddLine(ctx, float3(a, 0.0f), float3(b, 0.0f), color, true);
}

// Screen-space [0,1] rectangle outline — the HZB test-rect drawer
void DrawDebugRect(DebugRenderContext ctx, float2 rectMin, float2 rectMax, float4 color)
{
    DrawDebugLine(ctx, rectMin,                          float2(rectMax.x, rectMin.y), color);
    DrawDebugLine(ctx, float2(rectMax.x, rectMin.y),     rectMax,                      color);
    DrawDebugLine(ctx, rectMax,                          float2(rectMin.x, rectMax.y), color);
    DrawDebugLine(ctx, float2(rectMin.x, rectMax.y),     rectMin,                      color);
}

// --- Text writer -------------------------------------------------------------

struct DebugTextWriter
{
    DebugRenderContext Ctx;
    float2 StartLocation;
    float2 Cursor;
    float4 Color;
    float  Scale;

    void SetColor(float4 color) { Color = color; }

    void NewLine()
    {
        Cursor.y += Ctx.FontSize * Scale;
        Cursor.x  = StartLocation.x;
    }

    void Char(uint ch)
    {
        StructuredBuffer<DebugGlyph> glyphs = ResourceDescriptorHeap[Ctx.GlyphSRVIdx];
        DebugGlyph g = glyphs[ch];
        DebugAddCharacter(Ctx, ch, Cursor + g.Offset * Scale, Color, Scale);
        Cursor.x += g.AdvanceX * Scale;
    }

    void Int(int value)
    {
        if (value < 0)
        {
            Char('-');
            value = -value;
        }
        uint length  = value > 0 ? (uint)log10((float)value) + 1 : 1;
        uint divider = (uint)round(pow(10.0f, (float)(length - 1)));
        while (length > 0)
        {
            uint digit = (uint)(value / (int)divider);
            Char('0' + digit);
            --length;
            value   -= (int)(digit * divider);
            divider /= 10;
        }
    }

    void Float(float value)
    {
        if (isnan(value))
        {
            Char('N'); Char('a'); Char('N');
        }
        else if (!isfinite(value))
        {
            Char('I'); Char('N'); Char('F');
        }
        else
        {
            if (value < 0.0f)
                Char('-');
            float v = abs(value);
            Int((int)floor(v));
            Char('.');
            float f = frac(v);
            [unroll]
            for (int i = 0; i < 4; ++i)
            {
                f *= 10.0f;
                uint digit = (uint)floor(f);
                Char('0' + digit);
                f -= (float)digit;
            }
        }
    }
    void Float2(float2 v)
    {
        Float(v.x); Char(','); Char(' '); Float(v.y);
    }
    void Float3(float3 v)
    {
        Float(v.x); Char(','); Char(' ');
        Float(v.y); Char(','); Char(' ');
        Float(v.z);
    }
};

DebugTextWriter CreateDebugTextWriter(DebugRenderContext ctx, float2 position, float4 color, float scale)
{
    DebugTextWriter w;
    w.Ctx           = ctx;
    w.StartLocation = position;
    w.Cursor        = position;
    w.Color         = color;
    w.Scale         = scale;
    return w;
}

#endif // DEBUG_TEXT_RENDER_HLSLI
