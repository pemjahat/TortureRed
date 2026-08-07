// DebugTextRender.hlsl
// GPU on-screen debug text/lines — consumers
//
//   DebugTextBuildArgsCS — 1-thread CS: reads the text/line counters, RESETS them
//                          (self-clearing), and writes two D3D12_DRAW_ARGUMENTS.
//   DebugGlyphVS/PS      — instanced glyph quads (4 verts/instance, triangle strip,
//                          no vertex buffer), font atlas × color, onto the backbuffer.
//   DebugLineVS/PS       — 2 verts per line instance; screen-space [0,1] lines map
//                          straight to clip, world-space lines go through FrameCB.viewProj.
//
// All resources are bindless (ResourceDescriptorHeap) via DebugTextRenderParams at b1
// (main root signature param 12). Raster PSOs: alpha blend, no depth, R8G8B8A8_UNORM.

#include "Shared/SharedTypes.h"
#include "DebugTextRender.hlsli"

ConstantBuffer<FrameConstants>        FrameCB : register(b0);
ConstantBuffer<DebugTextRenderParams> Params  : register(b1); // main root signature param 12

SamplerState g_Sampler : register(s0); // main root signature static sampler (linear clamp)

// --- Indirect draw args builder ------------------------------------------------

[numthreads(1, 1, 1)]
void DebugTextBuildArgsCS()
{
    RWByteAddressBuffer data = ResourceDescriptorHeap[Params.DataUAVIdx];
    RWStructuredBuffer<uint4> args = ResourceDescriptorHeap[Params.ArgsUAVIdx];

    uint numChars = data.Load(DEBUG_TEXT_COUNTER_OFFSET);
    data.Store(DEBUG_TEXT_COUNTER_OFFSET, 0u);
    uint numLines = data.Load(DEBUG_LINE_COUNTER_OFFSET);
    data.Store(DEBUG_LINE_COUNTER_OFFSET, 0u);

    // D3D12_DRAW_ARGUMENTS = { VertexCountPerInstance, InstanceCount, StartVertex, StartInstance }
    args[0] = uint4(4u, min(numChars, (uint)DEBUG_TEXT_MAX_CHARS), 0u, 0u); // glyph quads (strip)
    args[1] = uint4(2u, min(numLines, (uint)DEBUG_TEXT_MAX_LINES), 0u, 0u); // lines (line list)
}

// --- Glyph rasterization ---------------------------------------------------------

void DebugGlyphVS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID,
                  out float4 outPosition : SV_Position, out float2 outUV : TEXCOORD0, out float4 outColor : COLOR0)
{
    ByteAddressBuffer data = ResourceDescriptorHeap[Params.DataSRVIdx];
    uint base = DEBUG_TEXT_INSTANCES_OFFSET + instanceID * 32;

    // Explicit dword-by-dword load — MUST match the Store layout in
    // DebugAddCharacter (DebugTextRender.hlsli).
    float2 posInst   = float2(asfloat(data.Load(base + 0)),
                              asfloat(data.Load(base + 4)));
    uint   character = data.Load(base + 8);
    float  scale     = asfloat(data.Load(base + 12));
    float4 color     = float4(asfloat(data.Load(base + 16)),
                              asfloat(data.Load(base + 20)),
                              asfloat(data.Load(base + 24)),
                              asfloat(data.Load(base + 28)));

    StructuredBuffer<DebugGlyph> glyphs = ResourceDescriptorHeap[Params.GlyphSRVIdx];
    DebugGlyph glyph = glyphs[character];

    // Triangle strip corners: (0,0) (1,0) (0,1) (1,1)
    float2 corner = float2(vertexID & 1, vertexID >> 1);
    float2 pos    = posInst + corner * glyph.Dimensions * scale;
    float2 ndc    = pos / float2(Params.TargetWidth, Params.TargetHeight) * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f);

    outPosition = float4(ndc, 0.0f, 1.0f);
    outUV       = lerp(glyph.MinUV, glyph.MaxUV, corner);
    outColor    = color;
}

float4 DebugGlyphPS(float4 position : SV_Position, float2 uv : TEXCOORD0, float4 color : COLOR0) : SV_Target
{
    Texture2D<float4> atlas = ResourceDescriptorHeap[Params.FontAtlasSRVIdx];
    float a = atlas.SampleLevel(g_Sampler, uv, 0).a;
    return float4(color.rgb, color.a * a);
}

// --- Line rasterization ----------------------------------------------------------

void DebugLineVS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID,
                 out float4 outPosition : SV_Position, out float4 outColor : COLOR0)
{
    ByteAddressBuffer data = ResourceDescriptorHeap[Params.DataSRVIdx];
    DebugLineInstance inst = data.Load<DebugLineInstance>(DEBUG_LINE_INSTANCES_OFFSET + instanceID * 32);

    bool screenSpace = (inst.ColorA & 1u) != 0;
    uint packedColor = (vertexID == 0 ? inst.ColorA : inst.ColorB) & 0xFFFFFFFEu;
    outColor = UnpackRGBA8(packedColor);

    float3 p = (vertexID == 0) ? inst.A : inst.B;
    if (screenSpace)
    {
        float2 ndc = p.xy * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f);
        outPosition = float4(ndc, 0.0f, 1.0f);
    }
    else
    {
        outPosition = mul(float4(p, 1.0f), FrameCB.viewProj);
    }
}

float4 DebugLinePS(float4 position : SV_Position, float4 color : COLOR0) : SV_Target
{
    return color;
}
