#include "Common.hlsl"

// Vertex and pixel shaders for rendering path visualization lines.
// The VS reads a small StructuredBuffer<PathVizLine> via the bindless heap
// and converts each line's two endpoints into clip-space positions with a colour.
// Lines whose valid flag is 0 are collapsed to a degenerate off-screen quad.
// Both VS and PS entry points live in this file.

ConstantBuffer<FrameConstants>   g_Frame   : register(b0);
ConstantBuffer<BindlessIndices>  g_Indices : register(b1);

// Colour lookup by path type (index = bits[3:0] of typeAndValid)
static const float3 k_pathColors[8] = {
    float3(1.00f, 0.90f, 0.00f),   // 0: surface normal     (yellow)
    float3(1.00f, 1.00f, 1.00f),   // 1: bounce 1           (white)
    float3(0.40f, 0.90f, 1.00f),   // 2: bounce 2           (cyan)
    float3(0.20f, 0.40f, 1.00f),   // 3: bounce 3           (blue)
    float3(1.00f, 0.50f, 0.00f),   // 4: temporal reuse     (orange)
    float3(1.00f, 0.00f, 1.00f),   // 5: spatial winner     (magenta)
    float3(0.50f, 0.50f, 0.50f),   // 6: (reserved)
    float3(0.50f, 0.50f, 0.50f),   // 7: (reserved)
};

struct VSOutput {
    float4 pos   : SV_Position;
    float3 color : COLOR0;
};

VSOutput VSMain(uint vertexID : SV_VertexID)
{
    VSOutput output;

    StructuredBuffer<PathVizLine> g_PathVizLines = ResourceDescriptorHeap[g_Indices.PathVizLineBufferIdx];

    uint lineIndex = vertexID / 2;   // which line slot
    uint isEnd     = vertexID & 1;   // 0 = start vertex, 1 = end vertex

    PathVizLine debugLine = g_PathVizLines[lineIndex];

    uint type  = debugLine.typeAndValid & 0xFu;
    uint valid = (debugLine.typeAndValid >> 4u) & 1u;

    if (!valid)
    {
        // Degenerate vertex placed far behind the near plane; rasteriser discards it.
        output.pos   = float4(0.0f, 0.0f, -1.0e6f, 1.0f);
        output.color = float3(0.0f, 0.0f, 0.0f);
        return output;
    }

    float3 worldPos = isEnd ? debugLine.end : debugLine.start;
    output.pos   = mul(float4(worldPos, 1.0f), g_Frame.viewProj);
    output.color = k_pathColors[type & 7u];
    return output;
}

// ---------------------------------------------------------------------------
// Pixel shader
// ---------------------------------------------------------------------------

struct PSInput {
    float4 pos   : SV_Position;
    float3 color : COLOR0;
};

float4 PSMain(PSInput input) : SV_Target
{
    return float4(input.color, 1.0f);
}
