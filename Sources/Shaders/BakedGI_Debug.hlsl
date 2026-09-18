// =============================================================================
// BakedGI_Debug.hlsl — probe placement debug overlay.
//
// Post-composite LDR overlay: drawn after the TAA resolve/tonemap directly
// onto the backbuffer at output resolution — display-referred colors with no
// taaEnabled branching and no exposure/tonemap interaction, and the temporal
// history stays unpolluted by overlay pixels. Occlusion is a manual reverse-Z
// compare against the GBuffer depth SRV (the internal-res DSV cannot be bound
// at output resolution under TAAU). One instanced unit cube per grid cell,
// generated from vertex-ID/instance-ID semantics (no vertex buffers): the
// cell index comes from SV_InstanceID (SV_VertexID restarts per instance).
// Green = valid (free space), red = invalid (inside geometry).
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// BakedGI.hlsli references FrameCB — include after the declaration above.
#include "BakedGI.hlsli"

struct VSOut
{
    float4 pos   : SV_POSITION;
    float4 color : COLOR;
};

// 12 triangles of a [-1,1]^3 cube.
static const float3 kCubeTris[36] =
{
    // +X
    float3( 1, -1, -1), float3( 1,  1, -1), float3( 1,  1,  1),
    float3( 1, -1, -1), float3( 1,  1,  1), float3( 1, -1,  1),
    // -X
    float3(-1, -1,  1), float3(-1,  1,  1), float3(-1,  1, -1),
    float3(-1, -1,  1), float3(-1,  1, -1), float3(-1, -1, -1),
    // +Y
    float3(-1,  1, -1), float3(-1,  1,  1), float3( 1,  1,  1),
    float3(-1,  1, -1), float3( 1,  1,  1), float3( 1,  1, -1),
    // -Y
    float3(-1, -1,  1), float3(-1, -1, -1), float3( 1, -1, -1),
    float3(-1, -1,  1), float3( 1, -1, -1), float3( 1, -1,  1),
    // +Z
    float3(-1, -1,  1), float3( 1, -1,  1), float3( 1,  1,  1),
    float3(-1, -1,  1), float3( 1,  1,  1), float3(-1,  1,  1),
    // -Z
    float3( 1, -1, -1), float3(-1, -1, -1), float3(-1,  1, -1),
    float3( 1, -1, -1), float3(-1,  1, -1), float3( 1,  1, -1),
};

VSOut VSMain(uint vid : SV_VertexID, uint instance : SV_InstanceID)
{
    VSOut o;

    uint corner = vid % 36u;

    uint3 dims = uint3(FrameCB.bakedGIDimX, FrameCB.bakedGIDimY, FrameCB.bakedGIDimZ);
    uint3 cell = uint3(instance % dims.x, (instance / dims.x) % dims.y, instance / (dims.x * dims.y));
    float3 gridMin = float3(FrameCB.bakedGIGridMinX, FrameCB.bakedGIGridMinY, FrameCB.bakedGIGridMinZ);

    float3 world = gridMin + float3(cell) * FrameCB.bakedGISpacing;
    float  halfSize = FrameCB.bakedGISpacing * 0.04f;
    world += kCubeTris[corner] * halfSize;

    // Unjittered matrix — post-composite pixels are unjittered; the jittered
    // viewProj would make the cubes wobble with the Halton sequence.
    o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);

    StructuredBuffer<uint> meta = ResourceDescriptorHeap[FrameCB.bakedGIProbeMetaSRVIndex];
    o.color = (meta[instance] & 1u)
        ? float4(0.1f, 0.9f, 0.2f, 1.0f)   // valid (free space)
        : float4(0.95f, 0.15f, 0.15f, 1.0f); // invalid (inside geometry)

    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    // Manual reverse-Z occlusion against the GBuffer (same semantics as the
    // old DSV GREATER_EQUAL test): map this output-res pixel to its
    // internal-res texel the way NaiveTsr_Resolve does. NDC z is unaffected
    // by the xy jitter, so the comparison is exact.
    Texture2D<float> depthTex = ResourceDescriptorHeap[FrameCB.depthIndex];
    float2 scale = float2(FrameCB.internalWidth, FrameCB.internalHeight)
                 / float2(FrameCB.outputWidth, FrameCB.outputHeight);
    int2 ip = min(int2(i.pos.xy * scale),
                  int2(FrameCB.internalWidth, FrameCB.internalHeight) - 1);
    float sceneZ = depthTex[ip];
    clip(i.pos.z - sceneZ + 1e-5f); // visible when fragZ >= sceneZ (reverse-Z)

    return i.color;
}
