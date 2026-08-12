// FullScreenDebug.hlsl
// Dedicated full-screen debug visualization pass.
// Replaces Lighting.hlsl completely when any debug mode is active.
// Outputs debug data directly to the screen — no NRD material factor modulation,
// no BSDF evaluation, no shadow rays.
//
// Single R16G16B16A16 input via InputIdx0 (FullScreenDebugTex).
// All debug data sources are pre-combined into this texture by upstream passes
// (SHaRC_Debug.hlsl, FullScreenDebugCombine.hlsl).
//
// Debug modes handled upstream (set via FrameConstants):
//   restirReservoirDebugMode 1-4 (field):  FinalDiffuse + FinalSpecular → combine → InputIdx0
//   restirReservoirDebugMode 5-10 (heatmap): RestirDebugHeatmap R16_FLOAT → combine → InputIdx0
//   sharcDebug != 0:                        SHaRC_Debug.hlsl → InputIdx0 directly
//   restirDIDebugMode != OFF:               FinalDiffuse + FinalSpecular → combine → InputIdx0
//
// Inputs: G-Buffer (depth only) + FullScreenDebugTex (InputIdx0, R16G16B16A16_FLOAT)

#include "Common.hlsl"

// Minimal resource declarations — FullScreenDebug does not need RT acceleration
// structures or DrawNode buffers. Only G-Buffer textures and the bindless heap.
Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0);

struct VSInput {
    uint vertexID : SV_VertexID;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    // Fullscreen triangle
    output.texCoord = float2((input.vertexID << 1) & 2, input.vertexID & 2);
    output.position = float4(output.texCoord * 2.0f - 1.0f, 0.0f, 1.0f);
    output.texCoord.y = 1.0f - output.texCoord.y;
    return output;
}

ConstantBuffer<FrameConstants>  FrameCB   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

float4 PSMain(PSInput input) : SV_Target {
    float depth = g_Textures[FrameCB.depthIndex].Sample(g_LinearSampler, input.texCoord).r;

    // Sky pixels — sample the baked sky cubemap for background color.
    // Reverse-Z: clear = 0.0 (far plane).
    if (depth <= 0.0f)
    {
        // Reconstruct camera ray direction (same as GetCameraRayDirection in CommonTracing.hlsl,
        // duplicated here to avoid pulling in the full ray-tracing include).
        float2 uv = input.texCoord;
        float2 ndcXY = float2(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f);
        float4 viewFar = mul(float4(ndcXY, 1.0f, 1.0f), FrameCB.projectionInverse);
        viewFar /= max(abs(viewFar.w), 1e-6f);
        float3 worldFar = mul(viewFar, FrameCB.viewInverse).xyz;
        float3 cameraRayDir = normalize(worldFar - FrameCB.cameraPosition.xyz);

        TextureCube<float4> skyCubemap = ResourceDescriptorHeap[FrameCB.skyCubemapIndex];
        float3 skyColor = skyCubemap.SampleLevel(g_LinearSampler, cameraRayDir, 0.0f).rgb;

        if (FrameCB.taaEnabled)
            return float4(skyColor, 1.0f);
        float3 exposed = skyColor * FrameCB.exposure;
        return float4(exposed / (exposed + 1.0f), 1.0f);
    }

    // Single unified debug input — all debug modes pre-combine into this texture
    Texture2D<float4> debugTex = ResourceDescriptorHeap[g_Indices.InputIdx0];
    float4 debugColor = debugTex.SampleLevel(g_LinearSampler, input.texCoord, 0);

    // ── Output ───────────────────────────────────────────────────────────
    // When TAA is active, output raw HDR — the TAA resolve shader handles
    // exposure and tonemapping. Otherwise, apply them here for direct display.
    if (FrameCB.taaEnabled)
    {
        return debugColor;
    }

    // Basic Tone Mapping
    float3 exposedColor = debugColor.rgb * FrameCB.exposure;
    float3 ldrColor = exposedColor / (exposedColor + 1.0f);

    return float4(ldrColor, 1.0f);
}
