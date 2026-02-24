#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// Inputs
StructuredBuffer<Reservoir> g_ResolvedReservoirs : register(t0, space3);

// Outputs
RWTexture2D<float4> g_IndirectLightingTex : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims;
    g_IndirectLightingTex.GetDimensions(launchDims.x, launchDims.y);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;
    
    // Read G-Buffer
    float depth = g_Textures[FrameCB.depthIndex].Sample(g_LinearSampler, screenPos).r;
    if (depth == 1.0f) {
        g_IndirectLightingTex[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float2 uv = ((float2)screenPos + 0.5f) / (float2)launchDims;
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 worldPos4 = mul(ndc, FrameCB.projectionInverse);
    worldPos4 /= worldPos4.w;
    float3 worldPos = mul(worldPos4, FrameCB.viewInverse).xyz;

    float3 normal = g_Textures[FrameCB.normalIndex].Sample(g_LinearSampler, screenPos).rgb * 2.0f - 1.0f;
    float3 albedo = g_Textures[FrameCB.albedoIndex].Sample(g_LinearSampler, screenPos).rgb;
    float4 material = g_Textures[FrameCB.materialIndex].Sample(g_LinearSampler, screenPos);
    float metallic = material.r;
    float roughness = max(0.01f, material.g);
    float3 viewDir = normalize(FrameCB.cameraPosition.xyz - worldPos);

    Reservoir r = g_ResolvedReservoirs[pixelIndex];
    
    float3 indirectLighting = 0.0f;

    if (r.w_sum > 0.0f) {
        float3 L = normalize(r.hitPos - worldPos);
        float NdotL = max(0.0f, dot(normal, L));
        
        if (NdotL > 0.0f) {
            float3 diffBRDF, specBRDF;
            EvaluateBSDF(normal, viewDir, L, albedo, metallic, roughness, diffBRDF, specBRDF);
            
            if (!FrameCB.enableIndirectSpecular) {
                specBRDF = 0.0f;
            }
            
            indirectLighting = (diffBRDF + specBRDF) * r.radiance * r.w_sum * NdotL;
        }
    }

    g_IndirectLightingTex[screenPos] = float4(indirectLighting, 1.0f);
}
