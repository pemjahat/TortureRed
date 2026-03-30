#include "Common.hlsl"
#include "NRD.hlsli"

Texture2D g_Textures[] : register(t0, space0);
ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y)
        return;

    Texture2D<float4> diffuseIn = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4> specularIn = ResourceDescriptorHeap[g_Indices.InputIdx1];
    RWTexture2D<float4> indirectLightingTex = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    if (g_Indices.InputIdx2 != 0xffffffffu)
    {
        Texture2D<float4> validationIn = ResourceDescriptorHeap[g_Indices.InputIdx2];
        indirectLightingTex[screenPos] = float4(validationIn.Load(int3(screenPos, 0)).rgb, 1.0f);
        return;
    }

    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        indirectLightingTex[screenPos] = 0.0f.xxxx;
        return;
    }

    float4 packedNormal = g_Textures[FrameCB.normalIndex].Load(int3(screenPos, 0));
    float4 packedMaterial = g_Textures[FrameCB.materialIndex].Load(int3(screenPos, 0));
    float3 albedo = g_Textures[FrameCB.albedoIndex].Load(int3(screenPos, 0)).rgb;

    float2 uv = (float2(screenPos) + 0.5f) / float2(launchDims);
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= max(viewPos.w, 1e-6f);
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);

    float3 N = normalize(packedNormal.xyz * 2.0f - 1.0f);
    float roughness = max(0.01f, packedMaterial.r);
    float metallic = packedMaterial.g;
    float3 V = normalize(FrameCB.cameraPosition.xyz - worldPos.xyz);
    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic);

    float3 diffuseFactor, specularFactor;
    NRD_MaterialFactors(N, V, albedo, F0, roughness, diffuseFactor, specularFactor);

    float3 diffuseRadiance = RELAX_BackEnd_UnpackRadiance(diffuseIn.Load(int3(screenPos, 0))).rgb;
    float3 specularRadiance = RELAX_BackEnd_UnpackRadiance(specularIn.Load(int3(screenPos, 0))).rgb;
    if (FrameCB.enableIndirectSpecular == 0u)
    {
        specularRadiance = 0.0f.xxx;
    }

    float3 indirectLighting = diffuseRadiance * diffuseFactor + specularRadiance * specularFactor;
    indirectLightingTex[screenPos] = float4(min(indirectLighting, 10.0f), 1.0f);
}
