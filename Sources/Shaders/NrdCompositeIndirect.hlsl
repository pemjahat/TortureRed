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

    // InputIdx0 = NrdDenoisedDiffuseTex (NRD output, RELAX-packed)
    // InputIdx1 = NrdDenoisedSpecularTex (NRD output, RELAX-packed)
    // InputIdx2 = NrdValidationTex (optional, UINT(-1) if unused)
    // OutputIdx0 = FinalDiffuseTex  (circular write-back: denoised radiance replaces raw)
    // OutputIdx1 = FinalSpecularTex (circular write-back: denoised radiance replaces raw)
    Texture2D<float4> diffuseIn  = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4> specularIn = ResourceDescriptorHeap[g_Indices.InputIdx1];
    RWTexture2D<float4> diffuseOut  = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> specularOut = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    // Validation overlay: bypass normal unpack and write debug visualization directly.
    if (g_Indices.InputIdx2 != 0xffffffffu)
    {
        Texture2D<float4> validationIn = ResourceDescriptorHeap[g_Indices.InputIdx2];
        float3 validationColor = validationIn.Load(int3(screenPos, 0)).rgb;
        diffuseOut[screenPos]  = float4(validationColor, 1.0f);
        specularOut[screenPos] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth <= 0.0f) // Reverse-Z: sky/clear pixels at far plane (depth = 0.0)
    {
        diffuseOut[screenPos]  = 0.0f.xxxx;
        specularOut[screenPos] = 0.0f.xxxx;
        return;
    }

    // Unpack raw denoised radiance from NRD output.
    // No material factor applied here — re-modulation happens in Lighting.hlsl.
    float3 diffuseRadiance  = RELAX_BackEnd_UnpackRadiance(diffuseIn.Load(int3(screenPos, 0))).rgb;
    float3 specularRadiance = RELAX_BackEnd_UnpackRadiance(specularIn.Load(int3(screenPos, 0))).rgb;

    if (FrameCB.enableIndirectSpecular == 0u)
        specularRadiance = 0.0f.xxx;

    diffuseOut[screenPos]  = float4(min(diffuseRadiance,  65000.0f.xxx), 1.0f);
    specularOut[screenPos] = float4(min(specularRadiance, 65000.0f.xxx), 1.0f);
}
