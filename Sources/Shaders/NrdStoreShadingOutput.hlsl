// NrdStoreShadingOutput.hlsl
// Generic 2-input / 2-output shader called once per active lighting source (DI, GI).
// Bridges each source's Diffuse+Specular intermediate pair into FinalDiffuse/FinalSpecular.
// The shader has zero knowledge of which source (DI or GI) is calling it.
//
// Call 1 (RestirDI, isFirstPass=1): overwrites FinalDiffuseTex / FinalSpecularTex.
// Call 2 (RestirGI, isFirstPass=0): loads prior Final content and additively blends.
//   If DI was not active this frame, GI calls with isFirstPass=1 (overwrite, no stale blend).
//
// Inputs  (per-call, Texture2D<float4>: raw NRD-normalized radiance + hitT):
//   InputIdx0 = SourceDiffuseIntermediate
//   InputIdx1 = SourceSpecularIntermediate
//
// Outputs (RWTexture2D<float4>: raw radiance + hitT, NOT RELAX-packed):
//   OutputIdx0 = FinalDiffuseTex
//   OutputIdx1 = FinalSpecularTex
//
// cbuffer StoreShadingOutputCB (b2):
//   isFirstPass — 1 = overwrite (DI base), 0 = load + additive blend

#include "Common.hlsl"

Texture2D g_Textures[] : register(t0, space0);
ConstantBuffer<FrameConstants>  FrameCB   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

cbuffer StoreShadingOutputCB : register(b2)
{
    uint isFirstPass;   // 1 = overwrite, 0 = load + add
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Inline luminance — avoids pulling in CommonTracing.hlsl (which has ray tracing resources)
static float Luminance(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos  = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y)
        return;

    Texture2D<float4>   srcDiffuseTex  = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4>   srcSpecularTex = ResourceDescriptorHeap[g_Indices.InputIdx1];
    RWTexture2D<float4> noiseDiffuse   = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> noiseSpecular  = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    // Sky pixels: write zero and exit (both overwrite and additive paths)
    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        if (isFirstPass)
        {
            noiseDiffuse[screenPos]  = float4(0.0f, 0.0f, 0.0f, 0.0f);
            noiseSpecular[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        // On additive pass, sky pixels in the prior content are already zero — no write needed.
        return;
    }

    float4 srcDiffuse  = srcDiffuseTex.Load(int3(screenPos, 0));
    float4 srcSpecular = srcSpecularTex.Load(int3(screenPos, 0));

    if (isFirstPass)
    {
        // Overwrite: this source sets the base
        noiseDiffuse[screenPos]  = srcDiffuse;
        noiseSpecular[screenPos] = srcSpecular;
    }
    else
    {
        // Additive blend: load prior signal and add this source's contribution
        float4 priorDiffuse  = noiseDiffuse[screenPos];
        float4 priorSpecular = noiseSpecular[screenPos];

        float3 outDiffuse  = priorDiffuse.rgb  + srcDiffuse.rgb;
        float3 outSpecular = priorSpecular.rgb + srcSpecular.rgb;

        // hitT selection: keep the brighter contributor's hitT
        float priorDiffuseHitT  = priorDiffuse.a;
        float priorSpecularHitT = priorSpecular.a;
        if (Luminance(srcDiffuse.rgb) > Luminance(priorDiffuse.rgb))
            priorDiffuseHitT  = srcDiffuse.a;
        if (Luminance(srcSpecular.rgb) > Luminance(priorSpecular.rgb))
            priorSpecularHitT = srcSpecular.a;

        noiseDiffuse[screenPos]  = float4(outDiffuse,  priorDiffuseHitT);
        noiseSpecular[screenPos] = float4(outSpecular, priorSpecularHitT);
    }
}
