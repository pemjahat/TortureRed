// NrdPackNoise.hlsl
// Converts raw float4(radiance, hitT) Final textures into NRD RELAX front-end format.
// This is the only shader that knows about the NRD packing format.
// Keeping this conversion separate from StoreShadingOutput makes SSO NRD-format-agnostic.
//
// Reads:  FinalDiffuseTex   (raw float4: radiance.rgb, hitT)
//         FinalSpecularTex  (raw float4: radiance.rgb, hitT)
// Writes: NrdRelaxDiffuseTex   (RELAX_FrontEnd_PackRadianceAndHitDist)
//         NrdRelaxSpecularTex  (RELAX_FrontEnd_PackRadianceAndHitDist)
//
// Bindings (BindlessIndices):
//   InputIdx0  = FinalDiffuseTex   (SRV)
//   InputIdx1  = FinalSpecularTex  (SRV)
//   OutputIdx0 = NrdRelaxDiffuseTex   (UAV)
//   OutputIdx1 = NrdRelaxSpecularTex  (UAV)

#include "Common.hlsl"
#include "NRD.hlsli"

Texture2D g_Textures[] : register(t0, space0);
ConstantBuffer<FrameConstants>  FrameCB   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos  = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y)
        return;

    Texture2D<float4>   noiseDiffuseTex  = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4>   noiseSpecularTex = ResourceDescriptorHeap[g_Indices.InputIdx1];
    RWTexture2D<float4> relaxDiffuse     = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> relaxSpecular    = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth <= 0.0f) // Reverse-Z: sky/clear pixels at far plane (depth = 0.0)
    {
        relaxDiffuse[screenPos]  = RELAX_FrontEnd_PackRadianceAndHitDist(0.0f.xxx, 0.0f, true);
        relaxSpecular[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(0.0f.xxx, 0.0f, true);
        return;
    }

    float4 nDiff = noiseDiffuseTex.Load(int3(screenPos, 0));
    float4 nSpec = noiseSpecularTex.Load(int3(screenPos, 0));

    // hasHit: true when the pixel has a valid surface and a non-zero hit distance
    bool hasHit = (nDiff.a > 0.0f);

    relaxDiffuse[screenPos]  = RELAX_FrontEnd_PackRadianceAndHitDist(nDiff.rgb, nDiff.a, hasHit);
    relaxSpecular[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(nSpec.rgb, nSpec.a, hasHit);
}
