// NrdMergeSignals.hlsl — Merge DI + GI signals into NRD noisy input textures.
//
// Replaces NrdPackRasterIndirect.hlsl. Combines:
//   - DI split intermediates (DIDiffuseIntermediate, DISpecularIntermediate):
//     raw float4(normalizedRadiance.rgb, hitT) written by RestirDI_SplitShade.
//   - GI reservoir intermediates (DiffuseReservoirIntermediate, SpecularReservoirIntermediate):
//     evaluated here with per-lobe BRDF and NRD material factor normalization.
//
// Additive merge (mirrors RTXDI StoreShadingOutput isFirstPass=true/false):
//   - DI is "first pass": provides the base radiance and hitT.
//   - GI is "second pass": additively blended; hitT selection keeps the brighter lobe's hitT.
//
// Outputs: NrdNoisyDiffuseTex, NrdNoisySpecularTex (RELAX front-end packed).
//
// Bindings (BindlessIndices):
//   InputIdx0  = DIDiffuseIntermediate  (SRV, raw float4: normalizedRadiance.rgb + hitT)
//   InputIdx1  = DISpecularIntermediate (SRV, raw float4: normalizedRadiance.rgb + hitT)
//   InputIdx2  = DiffuseReservoirIntermediate  (SRV, StructuredBuffer<Reservoir>)
//   OutputIdx2 = SpecularReservoirIntermediate (SRV, repurposed — StructuredBuffer<Reservoir>)
//   OutputIdx0 = NrdNoisyDiffuseTex  (UAV)
//   OutputIdx1 = NrdNoisySpecularTex (UAV)

#include "CommonTracing.hlsl"
#include "NRD.hlsli"

ConstantBuffer<FrameConstants>  FrameCB   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

// BindlessIndices only has 3 input slots; we need 4 inputs.
// Use a second BindlessIndices constant at register b2 for the extra GI inputs.
// The host sets InputIdx0/1 for DI and OutputIdx0/1 for NRD outputs in the main
// BindlessIndices, and passes GI reservoir SRV indices via a second constant.
// To avoid adding a new CB, we repurpose OutputIdx2 and PathVizLineBufferIdx
// as InputIdx3 and InputIdx4 respectively.
//
// Slot mapping (all from g_Indices):
//   InputIdx0  = DIDiffuseIntermediate  SRV (raw float4)
//   InputIdx1  = DISpecularIntermediate SRV (raw float4)
//   InputIdx2  = DiffuseReservoirIntermediate  SRV
//   OutputIdx2 = SpecularReservoirIntermediate SRV  (repurposed)
//   OutputIdx0 = NrdNoisyDiffuseTex  UAV
//   OutputIdx1 = NrdNoisySpecularTex UAV

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos  = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y)
        return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    // DI intermediates: raw float4(normalizedRadiance.rgb, hitT) from RestirDI_SplitShade
    Texture2D<float4> diDiffuseIn  = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4> diSpecularIn = ResourceDescriptorHeap[g_Indices.InputIdx1];

    // GI reservoir intermediates (post-spatial)
    StructuredBuffer<Reservoir> diffuseReservoirs  = ResourceDescriptorHeap[g_Indices.InputIdx2];
    StructuredBuffer<Reservoir> specularReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx2];

    // NRD noisy output textures
    RWTexture2D<float4> diffuseOut  = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> specularOut = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    // ---- Early exit for sky pixels ----
    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        diffuseOut[screenPos]  = RELAX_FrontEnd_PackRadianceAndHitDist(0.0f.xxx, 0.0f, true);
        specularOut[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(0.0f.xxx, 0.0f, true);
        return;
    }

    // ---- Reconstruct surface for GI evaluation ----
    float3 albedo        = g_Textures[FrameCB.albedoIndex].Load(int3(screenPos, 0)).rgb;
    float4 packedNormal  = g_Textures[FrameCB.normalIndex].Load(int3(screenPos, 0));
    float4 packedMaterial= g_Textures[FrameCB.materialIndex].Load(int3(screenPos, 0));

    float2 uv      = (float2(screenPos) + 0.5f) / float2(launchDims);
    float4 ndc     = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= max(viewPos.w, 1e-6f);
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);

    Surface surface;
    surface.worldPos  = worldPos.xyz;
    surface.normal    = normalize(packedNormal.xyz * 2.0f - 1.0f);
    surface.viewDir   = normalize(FrameCB.cameraPosition.xyz - worldPos.xyz);
    surface.albedo    = albedo;
    surface.metallic  = packedMaterial.g;
    surface.roughness = max(0.01f, packedMaterial.r);

    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), surface.albedo, surface.metallic);
    float3 diffuseFactor, specularFactor;
    NRD_MaterialFactors(surface.normal, surface.viewDir, surface.albedo, F0, surface.roughness, diffuseFactor, specularFactor);

    // ---- Step 1: Read DI contribution (isFirstPass = true) ----
    // DI intermediates store raw float4(normalizedRadiance.rgb, hitT) — NOT RELAX-packed.
    float4 diDiffuseRaw  = diDiffuseIn.Load(int3(screenPos, 0));
    float4 diSpecularRaw = diSpecularIn.Load(int3(screenPos, 0));

    float3 diffuseRadiance  = diDiffuseRaw.rgb;
    float3 specularRadiance = diSpecularRaw.rgb;
    float diffuseHitT  = diDiffuseRaw.a;
    float specularHitT = diSpecularRaw.a;

    // ---- Step 2: Evaluate GI contribution and additively blend (isFirstPass = false) ----
    Reservoir rDiffuse  = diffuseReservoirs[pixelIndex];
    Reservoir rSpecular = specularReservoirs[pixelIndex];

    if (rDiffuse.W > 0.0f && rDiffuse.firstBounceHitT > 0.0f)
    {
        float3 L    = normalize(rDiffuse.hitPos - surface.worldPos);
        float NdotL = max(0.0f, dot(surface.normal, L));
        if (NdotL > 0.0f)
        {
            float3 diffuseBRDF, specularBRDF;
            EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffuseBRDF, specularBRDF);
            float3 weightedIncoming = rDiffuse.radiance * (rDiffuse.W * NdotL);
            float3 giDiffuse = diffuseBRDF * weightedIncoming / max(diffuseFactor, 1e-4f.xxx);

            // hitT selection: keep DI's hitT if GI diffuse is not brighter
            // (mirrors RTXDI StoreShadingOutput: use prior hitT when GI lightDistance==0)
            if (Luminance(giDiffuse) > Luminance(diffuseRadiance))
                diffuseHitT = rDiffuse.firstBounceHitT;

            diffuseRadiance += giDiffuse;
        }
    }

    if (FrameCB.enableIndirectSpecular != 0u && rSpecular.W > 0.0f && rSpecular.firstBounceHitT > 0.0f)
    {
        float3 L    = normalize(rSpecular.hitPos - surface.worldPos);
        float NdotL = max(0.0f, dot(surface.normal, L));
        if (NdotL > 0.0f)
        {
            float3 diffuseBRDF, specularBRDF;
            EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffuseBRDF, specularBRDF);
            float3 weightedIncoming = rSpecular.radiance * (rSpecular.W * NdotL);
            float3 giSpecular = specularBRDF * weightedIncoming / max(specularFactor, 1e-4f.xxx);

            // hitT selection: keep DI's hitT if GI specular is not brighter
            if (Luminance(giSpecular) > Luminance(specularRadiance))
                specularHitT = rSpecular.firstBounceHitT;

            specularRadiance += giSpecular;
        }
    }

    // ---- Step 3: Pack merged signal for NRD RELAX ----
    diffuseOut[screenPos]  = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseRadiance,  diffuseHitT,  true);
    specularOut[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(specularRadiance, specularHitT, true);
}
