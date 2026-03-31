#include "CommonTracing.hlsl"
#include "NRD.hlsli"

ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y)
        return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    // InputIdx0 = diffuse reservoir (post-spatial), InputIdx1 = specular reservoir (post-spatial)
    StructuredBuffer<Reservoir> diffuseReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    StructuredBuffer<Reservoir> specularReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx1];
    RWTexture2D<float4> diffuseOut = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> specularOut = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        diffuseOut[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(0.0f.xxx, 0.0f, true);
        specularOut[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(0.0f.xxx, 0.0f, true);
        return;
    }

    float3 albedo = g_Textures[FrameCB.albedoIndex].Load(int3(screenPos, 0)).rgb;
    float4 packedNormal = g_Textures[FrameCB.normalIndex].Load(int3(screenPos, 0));
    float4 packedMaterial = g_Textures[FrameCB.materialIndex].Load(int3(screenPos, 0));

    float2 uv = (float2(screenPos) + 0.5f) / float2(launchDims);
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= max(viewPos.w, 1e-6f);
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);

    Surface surface;
    surface.worldPos = worldPos.xyz;
    surface.normal = normalize(packedNormal.xyz * 2.0f - 1.0f);
    surface.viewDir = normalize(FrameCB.cameraPosition.xyz - worldPos.xyz);
    surface.albedo = albedo;
    surface.metallic = packedMaterial.g;
    surface.roughness = max(0.01f, packedMaterial.r);

    Reservoir rDiffuse = diffuseReservoirs[pixelIndex];
    Reservoir rSpecular = specularReservoirs[pixelIndex];
    float3 diffuseRadiance = 0.0f;
    float3 specularRadiance = 0.0f;
    float diffuseHitT = 0.0f;
    float specularHitT = 0.0f;

    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), surface.albedo, surface.metallic);
    float3 diffuseFactor, specularFactor;
    NRD_MaterialFactors(surface.normal, surface.viewDir, surface.albedo, F0, surface.roughness, diffuseFactor, specularFactor);

    // Evaluate diffuse reservoir with diffuse BRDF
    if (rDiffuse.W > 0.0f && rDiffuse.firstBounceHitT > 0.0f)
    {
        float3 L = normalize(rDiffuse.hitPos - surface.worldPos);
        float NdotL = max(0.0f, dot(surface.normal, L));
        if (NdotL > 0.0f)
        {
            float3 diffuseBRDF, specularBRDF;
            EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffuseBRDF, specularBRDF);
            float3 weightedIncomingRadiance = rDiffuse.radiance * (rDiffuse.W * NdotL);
            diffuseRadiance = diffuseBRDF * weightedIncomingRadiance / max(diffuseFactor, 1e-4f.xxx);
            diffuseHitT = rDiffuse.firstBounceHitT;
        }
    }

    // Evaluate specular reservoir with specular BRDF
    if (FrameCB.enableIndirectSpecular != 0u && rSpecular.W > 0.0f && rSpecular.firstBounceHitT > 0.0f)
    {
        float3 L = normalize(rSpecular.hitPos - surface.worldPos);
        float NdotL = max(0.0f, dot(surface.normal, L));
        if (NdotL > 0.0f)
        {
            float3 diffuseBRDF, specularBRDF;
            EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffuseBRDF, specularBRDF);
            float3 weightedIncomingRadiance = rSpecular.radiance * (rSpecular.W * NdotL);
            specularRadiance = specularBRDF * weightedIncomingRadiance / max(specularFactor, 1e-4f.xxx);
            specularHitT = rSpecular.firstBounceHitT;
        }
    }

    diffuseOut[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseRadiance, diffuseHitT, true);
    specularOut[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(specularRadiance, specularHitT, true);
}
