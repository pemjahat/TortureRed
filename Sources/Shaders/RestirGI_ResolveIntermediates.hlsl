// RestirGI_ResolveIntermediates.hlsl
// Resolves GI reservoir StructuredBuffers into raw float4 intermediate textures.
// Extracts G-buffer reconstruction + BRDF evaluation + NRD material factor normalization
// from the old NrdMergeSignals pass, making GI output format-uniform with DI intermediates.
//
// Reads:  DiffuseReservoirIntermediate  (StructuredBuffer<Reservoir>, post-spatial)
//         SpecularReservoirIntermediate (StructuredBuffer<Reservoir>, post-spatial)
// Writes: GIDiffuseIntermediate  (RWTexture2D<float4>: NRD-normalized radiance, firstBounceHitT)
//         GISpecularIntermediate (RWTexture2D<float4>: NRD-normalized radiance, firstBounceHitT)
//
// Bindings (BindlessIndices):
//   InputIdx0  = DiffuseReservoirIntermediate  (SRV, StructuredBuffer<Reservoir>)
//   InputIdx1  = SpecularReservoirIntermediate (SRV, StructuredBuffer<Reservoir>)
//   OutputIdx0 = GIDiffuseIntermediate  (UAV, RWTexture2D<float4>)
//   OutputIdx1 = GISpecularIntermediate (UAV, RWTexture2D<float4>)

#include "CommonTracing.hlsl"
#include "NRD.hlsli"

ConstantBuffer<FrameConstants>  FrameCB   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos  = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y)
        return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    StructuredBuffer<Reservoir> diffuseReservoirs  = ResourceDescriptorHeap[g_Indices.InputIdx0];
    StructuredBuffer<Reservoir> specularReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx1];
    RWTexture2D<float4> giDiffuseOut  = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> giSpecularOut = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    // Sky pixels: write zero and exit
    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        giDiffuseOut[screenPos]  = float4(0.0f, 0.0f, 0.0f, 0.0f);
        giSpecularOut[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Reconstruct surface from G-buffer
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

    // Evaluate diffuse reservoir
    float3 giDiffuse  = 0.0f;
    float  giDiffuseHitT = 0.0f;

    Reservoir rDiffuse = diffuseReservoirs[pixelIndex];
    if (rDiffuse.W > 0.0f && rDiffuse.firstBounceHitT > 0.0f)
    {
        float3 L    = normalize(rDiffuse.hitPos - surface.worldPos);
        float NdotL = max(0.0f, dot(surface.normal, L));
        if (NdotL > 0.0f)
        {
            float3 diffuseBRDF, specularBRDF;
            EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffuseBRDF, specularBRDF);
            float3 weightedIncoming = rDiffuse.radiance * (rDiffuse.W * NdotL);
            giDiffuse    = diffuseBRDF * weightedIncoming / max(diffuseFactor, 1e-4f.xxx);
            giDiffuseHitT = rDiffuse.firstBounceHitT;
        }
    }

    // Evaluate specular reservoir
    float3 giSpecular  = 0.0f;
    float  giSpecularHitT = 0.0f;

    if (FrameCB.enableIndirectSpecular != 0u)
    {
        Reservoir rSpecular = specularReservoirs[pixelIndex];
        if (rSpecular.W > 0.0f && rSpecular.firstBounceHitT > 0.0f)
        {
            float3 L    = normalize(rSpecular.hitPos - surface.worldPos);
            float NdotL = max(0.0f, dot(surface.normal, L));
            if (NdotL > 0.0f)
            {
                float3 diffuseBRDF, specularBRDF;
                EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffuseBRDF, specularBRDF);
                float3 weightedIncoming = rSpecular.radiance * (rSpecular.W * NdotL);
                giSpecular    = specularBRDF * weightedIncoming / max(specularFactor, 1e-4f.xxx);
                giSpecularHitT = rSpecular.firstBounceHitT;
            }
        }
    }

    giDiffuseOut[screenPos]  = float4(giDiffuse,  giDiffuseHitT);
    giSpecularOut[screenPos] = float4(giSpecular, giSpecularHitT);
}
