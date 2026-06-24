// RestirDI_SplitShade.hlsl — ReSTIR DI Split Shade Pass
// Reads the final DI reservoir (post-spatial), traces one shadow ray,
// evaluates BSDF split into diffuse and specular lobes, normalizes by
// NRD material factors, and writes to DIDiffuseIntermediate + DISpecularIntermediate.
//
// Outputs raw float4(normalizedRadiance.rgb, hitT) — NOT RELAX-packed.
// The merge pass (NrdMergeSignals.hlsl) reads these directly and packs after additive blend.
//
// Local lights only: the main directional light (index 0) is handled analytically
// in Lighting.hlsl and is never included in the DI reservoir pool.

#include "CommonTracing.hlsl"
#include "NRD.hlsli"

ConstantBuffer<FrameConstants>  g_Frame   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

// Reconstruct surface from G-Buffer at the given screen position.
bool ReconstructGBufferSurface(uint2 screenPos, uint2 dims, out Surface surf)
{
    float2 uv = (float2(screenPos) + 0.5f) / float2(dims);

    Texture2D<float4> albedoTex   = ResourceDescriptorHeap[g_Frame.albedoIndex];
    Texture2D<float4> normalTex   = ResourceDescriptorHeap[g_Frame.normalIndex];
    Texture2D<float4> materialTex = ResourceDescriptorHeap[g_Frame.materialIndex];
    Texture2D<float>  depthTex    = ResourceDescriptorHeap[g_Frame.depthIndex];

    float depth = depthTex.SampleLevel(g_LinearSampler, uv, 0).r;
    if (depth <= 0.0f || depth >= 1.0f) { surf = (Surface)0; return false; }

    float4 albedo   = albedoTex.SampleLevel(g_LinearSampler, uv, 0);
    float3 normalWS = normalTex.SampleLevel(g_LinearSampler, uv, 0).rgb * 2.0f - 1.0f;
    float4 material = materialTex.SampleLevel(g_LinearSampler, uv, 0);

    float4 ndc     = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, g_Frame.projectionInverse);
    viewPos /= viewPos.w;
    float3 worldPos = mul(viewPos, g_Frame.viewInverse).xyz;

    surf.worldPos  = worldPos;
    surf.normal    = normalize(normalWS);
    surf.viewDir   = normalize(g_Frame.cameraPosition.xyz - worldPos);
    surf.albedo    = albedo.rgb;
    surf.roughness = max(0.01f, material.r);
    surf.metallic  = material.g;
    return true;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 dims      = uint2(g_Frame.screenWidth, g_Frame.screenHeight);
    if (screenPos.x >= dims.x || screenPos.y >= dims.y) return;

    uint pixelIdx = screenPos.y * dims.x + screenPos.x;

    // InputIdx0 = DIReservoirIntermediate (post-spatial)
    // OutputIdx0 = DIDiffuseIntermediate
    // OutputIdx1 = DISpecularIntermediate
    StructuredBuffer<DIRreservoir> intermediate  = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWTexture2D<float4>            diffuseOut    = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4>            specularOut   = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    DIRreservoir res = intermediate[pixelIdx];

    // Zero output for invalid reservoirs or sky pixels
    if (res.M <= 0.0f || res.W <= 0.0f || res.selectedLightIndex == 0u || res.selectedLightIndex >= g_Frame.numLights)
    {
        diffuseOut[screenPos]  = 0.0f.xxxx;
        specularOut[screenPos] = 0.0f.xxxx;
        return;
    }

    Surface surf;
    if (!ReconstructGBufferSurface(screenPos, dims, surf))
    {
        diffuseOut[screenPos]  = 0.0f.xxxx;
        specularOut[screenPos] = 0.0f.xxxx;
        return;
    }

    LightConstants winner = g_Lights[res.selectedLightIndex];

    // Compute light direction and distance
    float3 L;
    float  lightDistance = 0.0f;
    float  attenuation   = 1.0f;
    float  spotEffect    = 1.0f;
    float  tmax          = 10000.0f;

    if (winner.direction.w < 0.5f)
    {
        // Directional light — should not appear in DI reservoir (local lights only),
        // but guard against it gracefully.
        diffuseOut[screenPos]  = 0.0f.xxxx;
        specularOut[screenPos] = 0.0f.xxxx;
        return;
    }

    // Point / Spot light
    float3 diff    = winner.position.xyz - surf.worldPos;
    float  dist    = length(diff);
    if (dist < 0.0001f)
    {
        diffuseOut[screenPos]  = 0.0f.xxxx;
        specularOut[screenPos] = 0.0f.xxxx;
        return;
    }
    L             = diff / dist;
    lightDistance = dist;
    attenuation   = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
    float cosAngle = dot(-L, normalize(winner.direction.xyz));
    float cosOuter = winner.direction.w;
    float cosInner = asfloat(winner.padding[0]);
    spotEffect     = smoothstep(cosOuter, cosInner, cosAngle);
    tmax           = dist - 0.002f;

    float NdotL = dot(surf.normal, L);
    if (NdotL <= 0.0f)
    {
        diffuseOut[screenPos]  = 0.0f.xxxx;
        specularOut[screenPos] = 0.0f.xxxx;
        return;
    }

    // Shadow ray
    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex + 7u);

    RayDesc sr;
    sr.Origin    = surf.worldPos + surf.normal * 0.001f;
    sr.Direction = L;
    sr.TMin      = 0.001f;
    sr.TMax      = tmax;

    RayQuery<RAY_FLAG_NONE> sq;
    sq.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sr);
    while (sq.Proceed()) {
        PROCESS_ALPHA_MASK(sq, rng);
    }
    if (sq.CommittedStatus() != COMMITTED_NOTHING)
    {
        diffuseOut[screenPos]  = 0.0f.xxxx;
        specularOut[screenPos] = 0.0f.xxxx;
        return;
    }

    // Evaluate BSDF split into diffuse and specular lobes
    float3 diffuseBRDF, specularBRDF;
    EvaluateBSDF(surf.normal, surf.viewDir, L, surf.albedo, surf.metallic, surf.roughness, diffuseBRDF, specularBRDF);

    // Incoming radiance from the light (unweighted)
    float3 lightRadiance = winner.color.rgb * winner.intensity * attenuation * spotEffect;

    // Weighted radiance = BRDF * radiance * NdotL * W (unbiased RIS weight)
    float3 diffuseWeighted  = diffuseBRDF  * lightRadiance * NdotL * res.W;
    float3 specularWeighted = specularBRDF * lightRadiance * NdotL * res.W;

    // NRD material factor normalization: divide by the same factors used in PackSignals
    // so that NRD operates on a BRDF-independent signal space.
    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), surf.albedo, surf.metallic);
    float3 diffuseFactor, specularFactor;
    NRD_MaterialFactors(surf.normal, surf.viewDir, surf.albedo, F0, surf.roughness, diffuseFactor, specularFactor);

    float3 diffuseNormalized  = diffuseWeighted  / max(diffuseFactor,  1e-4f.xxx);
    float3 specularNormalized = specularWeighted / max(specularFactor, 1e-4f.xxx);

    // Write raw NRD-normalized radiance + hit distance.
    // The merge pass (NrdMergeSignals.hlsl) reads these directly and additively
    // combines them with GI before packing into RELAX front-end format.
    diffuseOut[screenPos]  = float4(diffuseNormalized,  lightDistance);
    specularOut[screenPos] = float4(specularNormalized, lightDistance);
}
