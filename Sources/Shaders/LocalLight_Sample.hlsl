// LocalLight_Sample.hlsl — Stochastic local light specular sampling pass
// Samples one local light per pixel via LUT-based PDF, evaluates specular BRDF,
// fires one shadow ray, and stores the result as a local light reservoir.
// The reservoir is later spatially reused and merged into the RTR specular stream.

#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  g_Frame   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

// Specular-only target PDF (same as RTR specular stream)
float GetSpecularTargetPDF(Surface s, float3 samplePos, float3 sampleRadiance)
{
    float3 L = normalize(samplePos - s.worldPos);
    float dotNL = max(0.0f, dot(s.normal, L));
    if (dotNL <= 0) return 0;
    float3 d, spec;
    EvaluateBSDF(s.normal, s.viewDir, L, s.albedo, s.metallic, s.roughness, d, spec);
    float3 reflected = spec * sampleRadiance * dotNL;
    return max(0.0f, Luminance(reflected));
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    // OutputIdx0 = current local light reservoirs (UAV)
    RWStructuredBuffer<Reservoir> localLightReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    // Early exit: no local lights
    if (g_Frame.numLights <= 1)
    {
        localLightReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex + 31u); // Unique seed offset for local light pass

    // Reconstruct primary surface from GBuffer
    float depth = g_Textures[g_Frame.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        localLightReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    float3 albedo = g_Textures[g_Frame.albedoIndex].Load(int3(screenPos, 0)).rgb;
    float4 packedNormal = g_Textures[g_Frame.normalIndex].Load(int3(screenPos, 0));
    float4 packedMaterial = g_Textures[g_Frame.materialIndex].Load(int3(screenPos, 0));

    float2 uv = (float2(screenPos) + 0.5f) / float2(launchDims);
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, g_Frame.projectionInverse);
    viewPos /= max(viewPos.w, 1e-6f);
    float4 worldPos = mul(viewPos, g_Frame.viewInverse);

    Surface surface;
    surface.worldPos = worldPos.xyz;
    surface.normal = normalize(packedNormal.xyz * 2.0f - 1.0f);
    surface.viewDir = normalize(g_Frame.cameraPosition.xyz - worldPos.xyz);
    surface.albedo = albedo;
    surface.metallic = packedMaterial.g;
    surface.roughness = max(0.01f, packedMaterial.r);

    // Sample one local light using LUT-based PDF
    LightSampleResult lightSample = SampleSingleLight(next_float(rng), g_Lights, g_Frame.numLights, g_Frame);

    if (lightSample.pdf <= 0.0f)
    {
        localLightReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    LightConstants light = lightSample.light;

    // Compute light direction and attenuation
    float3 toLight = light.position.xyz - surface.worldPos;
    float dist = length(toLight);
    float3 L = toLight / max(dist, 1e-6f);

    float NdotL = dot(surface.normal, L);
    if (NdotL <= 0.0f)
    {
        localLightReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    // Evaluate specular BRDF only
    float3 diffBRDF, specBRDF;
    EvaluateBSDF(surface.normal, surface.viewDir, L, surface.albedo, surface.metallic, surface.roughness, diffBRDF, specBRDF);

    if (Luminance(specBRDF) <= 0.0f)
    {
        localLightReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    // Attenuation
    float attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);

    // Spot light cone
    float cosAngle = dot(-L, normalize(light.direction.xyz));
    float cosOuter = light.direction.w;
    float cosInner = asfloat(light.padding[0]);
    float spotEffect = smoothstep(cosOuter, cosInner, cosAngle);

    // Fire one shadow ray
    RayDesc shadowRay;
    shadowRay.Origin = surface.worldPos + surface.normal * 0.001f;
    shadowRay.Direction = L;
    shadowRay.TMin = 0.001f;
    shadowRay.TMax = dist - 0.002f;

    RayQuery<RAY_FLAG_NONE> sq;
    sq.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
    while (sq.Proceed()) {
        PROCESS_ALPHA_MASK(sq, rng);
    }

    if (sq.CommittedStatus() != COMMITTED_NOTHING)
    {
        // Occluded
        localLightReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    // Compute light radiance at surface
    float3 lightRadiance = light.color.rgb * light.intensity * attenuation * spotEffect;

    // Store as reservoir: hitPos = light position, hitNormal = -L (toward surface),
    // radiance = light radiance (unweighted by BRDF — BRDF is applied at resolve time)
    Reservoir r = (Reservoir)0;
    float3 hitPos = light.position.xyz;
    float3 hitNormal = -L; // Normal pointing back toward the surface

    // Target PDF for specular: luminance(specBRDF * radiance * NdotL)
    float targetPDF = max(0.0f, Luminance(specBRDF * lightRadiance * NdotL));
    float risWeight = targetPDF / max(lightSample.pdf, 1e-6f);

    if (risWeight > 0.0f)
    {
        r.hitPos = hitPos;
        r.hitNormal = hitNormal;
        r.radiance = lightRadiance;
        r.w_sum = risWeight;
        r.M = 1.0f;
        r.firstBounceHitT = dist; // Distance to light for NRD hit-distance guidance
        r.W = risWeight / max(targetPDF, 1e-6f); // W = w_sum / (M * targetPDF)
        r.historyAge = ReservoirPackAge(0u, true); // Specular lobe
    }

    localLightReservoirs[pixelIndex] = r;
}
