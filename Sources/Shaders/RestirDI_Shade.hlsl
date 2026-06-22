// RestirDI_Shade.hlsl — ReSTIR DI Pass 4: Shade (Resolve)
// Reads the final reservoir from DIRreservoirIntermediate, traces one shadow ray,
// evaluates BSDF, and writes weighted radiance to DIOutputTex.

#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  g_Frame   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

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

// Evaluate the winning light: one shadow ray + BSDF.
// Returns the unshadowed radiance contribution (caller multiplies by W).
float3 EvaluateDIReservoirWinner(Surface surf, LightConstants light, inout RNG rng)
{
    float3 L;
    float  attenuation = 1.0f;
    float  spotEffect  = 1.0f;
    float  tmax        = 10000.0f;

    if (light.direction.w < 0.5f)
    {
        // Directional
        L = -normalize(light.direction.xyz);
    }
    else
    {
        // Point / Spot
        float3 diff = light.position.xyz - surf.worldPos;
        float  dist = length(diff);
        if (dist < 0.0001f) return 0.0f;
        L           = diff / dist;
        attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
        float cosAngle = dot(-L, normalize(light.direction.xyz));
        float cosOuter = light.direction.w;
        float cosInner = asfloat(light.padding[0]);
        spotEffect = smoothstep(cosOuter, cosInner, cosAngle);
        tmax = dist - 0.002f;
    }

    float NdotL = dot(surf.normal, L);
    if (NdotL <= 0.0f) return 0.0f;

    // Shadow ray
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
    if (sq.CommittedStatus() != COMMITTED_NOTHING) return 0.0f;

    float3 diff, spec;
    EvaluateBSDF(surf.normal, surf.viewDir, L, surf.albedo, surf.metallic, surf.roughness, diff, spec);

    return (diff + spec) * light.color.rgb * light.intensity * NdotL * attenuation * spotEffect;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 dims      = uint2(g_Frame.screenWidth, g_Frame.screenHeight);
    if (screenPos.x >= dims.x || screenPos.y >= dims.y) return;

    uint pixelIdx = screenPos.y * dims.x + screenPos.x;

    StructuredBuffer<DIRreservoir> intermediate = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWTexture2D<float4>            diOutput     = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    DIRreservoir res = intermediate[pixelIdx];

    if (res.M <= 0.0f || res.W <= 0.0f || res.selectedLightIndex == 0u || res.selectedLightIndex >= g_Frame.numLights)
    {
        diOutput[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    Surface surf;
    bool hasHit = ReconstructGBufferSurface(screenPos, dims, surf);
    if (!hasHit)
    {
        diOutput[screenPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Debug visualization: override output with reservoir field heatmap
    if (g_Frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF)
    {
        const float RESTIR_DI_SHADE_MAX_M = 40.0f; // spatial max history
        float v = 0.0f;
        if      (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_LIGHT_INDEX)    v = float(res.selectedLightIndex) / float(max(1u, g_Frame.numLights - 1u));
        else if (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_M_COUNT)        v = saturate(res.M / RESTIR_DI_SHADE_MAX_M);
        else if (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_WEIGHT)         v = saturate(res.W * 0.1f);
        else if (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_VISIBILITY_AGE) v = 0.0f; // not tracked in shade pass
        // Encode as a heat color (black -> red -> yellow -> white)
        float3 heat;
        heat.r = saturate(v * 3.0f);
        heat.g = saturate(v * 3.0f - 1.0f);
        heat.b = saturate(v * 3.0f - 2.0f);
        diOutput[screenPos] = float4(heat, 1.0f);
        return;
    }

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex + 5u);

    LightConstants winner = g_Lights[res.selectedLightIndex];
    float3 radiance = EvaluateDIReservoirWinner(surf, winner, rng);

    // Multiply by unbiased RIS weight W
    float3 result = radiance * res.W;

    diOutput[screenPos] = float4(result, res.W);
}
