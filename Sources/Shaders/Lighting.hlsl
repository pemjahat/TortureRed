#include "Common.hlsl"
#include "PBR.hlsl"
#include "CommonTracing.hlsl"
#include "IrCache_Lookup.hlsl"
#include "NRD.hlsli"

struct VSInput {
    uint vertexID : SV_VertexID;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    // Fullscreen triangle
    output.texCoord = float2((input.vertexID << 1) & 2, input.vertexID & 2);
    output.position = float4(output.texCoord * 2.0f - 1.0f, 0.0f, 1.0f);
    output.texCoord.y = 1.0f - output.texCoord.y;
    return output;
}

ConstantBuffer<FrameConstants>         FrameCB   : register(b0);
ConstantBuffer<BindlessIndices>        g_Indices : register(b1);
//ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);
//Texture2D<float4> g_IndirectLightingTex : register(t0, space3);
//Texture3D<float4> g_IrCacheTex : register(t1, space3);

// BakedGI.hlsli's SampleBakedGIProbe references FrameCB — include AFTER its
// declaration above.
#include "BakedGI.hlsli"

float4 PSMain(PSInput input) : SV_Target {
    float4 albedo = GetTexture2D(FrameCB.albedoIndex).Sample(g_LinearSampler, input.texCoord);
    float3 normal = GetTexture2D(FrameCB.normalIndex).Sample(g_LinearSampler, input.texCoord).rgb * 2.0f - 1.0f;
    float4 material = GetTexture2D(FrameCB.materialIndex).Sample(g_LinearSampler, input.texCoord);
    float depth = GetTexture2D(FrameCB.depthIndex).Sample(g_LinearSampler, input.texCoord).r;

    // Early exit for sky pixels (reverse-Z: depth <= 0.0 = clear/far plane).
    // Sample the baked sky cubemap for background color; no shading needed.
    if (depth <= 0.0f)
    {
        float3 cameraRayDir = GetCameraRayDirection(
            input.texCoord, FrameCB.projectionInverse,
            FrameCB.viewInverse, FrameCB.cameraPosition.xyz);
        float3 skyColor = SampleSky(cameraRayDir, FrameCB.skyCubemapIndex);
        if (FrameCB.taaEnabled)
            return float4(min(skyColor, FP16Max), 1.0f);
        float3 exposed = (skyColor/FP16Scale) * exp2(FrameCB.exposure);
        return float4(exposed / (exposed + 1.0f), 1.0f);
    }

    // Reconstruction of world position from depth
    float4 ndc = float4(input.texCoord.x * 2.0f - 1.0f, (1.0f - input.texCoord.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= viewPos.w;
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);

    LightConstants mainLight = g_Lights[0];

    RNG rng;
    uint2 pixelCoord = uint2(input.position.xy);
    seed_rng(rng, pixelCoord, FrameCB.frameIndex);

    // PBR setup
    float3 N = normalize(normal);
    float3 V = normalize(FrameCB.cameraPosition.xyz - worldPos.xyz);
    float roughness = max(0.01f, material.r);
    float metallic = material.g;

    // 1. Main Directional Light (Index 0) with Ray-Traced Shadows
    float3 L_main = normalize(-mainLight.direction.xyz);
    float NdotL_main = max(dot(N, L_main), 0.0);
    
    float shadowFactor = 1.0f;
    
    // Trace shadow ray using inline ray tracing (only if surface faces the light)
    if (NdotL_main > 0.0f) {
        RayDesc shadowRay;
        shadowRay.Origin = worldPos.xyz + N * 0.001f;  // Normal bias to avoid self-intersection
        shadowRay.Direction = L_main;
        shadowRay.TMin = 0.001f;
        shadowRay.TMax = 10000.0f;  // Large value for directional light
        
        RayQuery<RAY_FLAG_NONE> shadowQuery;
        shadowQuery.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
        while (shadowQuery.Proceed()) {
            PROCESS_ALPHA_MASK(shadowQuery, rng);
        }
        
        // Check if ray hit anything (shadowed) or missed (lit)
        shadowFactor = (shadowQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;
    } else {
        // Back-facing surface, no contribution from this light
        shadowFactor = 0.0f;
    }

    float3 diff_main, spec_main;
    EvaluateBSDF(N, V, L_main, albedo.rgb, metallic, roughness, diff_main, spec_main);
    float3 totalDirectLighting = (diff_main + spec_main) * mainLight.color.rgb * mainLight.intensity * NdotL_main * shadowFactor;

    // 2. Local Lights (Index 1+):
    //    When ReSTIR DI is active, its contribution is already in FinalDiffuseTex/FinalSpecularTex
    //    (written by StoreShadingOutput, optionally denoised by NRD).
    //    Otherwise fall back to inline RIS (4 candidates, 1 shadow ray).
    if (!FrameCB.enableRestirDI && FrameCB.numLights > 1)
    {
        const uint localLightRisCandidates = 4;
        totalDirectLighting += GetLocalLightDirectLightingRIS(
            worldPos.xyz, N, V,
            albedo.rgb, metallic, roughness,
            g_Scene, g_Lights, FrameCB.numLights,
            FrameCB, rng, false, localLightRisCandidates);
    }

    // Ambient term: SH9 sky irradiance (Tier 2). Cancel ambient as this doesn't have occlusion right now
    //float3 irradiance = EvalSH9IrradianceIndex(N, FrameCB.skySH9BufferIndex);
    //float3 ambient = irradiance * albedo.rgb / PI;
    //float3 finalColor = ambient + totalDirectLighting;
    float3 finalColor = totalDirectLighting;

    // Reservoir-path composition: the Final interchange textures are written
    // this frame iff a reservoir producer ran — ReSTIR DI (StoreShadingOutput
    // Call 1, overwrite) and/or ReSTIR GI (Call 2, additive over DI / overwrite
    // when DI is off). Fetch exactly when one did; otherwise the contents are
    // stale. The baked probe grid is the EXCLUSIVE alternative indirect source
    // (probe mode forces restirGI off, so the fetch below reduces to DI's
    // direct-only contribution there): probes supply the indirect diffuse —
    // sun/sky bounces + occluded first-bounce sky. Indirect specular is absent
    // in probe mode (SH9 probes are diffuse-only — known trade-off).
    const bool bakedGIActive = (FrameCB.bakedGIMode != 0) && (FrameCB.bakedGIValid != 0);

    if (FrameCB.enableRestirDI || FrameCB.enableRestirGI)
    {
        Texture2D<float4> finalDiffuseTex  = ResourceDescriptorHeap[g_Indices.InputIdx0];
        Texture2D<float4> finalSpecularTex = ResourceDescriptorHeap[g_Indices.InputIdx1];
        float3 finalDiffuse  = finalDiffuseTex.SampleLevel(g_LinearSampler, input.texCoord, 0).rgb;
        float3 finalSpecular = finalSpecularTex.SampleLevel(g_LinearSampler, input.texCoord, 0).rgb;

        float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo.rgb, metallic);
        float3 diffuseFactor, specularFactor;
        NRD_MaterialFactors(N, V, albedo.rgb, F0, roughness, diffuseFactor, specularFactor);

        // Reservoir debug view — raw reservoir-sampled signal, no material
        // re-modulation. Probe mode has no reservoirs (the UI only enables
        // this under ReSTIR GI; 
        if (FrameCB.restirReservoirDebugMode != RESTIR_RESERVOIR_DEBUG_OFF)
            return float4(finalDiffuse + finalSpecular, 1.0f);

        finalColor += finalDiffuse  * diffuseFactor;
        finalColor += finalSpecular * specularFactor;
    }

    // Baked probe indirect diffuse (validity-renormalized trilinear).
    if (bakedGIActive)
        finalColor += albedo.rgb / PI * SampleBakedGIProbe(worldPos.xyz, N);
    
    // When TAA is active, output raw HDR — the TAA resolve shader handles
    // exposure and tonemapping. Otherwise, apply them here for direct display.
    if (FrameCB.taaEnabled)
    {
        return float4(min(max(finalColor, 0.0f), FP16Max), 1.0f);
    }

    // Basic Tone Mapping
    float3 exposedColor = (finalColor/FP16Scale) * exp2(FrameCB.exposure);
    float3 ldrColor = exposedColor / (exposedColor + 1.0f);
    
    return float4(ldrColor, 1.0f);
}
