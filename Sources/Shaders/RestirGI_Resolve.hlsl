#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    // --- Read Reservoir for Primary Surface Properties ---
    uint pixelIdx = launchIndex.y * launchDims.x + launchIndex.x;

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    // Accessing texture bindless
    StructuredBuffer<Reservoir> tempReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWTexture2D<float4> accumulationBuffer = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> outputBuffer = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    Reservoir res = tempReservoirs[pixelIdx];

    float3 accumulatedColor = 0;

    Surface primarySurface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(launchIndex, launchDims, g_Frame, rng, primarySurface, primaryRayT);

    if (!hasPrimaryHit) {
        accumulatedColor = float3(0.5f, 0.7f, 1.0f) * 0.2f;
    } else {
        float3 N = primarySurface.normal;
        float3 worldPos = primarySurface.worldPos;
        float3 albedo = primarySurface.albedo;
        float roughness = primarySurface.roughness;
        float metallic = primarySurface.metallic;
        float3 V = primarySurface.viewDir;

        // --- Direct Lighting: dispatch based on lightSamplingMode ---
        // 0=Uniform (1 shadow ray), 1=ImportancePDF (1 shadow ray), 2=BruteForce (all lights)
        accumulatedColor += GetDirectLightingHybrid(
            worldPos, N, V, albedo, metallic, roughness,
            g_Scene, g_Lights, g_Frame.numLights, g_Frame, false, rng);

        // --- Indirect Lighting from Reservoir ---
        if (res.W > 0) {
            float3 L_res = normalize(res.hitPos - worldPos);
            float NdotL_res = max(0.0f, dot(N, L_res));
            
            if (NdotL_res > 0) {
                float3 diffuse, specular;
                EvaluateBSDF(N, V, L_res, albedo, metallic, roughness, diffuse, specular);
                
                // re-evaluate contribution scale
                float3 evalContrib = (diffuse + specular) * NdotL_res;
                
                // Final Unbiased ReSTIR GI Resolve:
                // res.radiance is incident radiance (L_in).
                // We multiply by the current BRDF and weight by normalization W.
                float3 indirectRadiance = evalContrib * res.radiance * res.W;
                
                // Clamp to prevent fireflies from extreme weights
                accumulatedColor += min(indirectRadiance, 10.0f);
            }
        }
    }

    // --- Temporal Post-Processing & Output ---
    if (g_Frame.frameIndex <= 1) {
        accumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = accumulationBuffer[launchIndex].rgb;
        float n = (float)g_Frame.frameIndex;
        float lerpFactor = min( (n - 1.0f) / min(n, 2000.0f), 1.0f );  // Clamp to <= 1
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        accumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    float3 exposedColor = accumulatedColor * g_Frame.exposure;
    outputBuffer[launchIndex] = float4(exposedColor / (exposedColor + 1.0f), 1.0f);
}
