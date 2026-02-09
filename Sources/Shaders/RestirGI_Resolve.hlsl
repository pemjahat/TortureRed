#include "CommonTracing.hlsl"

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);
RWStructuredBuffer<Reservoir> g_ReservoirCurrent : register(u2);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<LightConstants> g_Light : register(b1);

Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    // --- Read Reservoir for Primary Surface Properties ---
    uint pixelIdx = launchIndex.y * launchDims.x + launchIndex.x;
    Reservoir res = g_ReservoirCurrent[pixelIdx];

    float2 uv = ((float2)launchIndex + 0.5f) / (float2)launchDims;
    float depth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, uv, 0).r;

    float3 accumulatedColor = 0;

    if (depth >= 1.0f) {
        accumulatedColor = float3(0.5f, 0.7f, 1.0f) * 0.2f;
    } else {
        float3 N = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);
        float3 worldPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInverse);
        float3 albedo = g_Textures[g_Frame.albedoIndex].SampleLevel(g_LinearSampler, uv, 0).rgb;
        float4 materialProps = g_Textures[g_Frame.materialIndex].SampleLevel(g_LinearSampler, uv, 0);
        float roughness = max(0.01f, materialProps.r);
        float metallic = materialProps.g;
        
        float3 V = normalize(g_Frame.cameraPosition.xyz - worldPos);

        // --- Direct Lighting (Re-evaluating NEE since it's no longer in reservoir) ---
        accumulatedColor += GetDirectLighting(worldPos, N, V, albedo, metallic, roughness, g_Scene, g_Light, g_Frame);

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
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = g_AccumulationBuffer[launchIndex].rgb;
        float n = (float)g_Frame.frameIndex;
        float lerpFactor = min( (n - 1.0f) / min(n, 2000.0f), 1.0f );  // Clamp to <= 1
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    float3 exposedColor = accumulatedColor * g_Frame.exposure;
    g_Output[launchIndex] = float4(exposedColor / (exposedColor + 1.0f), 1.0f);
}
