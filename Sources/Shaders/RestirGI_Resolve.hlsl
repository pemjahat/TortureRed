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

    float3 accumulatedColor = 0;

    // Check if background (no primary hit normal)
    if (dot(res.primaryNormal, res.primaryNormal) < 0.0001f) {
        accumulatedColor = res.primaryDirect;
    } else {
        // todo: read from gbuffer
        float3 N = res.primaryNormal;
        float3 worldPos = res.primaryPos;
        float3 albedo = res.primaryAlbedo;
        float roughness = res.primaryRoughness;
        float metallic = res.primaryMetallic;
        
        float3 V = normalize(g_Frame.cameraPosition.xyz - worldPos);

        // --- Direct Lighting (Static NEE from Temporal Pass) ---
        accumulatedColor += res.primaryDirect;

        // --- Indirect Lighting from Reservoir ---
        if (res.W > 0 && res.targetPDF > 0) {
            float3 L_res = normalize(res.hitPos - worldPos);
            float NdotL_res = max(0.0f, dot(N, L_res));
            if (NdotL_res > 0) {
                float3 diffuse, specular;
                EvaluateBSDF(N, V, L_res, albedo, metallic, roughness, diffuse, specular);
                // Clamp specular to avoid fireflies in indirect resolve
                specular = clamp(specular, 0.0, 10.0);
                accumulatedColor += (diffuse + specular) * res.radiance * res.W * NdotL_res;
            }
        }
    }

    // --- Temporal Post-Processing & Output ---
    if (g_Frame.frameIndex <= 1) {
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = g_AccumulationBuffer[launchIndex].rgb;
        float n = (float)g_Frame.frameIndex;
        float lerpFactor = (n - 1.0f) / min(n, 2000.0f);
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    float3 exposedColor = accumulatedColor * g_Frame.exposure;
    g_Output[launchIndex] = float4(exposedColor / (exposedColor + 1.0f), 1.0f);
}
