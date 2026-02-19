#include "Common.hlsl"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);
Texture2D g_Textures[] : register(t0, space0);
RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirBuffer : register(u2);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);
SamplerState g_LinearSampler : register(s0);

#define RTXDI_GI_RESERVOIR_BUFFER g_ReservoirBuffer

#include "RtxdiBridge.hlsli"
#include "Rtxdi/GI/Reservoir.hlsli"

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RAB_Surface surface = RAB_GetGBufferSurface(launchIndex, false);
    
    RTXDI_ReservoirBufferParameters reservoirParams;
    reservoirParams.reservoirBlockRowPitch = (launchDims.x + 15) / 16 * 256;
    reservoirParams.reservoirArrayPitch = 0;

    RTXDI_GIReservoir res = RTXDI_LoadGIReservoir(reservoirParams, launchIndex, 0);

    float3 accumulatedColor = 0;

    if (!surface.valid) {
        accumulatedColor = float3(0.5f, 0.7f, 1.0f) * 0.2f;
    } else {
        // --- Direct Lighting: brute force all lights on primary surface ---
        accumulatedColor += GetDirectLightingMultiLights(
            surface.worldPos, surface.normal, surface.viewDir,
            surface.albedo, surface.metallic, surface.roughness,
            g_Scene, g_Lights, g_Frame.numLights, g_Frame, false);

        // --- Indirect Lighting from Reservoir ---
        if (res.weightSum > 0) {
            float3 L_res = normalize(res.position - surface.worldPos);
            float NdotL_res = max(0.0f, dot(surface.normal, L_res));
            
            if (NdotL_res > 0) {
                float3 diffuse, specular;
                EvaluateBSDF(surface.normal, surface.viewDir, L_res, surface.albedo, surface.metallic, surface.roughness, diffuse, specular);
                
                float3 evalContrib = (diffuse + specular) * NdotL_res;
                
                float3 indirectRadiance = evalContrib * res.radiance * res.weightSum;
                
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
        float lerpFactor = min( (n - 1.0f) / min(n, 2000.0f), 1.0f );
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    float3 exposedColor = accumulatedColor * g_Frame.exposure;
    g_Output[launchIndex] = float4(exposedColor / (exposedColor + 1.0f), 1.0f);
}
