#include "Common.hlsl"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirBuffer : register(u0);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

#define RTXDI_GI_RESERVOIR_BUFFER g_ReservoirBuffer

#include "RtxdiBridge.hlsli"
#include "Rtxdi/GI/Reservoir.hlsli"

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RAB_Surface surface = RAB_GetGBufferSurface(launchIndex, false);
    
    RTXDI_ReservoirBufferParameters reservoirParams;
    reservoirParams.reservoirBlockRowPitch = (launchDims.x + 15) / 16 * 256;
    reservoirParams.reservoirArrayPitch = 0;

    RTXDI_GIReservoir res = RTXDI_LoadGIReservoir(reservoirParams, launchIndex, 0);

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    RWTexture2D<float4> AccumulationBuffer = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> OutputBuffer = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    float3 accumulatedColor = 0;

    if (!surface.valid) {
        float3 cameraRayDir = GetPrimaryCameraRayDir(launchIndex, launchDims, g_Frame);
        accumulatedColor = SampleSky(cameraRayDir, g_Frame.skyCubemapIndex);
    } else {
        // --- Direct Lighting: dispatch based on lightSamplingMode ---
        // 0=Uniform (1 shadow ray), 1=ImportancePDF (1 shadow ray), 2=BruteForce (all lights)
        accumulatedColor += GetDirectLightingHybrid(
            surface.worldPos, surface.normal, surface.viewDir,
            surface.albedo, surface.metallic, surface.roughness,
            g_Scene, g_Lights, g_Frame.numLights, g_Frame, false, rng);

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
        AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = AccumulationBuffer[launchIndex].rgb;
        float n = (float)g_Frame.frameIndex;
        float lerpFactor = min( (n - 1.0f) / min(n, 2000.0f), 1.0f );
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    OutputBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
}
