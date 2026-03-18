#include "Common.hlsl"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirBuffer : register(u0);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

#define RTXDI_GI_RESERVOIR_BUFFER g_ReservoirBuffer
#define RTXDI_ENABLE_STORE_RESERVOIR 0

#include "Rtxdi/GI/Reservoir.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    RWTexture2D<float4> outputTex = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    RTXDI_ReservoirBufferParameters reservoirParams;
    reservoirParams.reservoirBlockRowPitch = (launchDims.x + 15) / 16 * 256;
    reservoirParams.reservoirArrayPitch = 0;

    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(reservoirParams, screenPos, 0);
    if (reservoir.M == 0u || reservoir.weightSum <= 0.0f)
    {
        outputTex[screenPos] = 0.0f;
        return;
    }

    switch (g_Frame.restirReservoirDebugMode)
    {
    case RESTIR_RESERVOIR_DEBUG_POSITION:
        outputTex[screenPos] = float4(reservoir.position, 1.0f);
        break;
    case RESTIR_RESERVOIR_DEBUG_NORMAL:
        outputTex[screenPos] = float4(reservoir.normal, 1.0f);
        break;
    case RESTIR_RESERVOIR_DEBUG_RADIANCE:
        outputTex[screenPos] = float4(reservoir.radiance, 1.0f);
        break;
    case RESTIR_RESERVOIR_DEBUG_WEIGHTSUM:
        outputTex[screenPos] = reservoir.weightSum;
        break;
    default:
        outputTex[screenPos] = 0.0f;
        break;
    }
}