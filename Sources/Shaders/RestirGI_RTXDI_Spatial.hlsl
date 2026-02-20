#include "Common.hlsl"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);

RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirOutput : register(u2);
RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirInput : register(u3);
Buffer<float2> g_NeighborOffsets : register(t5, space1);

ConstantBuffer<FrameConstants> g_Frame : register(b0);

#define RTXDI_GI_RESERVOIR_BUFFER g_ReservoirInput
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER g_NeighborOffsets

#include "RtxdiBridge.hlsli"
#include "Rtxdi/GI/Reservoir.hlsli"
#include "Rtxdi/GI/SpatialResampling.hlsli"

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_AccumulationBuffer.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(launchIndex, g_Frame.frameIndex + 100);
    RAB_Surface surface = RAB_GetGBufferSurface(launchIndex, false);

    RTXDI_ReservoirBufferParameters reservoirParams;
    reservoirParams.reservoirBlockRowPitch = (launchDims.x + 15) / 16 * 256;
    reservoirParams.reservoirArrayPitch = 0;

    if (!RAB_IsSurfaceValid(surface)) {
        uint ptr = RTXDI_ReservoirPositionToPointer(reservoirParams, launchIndex, 0);
        g_ReservoirOutput[ptr] = (RTXDI_PackedGIReservoir)0;
        return;
    }

    // Load input (temporal result)
    RTXDI_GIReservoir inputReservoir = RTXDI_LoadGIReservoir(reservoirParams, launchIndex, 0);

    RTXDI_RuntimeParameters params;
    params.activeCheckerboardField = 0;
    params.neighborOffsetMask = 31;

    RTXDI_GISpatialResamplingParameters sparams;
    sparams.numSamples = 4;
    sparams.samplingRadius = 20.0f;
    //sparams.biasCorrectionMode = RTXDI_GI_ALLOWED_BIAS_CORRECTION;
    sparams.biasCorrectionMode = 1;

    RTXDI_GIReservoir result = RTXDI_GISpatialResampling(
        launchIndex, surface, inputReservoir, rng, params, reservoirParams, sparams);

    uint ptr = RTXDI_ReservoirPositionToPointer(reservoirParams, launchIndex, 0);
    g_ReservoirOutput[ptr] = RTXDI_PackGIReservoir(result, 0);
}
