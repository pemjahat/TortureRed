#include "Common.hlsl"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirOutput : register(u0);
RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirInput : register(u1);
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
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

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
    params.frameIndex = g_Frame.frameIndex;
    params.pad2 = 0;

    RTXDI_GISpatialResamplingParameters sparams;
    sparams.depthThreshold = 0.1f;
    sparams.normalThreshold = 0.5f;
    sparams.numSamples = 4;
    sparams.samplingRadius = 20.0f;
    sparams.biasCorrectionMode = RTXDI_BIAS_CORRECTION_BASIC;
    sparams.pad1 = 0;
    sparams.pad2 = 0;
    sparams.pad3 = 0;

    uint sourceBufferIndex = 0;

    RTXDI_GIReservoir result = RTXDI_GISpatialResampling(
        launchIndex, surface, sourceBufferIndex, inputReservoir, rng, params, reservoirParams, sparams);

    uint ptr = RTXDI_ReservoirPositionToPointer(reservoirParams, launchIndex, 0);
    g_ReservoirOutput[ptr] = RTXDI_PackGIReservoir(result, 0);
}
