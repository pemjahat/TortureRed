#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

float MapPositiveHeat(float value)
{
    const float epsilon = 1e-8f;
    const float minLog2Value = -20.0f;
    const float maxLog2Value = 4.0f;

    if (value <= epsilon)
        return 0.0f;

    float logValue = log2(max(value, epsilon));
    return saturate((logValue - minLog2Value) / (maxLog2Value - minLog2Value));
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    StructuredBuffer<Reservoir> reservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWTexture2D<float4> outputTex = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    Texture2D<float4> debugHeatmap = ResourceDescriptorHeap[g_Indices.InputIdx1];

    Reservoir reservoir = reservoirs[pixelIndex];
    if ((g_Frame.restirReservoirDebugMode <= RESTIR_RESERVOIR_DEBUG_WEIGHTSUM ||
        g_Frame.restirReservoirDebugMode == RESTIR_RESERVOIR_DEBUG_W) &&
        (reservoir.M <= 0.0f || reservoir.w_sum <= 0.0f))
    {
        outputTex[screenPos] = 0.0f;
        return;
    }

    switch (g_Frame.restirReservoirDebugMode)
    {
    case RESTIR_RESERVOIR_DEBUG_POSITION:
        outputTex[screenPos] = float4(reservoir.hitPos, 1.0f);
        break;
    case RESTIR_RESERVOIR_DEBUG_NORMAL:
        outputTex[screenPos] =float4(reservoir.hitNormal, 1.0f);
        break;
    case RESTIR_RESERVOIR_DEBUG_RADIANCE:
        outputTex[screenPos] =float4(reservoir.radiance, 1.0f);
        break;
    case RESTIR_RESERVOIR_DEBUG_WEIGHTSUM:
        outputTex[screenPos] = reservoir.w_sum;
        break;
    case RESTIR_RESERVOIR_DEBUG_SOURCE_PDF:
    case RESTIR_RESERVOIR_DEBUG_TARGET_PDF:
    case RESTIR_RESERVOIR_DEBUG_TARGET_SHAPE:
    case RESTIR_RESERVOIR_DEBUG_TEMPORAL_TARGET_PDF:
    case RESTIR_RESERVOIR_DEBUG_SPATIAL_SHIFTED_TARGET_PDF:
    {
        float heat = MapPositiveHeat(debugHeatmap[screenPos].x);
        outputTex[screenPos] = float4(heat.xxx, 1.0f);
        break;
    }
    case RESTIR_RESERVOIR_DEBUG_W:
    {
        float heat = MapPositiveHeat(reservoir.W);
        outputTex[screenPos] = float4(heat.xxx, 1.0f);
        break;
    }
    default:
        outputTex[screenPos] = 0.0f;
        break;
    }
}