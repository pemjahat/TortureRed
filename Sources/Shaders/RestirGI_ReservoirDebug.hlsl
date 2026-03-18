#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    StructuredBuffer<Reservoir> reservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWTexture2D<float4> outputTex = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Reservoir reservoir = reservoirs[pixelIndex];
    if (reservoir.M <= 0.0f || reservoir.w_sum <= 0.0f)
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
    default:
        outputTex[screenPos] = 0.0f;
        break;
    }
}