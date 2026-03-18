#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    Texture2D<float4> hdrBuffer = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWTexture2D<float4> outputBuffer = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    float4 color = hdrBuffer[launchIndex];

    if (g_Frame.restirReservoirDebugMode == RESTIR_RESERVOIR_DEBUG_OFF)
    {
        float3 exposedColor = max(color.rgb, 0.0f) * g_Frame.exposure;
        color = float4(exposedColor / (exposedColor + 1.0f), color.a);
    }

    outputBuffer[launchIndex] = float4(saturate(color.rgb), 1.0f);
}