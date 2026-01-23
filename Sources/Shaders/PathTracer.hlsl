
#include "Common.hlsl"

RWTexture2D<float4> g_Output : register(u0);
RaytracingAccelerationStructure g_Scene : register(t2, space1);
StructuredBuffer<PrimitiveData> g_Primitives : register(t3, space1);
StructuredBuffer<MeshData> g_MeshBuffer : register(t1, space1);
StructuredBuffer<MaterialConstants> g_Materials : register(t0, space1);
ByteAddressBuffer g_Buffers[] : register(t0, space0);

ConstantBuffer<FrameConstants> g_Frame : register(b0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    g_Output[launchIndex] = float4(1.0f, 0.0f, 1.0f, 1.0f);
}
