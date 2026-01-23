#include "Common.hlsl"

struct PSInput {
    float4 position : SV_POSITION;
};

ConstantBuffer<FrameConstants> FrameCB : register(b0);
StructuredBuffer<DrawNodeData> DrawNodeBuffer : register(t1, space1);
StructuredBuffer<GLTFVertex> GlobalVertexBuffer : register(t4, space1);

PSInput VSMain(uint instanceID : SV_StartInstanceLocation, uint vertexID : SV_VertexID) {
    DrawNodeData drawData = DrawNodeBuffer[instanceID];
    GLTFVertex v = GlobalVertexBuffer[drawData.vertexOffset + vertexID];

    PSInput output;
    float4 worldPos = mul(float4(v.position, 1.0f), drawData.world);
    output.position = mul(worldPos, FrameCB.viewProj);
    return output;
}
