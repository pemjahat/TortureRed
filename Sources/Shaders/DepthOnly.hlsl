#include "Common.hlsl"

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct PSInput {
    float4 position : SV_POSITION;
};

ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<NodeData> NodeCB : register(b2);

StructuredBuffer<MeshData> MeshBuffer : register(t1, space1);
PSInput VSMain(VSInput input) {
    PSInput output;
    float4x4 world = MeshBuffer[NodeCB.meshID].world;
    float4 worldPos = mul(float4(input.position, 1.0f), world);
    output.position = mul(worldPos, FrameCB.viewProj);
    return output;
}
