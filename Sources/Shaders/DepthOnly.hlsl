struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct PSInput {
    float4 position : SV_POSITION;
};

cbuffer FrameCB : register(b0) {
    row_major float4x4 viewProj;
};

cbuffer MeshCB : register(b3) {
    row_major float4x4 world;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    float4 worldPos = mul(float4(input.position, 1.0f), world);
    output.position = mul(worldPos, viewProj);
    return output;
}
