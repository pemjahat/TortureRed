#include "Common.hlsl"

// Triangle.hlsl - Vertex and pixel shaders for GLTF model rendering

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct PSOutput
{
    float4 color : SV_TARGET;
};

ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<NodeData> NodeCB : register(b2);
StructuredBuffer<MaterialConstants> MaterialBuffer : register(t0, space1);
StructuredBuffer<MeshData> MeshBuffer : register(t1, space1);


// Textures
Texture2D textures[] : register(t0);
SamplerState samplerState : register(s0);

// Vertex shader
PSInput VSMain(VSInput input)
{
    PSInput output;

    float4x4 world = MeshBuffer[NodeCB.meshID].world;

    // Transform position to clip space: world * viewProj
    output.position = mul(float4(input.position, 1.0f), mul(world, FrameCB.viewProj));

    // Transform normal (simplified - assumes no non-uniform scaling)
    output.normal = mul(input.normal, (float3x3)world);

    output.texCoord = input.texCoord;
    return output;
}

// Pixel shader
PSOutput PSMain(PSInput input)
{
    PSOutput output;
    
    MaterialConstants material = MaterialBuffer[NodeCB.materialID];
    
    float4 albedo = material.baseColorFactor;
    if (material.baseColorTextureIndex >= 0)
    {
        albedo *= textures[material.baseColorTextureIndex].Sample(samplerState, input.texCoord);
    }    

    // Apply lighting to baseColorFactor
    float lighting = dot(input.normal, normalize(float3(1.0f, 1.0f, 1.0f))) * 0.5f + 0.5f;
    output.color = albedo * lighting;
    return output;
}