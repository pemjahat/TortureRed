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
ConstantBuffer<MaterialConstants> MaterialCB : register(b1);

// Mesh constant buffer - per mesh
cbuffer MeshCB : register(b3)
{
    row_major float4x4 world;
};

// Textures
Texture2D textures[] : register(t0);
SamplerState samplerState : register(s0);

// Vertex shader
PSInput VSMain(VSInput input)
{
    PSInput output;

    // Transform position to clip space: world * viewProj
    output.position = mul(float4(input.position, 1.0f), mul(world, FrameCB.viewProj));

    // Transform normal (simplified - assumes no non-uniform scaling)
    output.normal = input.normal;

    output.texCoord = input.texCoord;
    return output;
}

// Pixel shader - texture sampling with baseColorFactor lighting
PSOutput PSMain(PSInput input)
{
    PSOutput output;

    float4 color = MaterialCB.baseColorFactor;

    if (MaterialCB.baseColorTextureIndex >= 0)
    {
        color *= textures[MaterialCB.baseColorTextureIndex].Sample(samplerState, input.texCoord);
    }

    // Apply lighting to baseColorFactor
    float lighting = dot(input.normal, normalize(float3(1.0f, 1.0f, 1.0f))) * 0.5f + 0.5f;
    color *= lighting;

    output.color = color;
    return output;
}