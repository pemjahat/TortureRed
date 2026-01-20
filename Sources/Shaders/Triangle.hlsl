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

// Frame constant buffer - shared per frame
cbuffer FrameCB : register(b0)
{
    row_major float4x4 viewProj;
};

// Material constants - passed as root constants
cbuffer MaterialCB : register(b1)
{
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    uint hasBaseColorTexture;
};

// Mesh constant buffer - per mesh
cbuffer MeshCB : register(b3)
{
    row_major float4x4 world;
};

// Textures
Texture2D baseColorTexture : register(t0);
SamplerState samplerState : register(s0);

// Vertex shader
PSInput VSMain(VSInput input)
{
    PSInput output;

    // Transform position to clip space: world * viewProj
    output.position = mul(float4(input.position, 1.0f), mul(world, viewProj));

    // Transform normal (simplified - assumes no non-uniform scaling)
    output.normal = input.normal;

    output.texCoord = input.texCoord;
    return output;
}

// Pixel shader - texture sampling with baseColorFactor lighting
PSOutput PSMain(PSInput input)
{
    PSOutput output;

    float4 color = baseColorFactor;

    if (hasBaseColorTexture)
    {
        color *= baseColorTexture.Sample(samplerState, input.texCoord);
    }

    // Apply lighting to baseColorFactor
    float lighting = dot(input.normal, normalize(float3(1.0f, 1.0f, 1.0f))) * 0.5f + 0.5f;
    color *= lighting;

    output.color = color;
    return output;
}