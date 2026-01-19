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

// Simple view-projection matrix for 3D rendering
cbuffer ViewProjection : register(b0)
{
    row_major float4x4 viewProjection;
};

// Material constants
cbuffer MaterialConstants : register(b1)
{
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    uint hasBaseColorTexture;
};

// Textures
Texture2D baseColorTexture : register(t0);
SamplerState samplerState : register(s0);

// Vertex shader
PSInput VSMain(VSInput input)
{
    PSInput output;

    // Transform position to clip space
    output.position = mul(float4(input.position, 1.0f), viewProjection);

    // Transform normal (simplified - assumes no non-uniform scaling)
    output.normal = input.normal;

    output.texCoord = input.texCoord;
    return output;
}

// Pixel shader - texture sampling with fallback to normal color
PSOutput PSMain(PSInput input)
{
    PSOutput output;

    float4 color = baseColorFactor;

    if (hasBaseColorTexture)
    {
        color *= baseColorTexture.Sample(samplerState, input.texCoord);
    }
    else
    {
        // Fallback to normal color
        float3 normalColor = (input.normal * 0.5f) + 0.5f;
        float lighting = dot(input.normal, normalize(float3(1.0f, 1.0f, 1.0f))) * 0.5f + 0.5f;
        color = float4(normalColor * lighting, 1.0f);
    }

    output.color = color;
    return output;
}