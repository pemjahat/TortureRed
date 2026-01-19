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

// Pixel shader - simple shading using normal for color
PSOutput PSMain(PSInput input)
{
    PSOutput output;

    // Use normal as color (simple visualization)
    // Transform normal from [-1,1] to [0,1] range for color
    float3 color = (input.normal * 0.5f) + 0.5f;

    // Add some basic lighting effect
    float lighting = dot(input.normal, normalize(float3(1.0f, 1.0f, 1.0f))) * 0.5f + 0.5f;
    color *= lighting;

    output.color = float4(color, 1.0f);
    return output;
}