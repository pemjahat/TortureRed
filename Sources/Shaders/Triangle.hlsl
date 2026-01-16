// Triangle.hlsl - Simple vertex and pixel shaders for rendering a triangle

struct VSInput
{
    float3 position : POSITION;
    float3 color : COLOR;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 color : COLOR;
};

struct PSOutput
{
    float4 color : SV_TARGET;
};

// Vertex shader
PSInput VSMain(VSInput input)
{
    PSInput output;
    output.position = float4(input.position, 1.0f);
    output.color = input.color;
    return output;
}

// Pixel shader
PSOutput PSMain(PSInput input)
{
    PSOutput output;
    output.color = float4(input.color, 1.0f);
    return output;
}