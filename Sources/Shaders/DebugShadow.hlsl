struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

VSOutput VSMain(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    // Full screen triangle
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };
    
    float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };
    
    output.position = float4(positions[vertexID], 0.0, 1.0);
    output.uv = uvs[vertexID];
    
    return output;
}

Texture2D depthTexture : register(t3);
Texture2D shadowMap : register(t4);

SamplerState linearSampler : register(s0);
SamplerComparisonState shadowSampler : register(s1);

struct FrameConstants {
    row_major float4x4 viewProj;
    row_major float4x4 invViewProj;
};

ConstantBuffer<FrameConstants> FrameCB : register(b0);

struct LightConstants {
    row_major float4x4 viewProj;
    float4 position;
    float4 color;
    float4 direction;
};

ConstantBuffer<LightConstants> LightCB : register(b2);

float CalcShadow(float4 shadowPos) {
    shadowPos.xyz /= shadowPos.w;
    float2 shadowTexCoord = shadowPos.xy * 0.5f + 0.5f;
    shadowTexCoord.y = 1.0f - shadowTexCoord.y;

    if (saturate(shadowTexCoord.x) != shadowTexCoord.x || saturate(shadowTexCoord.y) != shadowTexCoord.y) {
        return 1.0f;
    }

    float shadowDepth = shadowPos.z;
    float shadow = 0.0f;
    
    // 3x3 PCF
    const float shadowMapSize = 2048.0f;
    const float texelSize = 1.0f / shadowMapSize;
    
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            shadow += shadowMap.SampleCmpLevelZero(shadowSampler, shadowTexCoord + float2(x, y) * texelSize, shadowDepth);
        }
    }
    
    return shadow / 9.0f;
}

float4 PSMain(VSOutput input) : SV_Target
{
    // Sample depth
    float depth = depthTexture.Sample(linearSampler, input.uv).r;
    
    // Reconstruct world position
    float4 ndc = float4(input.uv.x * 2.0f - 1.0f, (1.0f - input.uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 worldPos = mul(ndc, FrameCB.invViewProj);
    worldPos /= worldPos.w;    
    
    // Compute shadow position
    float4 shadowPos = mul(worldPos, LightCB.viewProj);
    
    // Calculate shadow factor
    float shadow = CalcShadow(shadowPos);
    
    // Output shadow factor as grayscale (1.0 = fully lit, 0.0 = fully shadowed)
    return float4(shadow, shadow, shadow, 1.0);
}