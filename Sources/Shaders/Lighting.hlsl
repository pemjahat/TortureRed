struct VSInput {
    uint vertexID : SV_VertexID;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    // Fullscreen triangle
    output.texCoord = float2((input.vertexID << 1) & 2, input.vertexID & 2);
    output.position = float4(output.texCoord * 2.0f - 1.0f, 0.0f, 1.0f);
    output.texCoord.y = 1.0f - output.texCoord.y;
    return output;
}

Texture2D albedoTexture : register(t0);
Texture2D normalTexture : register(t1);
Texture2D materialTexture : register(t2);
Texture2D depthTexture : register(t3);
Texture2D shadowMap : register(t4); // Shadow map is index 4 in SRV heap if allocated that way

SamplerState pointSampler : register(s0);
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

float4 PSMain(PSInput input) : SV_Target {
    float4 albedo = albedoTexture.Sample(pointSampler, input.texCoord);
    float3 normal = normalTexture.Sample(pointSampler, input.texCoord).rgb * 2.0f - 1.0f;
    float4 material = materialTexture.Sample(pointSampler, input.texCoord);
    float depth = depthTexture.Sample(pointSampler, input.texCoord).r;

    // Reconstruction of world position from depth
    float4 ndc = float4(input.texCoord.x * 2.0f - 1.0f, (1.0f - input.texCoord.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 worldPos = mul(ndc, FrameCB.invViewProj);
    worldPos /= worldPos.w;

    // Shadow calculation
    float4 shadowPos = mul(worldPos, LightCB.viewProj);
    float shadow = CalcShadow(shadowPos);

    // Direct lighting with shadows
    float3 lightDir = normalize(-LightCB.direction.xyz);
    float diff = max(dot(normal, lightDir), 0.0f);
    
    float3 diffuse = diff * albedo.rgb * LightCB.color.rgb * shadow;
    float3 ambient = 0.1f * albedo.rgb;
    
    return float4(ambient + diffuse, 1.0f);
}
