#include "Common.hlsl"
#include "PBR.hlsl"

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

Texture2D textures[] : register(t0, space0);

SamplerState pointSampler : register(s0);
SamplerComparisonState shadowSampler : register(s1);

ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<LightConstants> LightCB : register(b1);

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
            shadow += textures[FrameCB.shadowMapIndex].SampleCmpLevelZero(shadowSampler, shadowTexCoord + float2(x, y) * texelSize, shadowDepth);
        }
    }
    
    return shadow / 9.0f;
}

float4 PSMain(PSInput input) : SV_Target {
    float4 albedo = textures[FrameCB.albedoIndex].Sample(pointSampler, input.texCoord);
    float3 normal = textures[FrameCB.normalIndex].Sample(pointSampler, input.texCoord).rgb * 2.0f - 1.0f;
    float4 material = textures[FrameCB.materialIndex].Sample(pointSampler, input.texCoord);
    float depth = textures[FrameCB.depthIndex].Sample(pointSampler, input.texCoord).r;

    // Reconstruction of world position from depth
    float4 ndc = float4(input.texCoord.x * 2.0f - 1.0f, (1.0f - input.texCoord.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= viewPos.w;
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);

    // Shadow calculation
    float4 shadowPos = mul(worldPos, LightCB.viewProj);
    float shadow = CalcShadow(shadowPos);

    // PBR Lighting
    float3 N = normalize(normal);
    float3 V = normalize(FrameCB.cameraPosition.xyz - worldPos.xyz);
    float3 L = normalize(-LightCB.direction.xyz);
    float3 H = normalize(V + L);

    float roughness = max(0.01f, material.r);
    float metallic = material.g;

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo.rgb, metallic);
    float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);

    float3 numerator = D * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    float3 specular = numerator / denominator;

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float NdotL = max(dot(N, L), 0.0);
    float3 directLighting = (kD * albedo.rgb / 3.14159265 + specular) * LightCB.color.rgb * LightCB.intensity * NdotL * shadow;

    float3 ambient = 0.03f * albedo.rgb;
    float3 finalColor = ambient + directLighting;
    
    // Basic Tone Mapping
    float3 exposedColor = finalColor * FrameCB.exposure;
    float3 ldrColor = exposedColor / (exposedColor + 1.0f);
    
    return float4(ldrColor, 1.0f);
}
