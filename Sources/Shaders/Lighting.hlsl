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

    StructuredBuffer<LightConstants> lightBuffer = ResourceDescriptorHeap[FrameCB.lightsBufferIndex];
    LightConstants mainLight = lightBuffer[0];

    // PBR setup
    float3 N = normalize(normal);
    float3 V = normalize(FrameCB.cameraPosition.xyz - worldPos.xyz);
    float roughness = max(0.01f, material.r);
    float metallic = material.g;

    // 1. Main Directional Light (Index 0)
    float3 L_main = normalize(-mainLight.direction.xyz);
    float NdotL_main = max(dot(N, L_main), 0.0);
    
    float4 shadowPos = mul(worldPos, mainLight.viewProj);
    float shadowFactor = CalcShadow(shadowPos);

    float3 diff_main, spec_main;
    EvaluateBSDF(N, V, L_main, albedo.rgb, metallic, roughness, diff_main, spec_main);
    float3 totalDirectLighting = (diff_main + spec_main) * mainLight.color.rgb * mainLight.intensity * NdotL_main * shadowFactor;

    // 2. Local Light Loop (Index 1+)
    for(uint i = 1; i < FrameCB.numLights; ++i)
    {
        LightConstants light = lightBuffer[i];
        
        // Only handle point/spot lights in this loop
        if (light.position.w > 0.5f) 
        {
             float3 d = light.position.xyz - worldPos.xyz;
             float dist = length(d);
             float3 L_i = normalize(d);
             
             // Spot attenuation
             float cosAngle = dot(-L_i, normalize(light.direction.xyz));
             float cosOuter = light.direction.w;
             float cosInner = asfloat(light.padding[0]);
             
             float spotEffect = smoothstep(cosOuter, cosInner, cosAngle);
             float attenuation = 1.0f / (1.0f + 0.1f*dist + 0.01f*dist*dist);
             
             float NdotL_i = max(dot(N, L_i), 0.0);
             
             float3 diff_i, spec_i;
             EvaluateBSDF(N, V, L_i, albedo.rgb, metallic, roughness, diff_i, spec_i);
             
             totalDirectLighting += (diff_i + spec_i) * light.color.rgb * light.intensity * NdotL_i * spotEffect * attenuation;
        }
    }

    float3 ambient = 0.03f * albedo.rgb;
    float3 finalColor = ambient + totalDirectLighting;
    
    // Basic Tone Mapping
    float3 exposedColor = finalColor * FrameCB.exposure;
    float3 ldrColor = exposedColor / (exposedColor + 1.0f);
    
    return float4(ldrColor, 1.0f);
}
