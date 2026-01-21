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
SamplerState pointSampler : register(s0);

float4 PSMain(PSInput input) : SV_Target {
    float4 albedo = albedoTexture.Sample(pointSampler, input.texCoord);
    float3 normal = normalTexture.Sample(pointSampler, input.texCoord).rgb * 2.0f - 1.0f;
    float4 material = materialTexture.Sample(pointSampler, input.texCoord);
    float depth = depthTexture.Sample(pointSampler, input.texCoord).r;

    // Simple ambient + directional light
    float3 lightDir = normalize(float3(1.0f, 1.0f, -1.0f));
    float3 ambientLight = float3(1.f, 0.f, 0.f);
    float diff = max(dot(normal, lightDir), 0.0f);
    float3 diffuse = diff * albedo.rgb;
    
    float3 ambient = 0.5f * albedo.rgb;
    
    return float4(ambient + diffuse, 1.0f);
}
