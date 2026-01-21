struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

cbuffer FrameCB : register(b0) {
    row_major float4x4 viewProj;
};

cbuffer MaterialCB : register(b1) {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorIndex;
    int padding;
};

cbuffer MeshCB : register(b3) {
    row_major float4x4 world;
};

Texture2D textures[] : register(t0);
SamplerState pointSampler : register(s0);

PSInput VSMain(VSInput input) {
    PSInput output;
    float4 worldPos = mul(float4(input.position, 1.0f), world);
    output.position = mul(worldPos, viewProj);
    output.worldPos = worldPos.xyz;
    output.normal = mul(input.normal, (float3x3)world);
    output.texCoord = input.texCoord;
    return output;
}

struct GBufferOutput {
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 material : SV_Target2;
};

GBufferOutput PSMain(PSInput input) {
    GBufferOutput output;
    
    float4 albedo = baseColorFactor;
    if (baseColorIndex >= 0) {
        float4 sampled = textures[baseColorIndex].Sample(pointSampler, input.texCoord);
        albedo *= sampled;
        
        // Simple alpha test for Masked geometry
        // Note: For a more robust implementation, only discard for AlphaMode::Mask
        if (sampled.a < 0.5f) discard;
    }
    
    output.albedo = albedo;
    output.normal = float4(normalize(input.normal) * 0.5f + 0.5f, 1.0f);
    output.material = float4(roughnessFactor, metallicFactor, 0.0f, 1.0f);
    
    return output;
}
