#include "Common.hlsl"

struct PSInput {
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
    nointerpolation uint materialID : MATERIAL_ID;
};

ConstantBuffer<FrameConstants> FrameCB : register(b0);

StructuredBuffer<MaterialConstants> MaterialBuffer : register(t0, space1);
StructuredBuffer<DrawNodeData> DrawNodeBuffer : register(t1, space1);
StructuredBuffer<GLTFVertex> GlobalVertexBuffer : register(t4, space1);

Texture2D textures[] : register(t0);
SamplerState pointSampler : register(s0);

PSInput VSMain(uint instanceID : SV_StartInstanceLocation, uint vertexID: SV_VertexID)
{
    DrawNodeData drawData = DrawNodeBuffer[instanceID];
    GLTFVertex v = GlobalVertexBuffer[drawData.vertexOffset + vertexID];

    PSInput output;
    float4 worldPos = mul(float4(v.position, 1.0f), drawData.world);
    output.position = mul(worldPos, FrameCB.viewProj);
    output.worldPos = worldPos.xyz;
    output.normal = mul(v.normal, (float3x3)drawData.world);
    output.texCoord = v.texCoord;
    output.materialID = drawData.materialID;
    return output;
}

struct GBufferOutput {
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 material : SV_Target2;
};

GBufferOutput PSMain(PSInput input) {
    GBufferOutput output;
    
    MaterialConstants material = MaterialBuffer[input.materialID];
    
    float4 albedo = material.baseColorFactor;
    if (material.baseColorTextureIndex >= 0) {
        float4 sampled = textures[material.baseColorTextureIndex].Sample(pointSampler, input.texCoord);
        albedo *= sampled;
        
        // Simple alpha test for Masked geometry
        // Note: For a more robust implementation, only discard for AlphaMode::Mask
        if (sampled.a < 0.5f) discard;
    }
    
    output.albedo = albedo;
    output.normal = float4(normalize(input.normal) * 0.5f + 0.5f, 1.0f);
    
    float roughness = material.roughnessFactor;
    float metallic = material.metallicFactor;
    
    if (material.metallicRoughnessTextureIndex >= 0) {
        float4 mrSample = textures[material.metallicRoughnessTextureIndex].Sample(pointSampler, input.texCoord);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }
    
    output.material = float4(roughness, metallic, 0.0f, 1.0f);
    
    return output;
}
