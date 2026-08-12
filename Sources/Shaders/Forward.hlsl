#include "Common.hlsl"
#include "PBR.hlsl"
#include "CommonTracing.hlsl"

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
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

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

float4 PSMain(PSInput input) : SV_Target0 {
    MaterialConstants material = MaterialBuffer[input.materialID];
    
    float4 albedo = material.baseColorFactor;
    if (material.baseColorTextureIndex >= 0) {
        float4 sampled = g_Textures[material.baseColorTextureIndex].Sample(g_LinearSampler, input.texCoord);
        albedo *= sampled;
    }

    // Discard fully transparent pixels
    if (albedo.a < 0.01f) {
        discard;
    }

    float3 normal = normalize(input.normal);
    if (material.normalTextureIndex >= 0) {
        float3 sampledNormal = g_Textures[material.normalTextureIndex].Sample(g_LinearSampler, input.texCoord).rgb;
        sampledNormal = sampledNormal * 2.0f - 1.0f;
        
        // Basic normal mapping (assuming tangent space is aligned with world space for simplicity, 
        // or you can compute TBN matrix here if you have tangents)
        // For now, we just blend it with the vertex normal
        normal = normalize(normal + sampledNormal * 0.5f);
    }

    float roughness = material.roughnessFactor;
    float metallic = material.metallicFactor;
    if (material.metallicRoughnessTextureIndex >= 0) {
        float4 mrSample = g_Textures[material.metallicRoughnessTextureIndex].Sample(g_LinearSampler, input.texCoord);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }

    roughness = max(0.01f, roughness);

    float3 N = normal;
    float3 V = normalize(FrameCB.cameraPosition.xyz - input.worldPos);

    LightConstants mainLight = g_Lights[0];

    // 1. Main Directional Light (Index 0) with Ray-Traced Shadows
    float3 L_main = normalize(-mainLight.direction.xyz);
    float NdotL_main = max(dot(N, L_main), 0.0);
    
    float shadowFactor = 1.0f;

    // No shadow factor for blended materials    
    float3 diff_main, spec_main;
    EvaluateBSDF(N, V, L_main, albedo.rgb, metallic, roughness, diff_main, spec_main);
    float3 totalDirectLighting = (diff_main + spec_main) * mainLight.color.rgb * mainLight.intensity * NdotL_main * shadowFactor;

    // 2. Local Light Loop (Index 1+)
    for(uint i = 1; i < FrameCB.numLights; ++i)
    {
        LightConstants light = g_Lights[i];
        
        // Only handle point/spot lights in this loop
        if (light.position.w > 0.5f) 
        {
             float3 d = light.position.xyz - input.worldPos;
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
    
    // When TAA is active, output raw HDR — the TAA resolve shader handles
    // exposure and tonemapping. Otherwise, apply them here for direct display.
    if (FrameCB.taaEnabled)
    {
        return float4(min(max(finalColor, 0.0f), FP16Max), albedo.a);
    }

    // Basic Tone Mapping
    float3 exposedColor = (finalColor/FP16Scale) * exp2(FrameCB.exposure);
    float3 ldrColor = exposedColor / (exposedColor + 1.0f);
    
    return float4(ldrColor, albedo.a);
}
