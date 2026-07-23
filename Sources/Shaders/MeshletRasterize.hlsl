#include "MeshletCommon.hlsli"
#include "PBR.hlsl"

// Bindless texture and sampler declarations (shared with existing shaders)
Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0);

// Lights buffer — must match the layout used by all other shaders
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// 7 global stream buffers (bound once per frame)
StructuredBuffer<float3>          GlobalPositions         : register(t0, space3);
StructuredBuffer<uint>            GlobalNormals           : register(t1, space3);
StructuredBuffer<uint>            GlobalUVs               : register(t2, space3);
StructuredBuffer<Meshlet>         GlobalMeshlets          : register(t3, space3);
StructuredBuffer<uint>            GlobalMeshletVertices   : register(t4, space3);
StructuredBuffer<MeshletTriangle> GlobalMeshletTriangles  : register(t5, space3);

// Per-mesh / per-instance metadata
StructuredBuffer<MeshData>        GlobalMeshData          : register(t6, space3);
StructuredBuffer<InstanceData>    GlobalInstanceData      : register(t7, space3);

// Visible meshlet candidates (output of culling pass)
StructuredBuffer<MeshletCandidate> VisibleMeshlets        : register(t8, space3);

// Material and light buffers (existing)
StructuredBuffer<MaterialConstants> MaterialBuffer        : register(t0, space1);

// PS input
struct PSInput {
    float4 position     : SV_POSITION;
    float3 worldPos     : WORLD_POS;
    float3 normal       : NORMAL;
    float2 texCoord     : TEXCOORD;
    nointerpolation uint materialID : MATERIAL_ID;
};

// VS: one invocation per vertex per visible meshlet
// The input assembler feeds SV_VertexID and SV_InstanceID.
// SV_InstanceID = index into VisibleMeshlets
// SV_VertexID   = vertex index within the meshlet
PSInput VSMain(uint instanceID : SV_InstanceID, uint vertexID : SV_VertexID)
{
    // Load meshlet candidate
    MeshletCandidate cand = VisibleMeshlets[instanceID];

    // Load instance and mesh data
    InstanceData inst = GlobalInstanceData[cand.InstanceID];
    MeshData md = GlobalMeshData[inst.MeshDataIndex];

    // Load meshlet header
    Meshlet m = GlobalMeshlets[md.MeshletOffset + cand.MeshletIndex];

    // Guard: if vertexID exceeds meshlet vertex count, degenerate vertex
    if (vertexID >= m.VertexCount)
    {
        PSInput degenerate;
        degenerate.position = float4(0, 0, 0, 0);
        degenerate.worldPos = float3(0, 0, 0);
        degenerate.normal = float3(0, 1, 0);
        degenerate.texCoord = float2(0, 0);
        degenerate.materialID = 0;
        return degenerate;
    }

    // Look up the actual global vertex index via indirection table
    uint globalVertexIdx = GlobalMeshletVertices[md.MeshletVertexOffset + m.VertexOffset + vertexID];

    // Fetch vertex attributes from global streams
    float3 localPos   = GlobalPositions[md.PositionOffset + globalVertexIdx];
    float3 localNormal = UnpackNormalRGB10A2(GlobalNormals, md.NormalOffset, globalVertexIdx);
    float2 uv         = UnpackUVRG16(GlobalUVs, md.UVOffset, globalVertexIdx);

    // Transform to world/clip space
    float4 worldPos = mul(float4(localPos, 1.0), inst.LocalToWorld);
    float4 clipPos  = mul(worldPos, FrameCB.viewProj);

    PSInput output;
    output.position   = clipPos;
    output.worldPos   = worldPos.xyz;
    output.normal     = normalize(mul(localNormal, (float3x3)inst.LocalToWorld));
    output.texCoord   = uv;
    output.materialID = md.MaterialIndex;
    return output;
}

// The actual per-pixel shading is identical to the existing Forward/GBuffer PS.
// The key difference is the VS which uses meshlet indirection.

// PS: identical to Forward.hlsl PSMain — reuses existing material/lighting
float4 PSMain(PSInput input) : SV_Target0 {
    MaterialConstants material = MaterialBuffer[input.materialID];
    
    float4 albedo = material.baseColorFactor;
    if (material.baseColorTextureIndex >= 0) {
        float4 sampled = g_Textures[material.baseColorTextureIndex].Sample(g_LinearSampler, input.texCoord);
        albedo *= sampled;
    }

    if (albedo.a < 0.01f) {
        discard;
    }

    float3 normal = normalize(input.normal);
    if (material.normalTextureIndex >= 0) {
        float3 sampledNormal = g_Textures[material.normalTextureIndex].Sample(g_LinearSampler, input.texCoord).rgb;
        sampledNormal = sampledNormal * 2.0f - 1.0f;
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
    float3 L_main = normalize(-mainLight.direction.xyz);
    float NdotL_main = max(dot(N, L_main), 0.0);
    
    float3 diff_main, spec_main;
    EvaluateBSDF(N, V, L_main, albedo.rgb, metallic, roughness, diff_main, spec_main);
    float3 totalDirectLighting = (diff_main + spec_main) * mainLight.color.rgb * mainLight.intensity * NdotL_main;

    for(uint i = 1; i < FrameCB.numLights; ++i)
    {
        LightConstants light = g_Lights[i];
        if (light.position.w > 0.5f) 
        {
             float3 d = light.position.xyz - input.worldPos;
             float dist = length(d);
             float3 L_i = normalize(d);
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
    
    if (FrameCB.taaEnabled)
        return float4(max(finalColor, 0.0f), albedo.a);

    float3 exposedColor = finalColor * FrameCB.exposure;
    float3 ldrColor = exposedColor / (exposedColor + 1.0f);
    return float4(ldrColor, albedo.a);
}