#include "Common.hlsl"
#include "PBR.hlsl"
#include "CommonTracing.hlsl"
#include "IrCache_Lookup.hlsl"

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

ConstantBuffer<FrameConstants>         FrameCB   : register(b0);
ConstantBuffer<BindlessIndices>        g_Indices : register(b1);
ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);
//Texture2D<float4> g_IndirectLightingTex : register(t0, space3);
//Texture3D<float4> g_IrCacheTex : register(t1, space3);

float4 PSMain(PSInput input) : SV_Target {
    float4 albedo = g_Textures[FrameCB.albedoIndex].Sample(g_LinearSampler, input.texCoord);
    float3 normal = g_Textures[FrameCB.normalIndex].Sample(g_LinearSampler, input.texCoord).rgb * 2.0f - 1.0f;
    float4 material = g_Textures[FrameCB.materialIndex].Sample(g_LinearSampler, input.texCoord);
    float depth = g_Textures[FrameCB.depthIndex].Sample(g_LinearSampler, input.texCoord).r;

    // Early exit for sky pixels
    // if (depth == 0.0f) {
    //     return float4(0.0f, 0.0f, 0.0f, 1.0f);
    // }

    // Reconstruction of world position from depth
    float4 ndc = float4(input.texCoord.x * 2.0f - 1.0f, (1.0f - input.texCoord.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= viewPos.w;
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);

    LightConstants mainLight = g_Lights[0];

    RNG rng;
    uint2 pixelCoord = uint2(input.position.xy);
    seed_rng(rng, pixelCoord, FrameCB.frameIndex);

    // PBR setup
    float3 N = normalize(normal);
    float3 V = normalize(FrameCB.cameraPosition.xyz - worldPos.xyz);
    float roughness = max(0.01f, material.r);
    float metallic = material.g;

    // 1. Main Directional Light (Index 0) with Ray-Traced Shadows
    float3 L_main = normalize(-mainLight.direction.xyz);
    float NdotL_main = max(dot(N, L_main), 0.0);
    
    float shadowFactor = 1.0f;
    
    // Trace shadow ray using inline ray tracing (only if surface faces the light)
    if (NdotL_main > 0.0f) {
        RayDesc shadowRay;
        shadowRay.Origin = worldPos.xyz + N * 0.001f;  // Normal bias to avoid self-intersection
        shadowRay.Direction = L_main;
        shadowRay.TMin = 0.001f;
        shadowRay.TMax = 10000.0f;  // Large value for directional light
        
        RayQuery<RAY_FLAG_NONE> shadowQuery;
        shadowQuery.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
        while (shadowQuery.Proceed()) {
            PROCESS_ALPHA_MASK(shadowQuery, rng);
        }
        
        // Check if ray hit anything (shadowed) or missed (lit)
        shadowFactor = (shadowQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;
    } else {
        // Back-facing surface, no contribution from this light
        shadowFactor = 0.0f;
    }

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
    
    if (FrameCB.enableRasterIndirectGI)
    {
        // Sample the indirect lighting texture
        Texture2D<float4> indirectIrradiance = ResourceDescriptorHeap[g_Indices.InputIdx0];
        
        float3 indirectLighting = indirectIrradiance.SampleLevel(g_LinearSampler, input.texCoord, 0).rgb;
        finalColor += indirectLighting;
    }
    
    if (FrameCB.debugIrCache != IRCACHE_DEBUG_OFF)
    {
        float3 debugVal = float3(0.0f, 0.0f, 0.0f);
        if (FrameCB.debugIrCache == IRCACHE_DEBUG_IRRADIANCE)
            debugVal = SampleIrCache(worldPos.xyz, g_IrCache, FrameCB.irCacheCameraPosition.xyz, N);
        else if (FrameCB.debugIrCache == IRCACHE_DEBUG_LIFE)
            debugVal = DebugIrCacheLife(worldPos.xyz, g_IrCache, FrameCB.irCacheCameraPosition.xyz);
        else if (FrameCB.debugIrCache == IRCACHE_DEBUG_CASCADE)
            debugVal = DebugIrCacheCascade(worldPos.xyz, g_IrCache, FrameCB.irCacheCameraPosition.xyz);
        finalColor = debugVal;
    }
    
    // Basic Tone Mapping
    float3 exposedColor = finalColor * FrameCB.exposure;
    float3 ldrColor = exposedColor / (exposedColor + 1.0f);
    
    return float4(ldrColor, 1.0f);
}
