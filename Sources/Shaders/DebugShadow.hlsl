#include "Common.hlsl"
#include "CommonTracing.hlsl"

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

ConstantBuffer<FrameConstants> FrameCB : register(b0);
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

float4 PSMain(VSOutput input) : SV_Target
{
    float depth = g_Textures[FrameCB.depthIndex].Sample(g_LinearSampler, input.uv).r;
    float3 normal = g_Textures[FrameCB.normalIndex].Sample(g_LinearSampler, input.uv).rgb * 2.0f - 1.0f;
    
    // Early exit for sky pixels
    if (depth == 0.0f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    // Reconstruct world position from depth
    float4 ndc = float4(input.uv.x * 2.0f - 1.0f, (1.0f - input.uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= viewPos.w;
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);
    
    LightConstants mainLight = g_Lights[0];
    
    float3 N = normalize(normal);
    float3 L_main = normalize(-mainLight.direction.xyz);
    float NdotL_main = max(dot(N, L_main), 0.0);
    
    float shadowFactor = 1.0f;
    
    // Ray-Traced Shadow for Main Light (same logic as Lighting.hlsl)
    if (NdotL_main > 0.0f) {
        RayDesc shadowRay;
        shadowRay.Origin = worldPos.xyz + N * 0.001f;  // Normal bias to avoid self-intersection
        shadowRay.Direction = L_main;
        shadowRay.TMin = 0.001f;
        shadowRay.TMax = 10000.0f;  // Large value for directional light
        
        RayQuery<RAY_FLAG_NONE> shadowQuery;
        shadowQuery.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
        while (shadowQuery.Proceed()) {
            PROCESS_ALPHA_MASK(shadowQuery);
        }
        
        // Check if ray hit anything (shadowed) or missed (lit)
        shadowFactor = (shadowQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.0f;
    } else {
        // Back-facing surface
        shadowFactor = 0.0f;
    }
    
    // Output shadow mask for debug: white = lit, black = shadowed
    return float4(shadowFactor, shadowFactor, shadowFactor, 1.0);
}