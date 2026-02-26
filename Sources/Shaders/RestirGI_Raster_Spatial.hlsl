#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);

// Outputs
RWTexture2D<float4> g_IndirectLightingTex : register(u1);

RWStructuredBuffer<Reservoir> g_ReservoirOutput : register(u2);      // Spatial output (goes to Resolve)
RWStructuredBuffer<Reservoir> g_ReservoirTemporalInput : register(u3); // Temporal output for this frame

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims;
    g_IndirectLightingTex.GetDimensions(launchDims.x, launchDims.y);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;
    
    // Initialize RNG
    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex + 1); // Offset seed for spatial

    // Read G-Buffer
    float depth = g_Textures[g_Frame.depthIndex].Sample(g_LinearSampler, screenPos).r;
    if (depth == 1.0f) {
        g_ReservoirOutput[pixelIndex] = (Reservoir)0;
        return;
    }

    float2 uv = ((float2)screenPos + 0.5f) / (float2)launchDims;
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 worldPos4 = mul(ndc, g_Frame.projectionInverse);
    worldPos4 /= worldPos4.w;
    float3 worldPos = mul(worldPos4, g_Frame.viewInverse).xyz;

    float3 normal = g_Textures[g_Frame.normalIndex].Sample(g_LinearSampler, screenPos).rgb * 2.0f - 1.0f;
    float3 albedo = g_Textures[g_Frame.albedoIndex].Sample(g_LinearSampler, screenPos).rgb;
    float4 material = g_Textures[g_Frame.materialIndex].Sample(g_LinearSampler, screenPos);
    float metallic = material.r;
    float roughness = max(0.01f, material.g);
    float3 viewDir = normalize(g_Frame.cameraPosition.xyz - worldPos);

    Surface s;
    s.worldPos = worldPos;
    s.normal = normal;
    s.viewDir = viewDir;
    s.albedo = albedo;
    s.metallic = metallic;
    s.roughness = roughness;

    Reservoir r = g_ReservoirTemporalInput[pixelIndex];
    float selectedPDF = 0.f;

    // Spatial Reuse
    int numNeighbors = 3;
    float radius = 20.0f;

    for (int i = 0; i < numNeighbors; ++i) {
        float2 offset = float2(next_float(rng) * 2.0f - 1.0f, next_float(rng) * 2.0f - 1.0f) * radius;
        int2 neighborPos = int2(screenPos) + int2(offset);
        
         if (neighborPos.x >= 0 && neighborPos.x < (int)launchDims.x && neighborPos.y >= 0 && neighborPos.y < (int)launchDims.y) {
            uint neighborIndex = neighborPos.y * launchDims.x + neighborPos.x;
            
            float3 neighborNormal = g_Textures[g_Frame.normalIndex].Sample(g_LinearSampler, neighborPos).rgb * 2.0f - 1.0f;
            
            // Geometric consistency check
            float dotNormal = dot(normal, neighborNormal);

            Reservoir neighborR = g_ReservoirTemporalInput[neighborIndex];
            if (neighborR.M > 0.0f && dotNormal > 0.95f) {
                // Re-evaluate target PDF for neighbor sample at current surface
                float neighborTargetPDF = GetTargetPDF(s, neighborR.hitPos, neighborR.radiance);
                
                if (neighborTargetPDF > 0.0f) {
                    if (mergeReservoirs(r, neighborR, neighborTargetPDF, next_float(rng))) {
                        selectedPDF = neighborTargetPDF;
                    }
                }
            }
        }
    }

    // Normalize reservoir weight
    if (r.M > 0.0f && selectedPDF > 0.0f) {
        r.W = r.w_sum / (r.M * selectedPDF);
    } else {
        r.W = 0.0f;
    }

    g_ReservoirOutput[pixelIndex] = r;
}
