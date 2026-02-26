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

    Surface centerSurface;
    float centerRayT;
    bool hasCenterHit = TracePrimarySurface(screenPos, launchDims, g_Frame, rng, centerSurface, centerRayT);
    if (!hasCenterHit) {
        g_ReservoirOutput[pixelIndex] = (Reservoir)0;
        return;
    }

    Surface s;
    s.worldPos = centerSurface.worldPos;
    s.normal = centerSurface.normal;
    s.viewDir = centerSurface.viewDir;
    s.albedo = centerSurface.albedo;
    s.metallic = centerSurface.metallic;
    s.roughness = centerSurface.roughness;

    Reservoir r = g_ReservoirTemporalInput[pixelIndex];
    float selectedPDF = 0.f;
    if (r.M > 0.f) {
        selectedPDF = GetTargetPDF(s, r.hitPos, r.radiance);
    }

    // Spatial Reuse
    int numNeighbors = 3;
    float radius = 20.0f;

    for (int i = 0; i < numNeighbors; ++i) {
        float2 offset = float2(next_float(rng) * 2.0f - 1.0f, next_float(rng) * 2.0f - 1.0f) * radius;
        int2 neighborPos = int2(screenPos) + int2(offset);
        
         if (neighborPos.x >= 0 && neighborPos.x < (int)launchDims.x && neighborPos.y >= 0 && neighborPos.y < (int)launchDims.y) {
            uint neighborIndex = neighborPos.y * launchDims.x + neighborPos.x;

                RNG neighborRng;
                seed_rng(neighborRng, (uint2)neighborPos, g_Frame.frameIndex + 17u + (uint)i);
                Surface neighborSurface;
                float neighborRayT;
                bool hasNeighborHit = TracePrimarySurface((uint2)neighborPos, launchDims, g_Frame, neighborRng, neighborSurface, neighborRayT);

            Reservoir neighborR = g_ReservoirTemporalInput[neighborIndex];
                if (hasNeighborHit && neighborR.M > 0.0f && dot(centerSurface.normal, neighborSurface.normal) > 0.95f) {
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
