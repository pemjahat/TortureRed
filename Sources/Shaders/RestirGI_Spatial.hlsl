#include "CommonTracing.hlsl"

RWTexture2D<float4> g_Output : register(u1);
RWStructuredBuffer<Reservoir> g_ReservoirCurrent : register(u2);
RWStructuredBuffer<Reservoir> g_ReservoirIntermediate : register(u3);

ConstantBuffer<FrameConstants> g_Frame : register(b0);

Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex + 1); // Avoid same RNG as temporal

    uint pixelIdx = launchIndex.y * launchDims.x + launchIndex.x;
    Reservoir res = g_ReservoirIntermediate[pixelIdx];

    float2 uv = ((float2)launchIndex + 0.5f) / (float2)launchDims;
    float depth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, uv, 0).r;
    float3 primaryPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInverse);
    float3 primaryNormal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);

    if (res.M > 0 && depth < 1.0f) {
        // --- Spatial Reuse ---
        for (int i = 0; i < 4; i++) {
            float2 offset = float2(next_float(rng), next_float(rng)) * 2.0f - 1.0f;
            int2 neighborIndex = (int2)launchIndex + (int2)(offset * 20.0f); // Radius 20

            if (neighborIndex.x >= 0 && neighborIndex.x < (int)launchDims.x && 
                neighborIndex.y >= 0 && neighborIndex.y < (int)launchDims.y) {
                
                uint neighborPixelIdx = neighborIndex.y * launchDims.x + neighborIndex.x;
                Reservoir neighborRes = g_ReservoirIntermediate[neighborPixelIdx];

                float2 neighborUV = ((float2)neighborIndex + 0.5f) / (float2)launchDims;
                float neighborDepth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, neighborUV, 0).r;
                float3 neighborNormal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, neighborUV, 0).xyz * 2.0f - 1.0f);
                float3 neighborPos = ReconstructWorldPos(neighborUV, neighborDepth, g_Frame.projectionInverse, g_Frame.viewInverse);
                
                // Consistency Checks
                float dotNormal = dot(primaryNormal, neighborNormal);
                float distPos = distance(primaryPos, neighborPos);
                
                if (neighborRes.M > 0 && neighborRes.targetPDF > 0 && dotNormal > 0.95f && distPos < 0.5f) {
                    // Jacobian for geometry shift
                    float jacobian = ComputeJacobian(primaryPos, neighborPos, neighborRes.hitPos, neighborRes.hitNormal);
                    jacobian = clamp(jacobian, 0.1f, 10.0f);
                    
                    // Re-evaluate readability of neighbor sample at current pixel
                    // This uses the fact that neighborRes.radiance is demodulated incoming radiance.
                    float3 L_res = normalize(neighborRes.hitPos - primaryPos);
                    float NdotL = max(0.0f, dot(primaryNormal, L_res));
                    float shiftedTargetPDF = neighborRes.targetPDF * jacobian; 
                    
                    if (NdotL > 0) {
                        // Correct RIS Weight for merging
                        float weight = shiftedTargetPDF * neighborRes.W * neighborRes.M;
                        mergeReservoirs(res, neighborRes, shiftedTargetPDF, weight, next_float(rng));
                    }
                }
            }
        }
    }

    // Final M-Clamping for Spatial
    if (res.M > 60.0f) {
        res.w_sum *= (60.0f / res.M);
        res.M = 60.0f;
    }

    // Final Normalization (Avoid bias at low light levels)
    if (res.targetPDF > 0) {
        res.W = res.w_sum / max(1e-6f, res.M * res.targetPDF);
        res.W = min(res.W, 10.0f); // Weight Clamping to 10.0
    } else {
        res.W = 0;
    }

    g_ReservoirCurrent[pixelIdx] = res;
}
