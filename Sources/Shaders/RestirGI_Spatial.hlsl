#include "CommonTracing.hlsl"

RWTexture2D<float4> g_Output : register(u1);
RWStructuredBuffer<Reservoir> g_ReservoirCurrent : register(u2);
RWStructuredBuffer<Reservoir> g_ReservoirIntermediate : register(u3);

ConstantBuffer<FrameConstants> g_Frame : register(b0);

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

    if (res.M > 0) {
        // --- Spatial Reuse ---
        for (int i = 0; i < 4; i++) {
            float2 offset = float2(next_float(rng), next_float(rng)) * 2.0f - 1.0f;
            int2 neighborIndex = (int2)launchIndex + (int2)(offset * 20.0f); // Radius 20

            if (neighborIndex.x >= 0 && neighborIndex.x < (int)launchDims.x && 
                neighborIndex.y >= 0 && neighborIndex.y < (int)launchDims.y) {
                
                uint neighborPixelIdx = neighborIndex.y * launchDims.x + neighborIndex.x;
                Reservoir neighborRes = g_ReservoirIntermediate[neighborPixelIdx];
                
                // Consistency Checks
                float dotNormal = dot(res.primaryNormal, neighborRes.primaryNormal);
                float distPos = distance(res.primaryPos, neighborRes.primaryPos);
                
                if (neighborRes.M > 0 && neighborRes.targetPDF > 0 && dotNormal > 0.95f && distPos < 0.5f) {
                    // Clamped Jacobian Calculation
                    float jacobian = ComputeJacobian(res.primaryPos, neighborRes.primaryPos, neighborRes.hitPos, neighborRes.hitNormal);
                    jacobian = clamp(jacobian, 0.1f, 10.0f);
                    
                    // Shifted Target PDF for Selection
                    float shiftedTargetPDF = neighborRes.targetPDF * jacobian;
                    
                    // RIS Weight for merging
                    float weight = shiftedTargetPDF * neighborRes.W * neighborRes.M;
                    
                    // Merge into current reservoir
                    mergeReservoirs(res, neighborRes, shiftedTargetPDF, weight, next_float(rng));
                }
            }
        }
    }

    // Final M-Clamping for Spatial
    if (res.M > 240.0f) {
        res.w_sum *= (240.0f / res.M);
        res.M = 240.0f;
    }

    // Final Normalization with Weight Clamping
    if (res.targetPDF > 0) {
        res.W = res.w_sum / max(1.f, res.M * res.targetPDF);
        res.W = min(res.W, 10.0f); // Weight Clamping to 10.0
    } else {
        res.W = 0;
    }

    g_ReservoirCurrent[pixelIdx] = res;
}
