#include "CommonTracing.hlsl"

RWTexture2D<float4> g_Output : register(u1);
RWStructuredBuffer<Reservoir> g_ReservoirOutput : register(u2);      // Spatial output (goes to Resolve)
RWStructuredBuffer<Reservoir> g_ReservoirTemporalInput : register(u3); // Temporal output for this frame

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
    Reservoir temporalRes = g_ReservoirTemporalInput[pixelIdx];
    float selectedTargetPdf = 0;

    float2 uv = ((float2)launchIndex + 0.5f) / (float2)launchDims;
    float depth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, uv, 0).r;
    
    Surface centerSurface = (Surface)0;
    if (depth < 1.0f) {
        centerSurface.worldPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInverse);
        centerSurface.normal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);
        centerSurface.viewDir = normalize(g_Frame.cameraPosition.xyz - centerSurface.worldPos);
        centerSurface.albedo = g_Textures[g_Frame.albedoIndex].SampleLevel(g_LinearSampler, uv, 0).rgb;
        float4 materialProps = g_Textures[g_Frame.materialIndex].SampleLevel(g_LinearSampler, uv, 0);
        centerSurface.roughness = max(0.01f, materialProps.r);
        centerSurface.metallic = materialProps.g;
    }

    // 1. Start with an EMPTY reservoir for the spatial accumulation
    Reservoir spatialRes;
    spatialRes.hitPos = 0;
    spatialRes.hitNormal = 0;
    spatialRes.radiance = 0;
    spatialRes.targetPDF = 0;
    spatialRes.w_sum = 0;
    spatialRes.M = 0;

    // 2. Merge the temporal reservoir as the first candidate
    if (temporalRes.M > 0) {
        selectedTargetPdf = GetTargetPDF(centerSurface, temporalRes.hitPos, temporalRes.radiance);
        mergeReservoirs(spatialRes, temporalRes, selectedTargetPdf, 0.5f);
    }

    if (spatialRes.M > 0 && depth < 1.0f) {
        // --- Spatial Reuse ---
        for (int i = 0; i < 4; i++) {
            float2 offset = float2(next_float(rng), next_float(rng)) * 2.0f - 1.0f;
            int2 neighborIndex = (int2)launchIndex + (int2)(offset * 20.0f); // Radius 20

            if (neighborIndex.x >= 0 && neighborIndex.x < (int)launchDims.x && 
                neighborIndex.y >= 0 && neighborIndex.y < (int)launchDims.y) {
                
                uint neighborPixelIdx = neighborIndex.y * launchDims.x + neighborIndex.x;
                Reservoir neighborRes = g_ReservoirTemporalInput[neighborPixelIdx];

                float2 neighborUV = ((float2)neighborIndex + 0.5f) / (float2)launchDims;
                float neighborDepth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, neighborUV, 0).r;
                float3 neighborNormal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, neighborUV, 0).xyz * 2.0f - 1.0f);
                float3 neighborPos = ReconstructWorldPos(neighborUV, neighborDepth, g_Frame.projectionInverse, g_Frame.viewInverse);
                
                // Consistency Checks
                float dotNormal = dot(centerSurface.normal, neighborNormal);
                float distPos = distance(centerSurface.worldPos, neighborPos);
                
                if (neighborRes.M > 0 && dotNormal > 0.95f && distPos < 0.5f) {
                    // Jacobian for geometry shift
                    float jacobian = ComputeJacobian(centerSurface.worldPos, neighborPos, neighborRes.hitPos, neighborRes.hitNormal);
                    jacobian = clamp(jacobian, 0.1f, 10.0f);
                    
                    // Re-evaluate target PDF of neighbor sample at current pixel
                    float currentTargetPDF = GetTargetPDF(centerSurface, neighborRes.hitPos, neighborRes.radiance);
                    float shiftedTargetPDF = currentTargetPDF * jacobian; 
                    
                    if (shiftedTargetPDF > 0) {
                        // Conservative Visibility Check
                        //if (CheckVisibility(centerSurface.worldPos, centerSurface.normal, neighborRes.hitPos)) {
                            //mergeReservoirs(res, neighborRes, currentTargetPDF, weight, next_float(rng));
                            if (mergeReservoirs(spatialRes, neighborRes, shiftedTargetPDF, next_float(rng)))
                            {
                                selectedTargetPdf = currentTargetPDF;
                            }
                        //}
                    }
                }
            }
        }
    }

    // Final M-Clamping for Spatial
     if (spatialRes.M > 60.0f) {
         spatialRes.w_sum *= (60.0f / spatialRes.M);
         spatialRes.M = 60.0f;
     }

    // Final Normalization: Re-evaluate target PDF for the winning sample
    if (selectedTargetPdf > 0 && spatialRes.M > 0) {
        spatialRes.w_sum = spatialRes.w_sum / (spatialRes.M * selectedTargetPdf);
    } else {
        spatialRes.w_sum = 0;
    }

    g_ReservoirOutput[pixelIdx] = spatialRes;
}
