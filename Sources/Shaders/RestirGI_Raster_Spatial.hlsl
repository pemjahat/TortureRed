#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

static const float RESTIR_SPATIAL_DEPTH_THRESHOLD = 0.1f;
static const float RESTIR_SPATIAL_NORMAL_THRESHOLD = 0.95f;
static const float RESTIR_SPATIAL_ALBEDO_THRESHOLD = 0.15f;
static const float RESTIR_SPATIAL_ROUGHNESS_THRESHOLD = 0.15f;
static const float RESTIR_SPATIAL_METALLIC_THRESHOLD = 0.15f;
static const float RESTIR_SPATIAL_MIN_JACOBIAN = 0.1f;
static const float RESTIR_SPATIAL_MAX_JACOBIAN = 10.0f;

static const float RESTIR_GLOSSY_MIN_ROUGHNESS = 0.05f;
static const float RESTIR_GLOSSY_MAX_ROUGHNESS = 0.30f;
static const float RESTIR_SPATIAL_GLOSSY_MAX_JACOBIAN = 1.25f;
static const float RESTIR_SPATIAL_REUSE_WEIGHT_CLAMP_GLOSSY = 6.0f;
static const float RESTIR_SPATIAL_REUSE_WEIGHT_CLAMP_ROUGH  = 48.0f;
static const float RESTIR_SPATIAL_REFLECTION_THRESHOLD_MIN = 0.80f;
static const float RESTIR_SPATIAL_REFLECTION_THRESHOLD_MAX = 0.98f;

float GetGlossyFactor(float roughness)
{
    return 1.0f - saturate((roughness - RESTIR_GLOSSY_MIN_ROUGHNESS) / (RESTIR_GLOSSY_MAX_ROUGHNESS - RESTIR_GLOSSY_MIN_ROUGHNESS));
}

float3 GetSurfaceReflectionDir(Surface s)
{
    return reflect(-s.viewDir, s.normal);
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;
    
    // Initialize RNG
    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex + 1); // Offset seed for spatial

    StructuredBuffer<Reservoir> currReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<Reservoir> tempReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Surface centerSurface;
    float centerRayT;
    bool hasCenterHit = TracePrimarySurface(screenPos, launchDims, g_Frame, rng, centerSurface, centerRayT);
    if (!hasCenterHit) {
        tempReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    Surface s;
    s.worldPos = centerSurface.worldPos;
    s.normal = centerSurface.normal;
    s.viewDir = centerSurface.viewDir;
    s.albedo = centerSurface.albedo;
    s.metallic = centerSurface.metallic;
    s.roughness = centerSurface.roughness;

    float glossyFactor = GetGlossyFactor(centerSurface.roughness);
    float3 centerReflectionDir = GetSurfaceReflectionDir(centerSurface);

    Reservoir r = currReservoirs[pixelIndex];
    float selectedPDF = 0.f;
    float selectedShiftedTargetPdf = 0.0f;
    if (r.M > 0.f) {
        selectedPDF = GetTargetPDF(s, r.hitPos, r.radiance, true);
    }

    // Spatial Reuse
    int numNeighbors = 3;
    float radius = 20.f;

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

            Reservoir neighborR = currReservoirs[neighborIndex];
            bool normalsMatch = dot(centerSurface.normal, neighborSurface.normal) > RESTIR_SPATIAL_NORMAL_THRESHOLD;
            bool depthMatch = abs(neighborRayT - centerRayT) <= (RESTIR_SPATIAL_DEPTH_THRESHOLD * max(1.0f, centerRayT));
            bool materialMatch = AreMaterialsSimilar(centerSurface, neighborSurface,
                RESTIR_SPATIAL_ALBEDO_THRESHOLD,
                RESTIR_SPATIAL_ROUGHNESS_THRESHOLD,
                RESTIR_SPATIAL_METALLIC_THRESHOLD);

            if (hasNeighborHit && neighborR.M > 0.0f && normalsMatch && depthMatch && materialMatch) {
                bool neighborIsSpecular = ReservoirIsSpecular(neighborR);

                float maxJac = RESTIR_SPATIAL_MAX_JACOBIAN;
                float neighborTargetPDF = GetTargetPDF(s, neighborR.hitPos, neighborR.radiance, true);

                if (g_Frame.enableReservoirLobeCheck != 0u) {
                    float3 candidateDir = normalize(neighborR.hitPos - centerSurface.worldPos);

                    if (glossyFactor > 0.0f) {
                        float reflectionThreshold = lerp(
                            RESTIR_SPATIAL_REFLECTION_THRESHOLD_MIN,
                            RESTIR_SPATIAL_REFLECTION_THRESHOLD_MAX,
                            glossyFactor);

                        if (dot(centerReflectionDir, candidateDir) <= reflectionThreshold) {
                            continue;
                        }
                    }

                    float glossyReuseFactor = glossyFactor;
                    if (neighborIsSpecular) {
                        glossyReuseFactor = max(glossyReuseFactor, 0.5f);
                    }

                    maxJac = lerp(
                        RESTIR_SPATIAL_MAX_JACOBIAN,
                        RESTIR_SPATIAL_GLOSSY_MAX_JACOBIAN,
                        glossyReuseFactor);
                }

                float jacobian = ComputeJacobian(centerSurface.worldPos, neighborSurface.worldPos, neighborR.hitPos, neighborR.hitNormal);
                bool jacobianValid = jacobian >= RESTIR_SPATIAL_MIN_JACOBIAN && jacobian <= maxJac;

                if (neighborTargetPDF > 0.0f && jacobianValid) {
                    float shiftedTargetPDF = neighborTargetPDF * jacobian;

                    Reservoir adjustedNeighbor = neighborR;
                    float spatialReuseWeight = shiftedTargetPDF * adjustedNeighbor.W * adjustedNeighbor.M;
                    if (g_Frame.enableReservoirLobeCheck != 0u) {
                        float glossyReuseFactor = glossyFactor;
                        if (neighborIsSpecular) {
                            glossyReuseFactor = max(glossyReuseFactor, 0.5f);
                        }

                        float reuseClamp = lerp(
                            RESTIR_SPATIAL_REUSE_WEIGHT_CLAMP_ROUGH,
                            RESTIR_SPATIAL_REUSE_WEIGHT_CLAMP_GLOSSY,
                            glossyReuseFactor);

                        spatialReuseWeight = min(spatialReuseWeight, reuseClamp);
                    }
                    if (mergeReservoirsWithWeight(r, adjustedNeighbor, spatialReuseWeight, next_float(rng))) {
                        selectedPDF = neighborTargetPDF;
                        selectedShiftedTargetPdf = neighborTargetPDF * jacobian;
                        // historyAge (with lobe flag) is copied by mergeReservoirs automatically.
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

    if (g_Frame.restirReservoirDebugMode == RESTIR_RESERVOIR_DEBUG_SPATIAL_SHIFTED_TARGET_PDF)
    {
        RWTexture2D<float4> debugHeatmap = ResourceDescriptorHeap[g_Indices.OutputIdx1];
        debugHeatmap[screenPos] = float4(selectedShiftedTargetPdf, 0.0f, 0.0f, 1.0f);
    }

    tempReservoirs[pixelIndex] = r;
}
