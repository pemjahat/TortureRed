#include "SharcCommon.h"
#include "CommonTracing.hlsl"
#include "SHaRC_Integration.hlsl"

ConstantBuffer<FrameConstants>       g_Frame   : register(b0);
ConstantBuffer<BindlessIndices>      g_Indices : register(b1);
ConstantBuffer<SharcBindlessIndices> g_Sharc   : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

static const float RESTIR_TEMPORAL_DEPTH_THRESHOLD = 0.1f;
static const float RESTIR_TEMPORAL_NORMAL_THRESHOLD = 0.95f;
static const float RESTIR_TEMPORAL_ALBEDO_THRESHOLD = 0.15f;
static const float RESTIR_TEMPORAL_ROUGHNESS_THRESHOLD = 0.15f;
static const float RESTIR_TEMPORAL_METALLIC_THRESHOLD = 0.15f;
static const float RESTIR_TEMPORAL_MAX_HISTORY_LENGTH = 16.0f;
static const uint  RESTIR_TEMPORAL_MAX_HISTORY_AGE = 12u;
static const float RESTIR_TEMPORAL_MAX_JACOBIAN = 10.0f;
static const float RESTIR_TEMPORAL_MIN_JACOBIAN = 0.1f;

static const float RESTIR_GLOSSY_MIN_ROUGHNESS = 0.05f;
static const float RESTIR_GLOSSY_MAX_ROUGHNESS = 0.30f;
static const float RESTIR_TEMPORAL_GLOSSY_MAX_HISTORY = 3.0f;
static const float RESTIR_TEMPORAL_GLOSSY_MAX_JACOBIAN = 1.5f;
static const float RESTIR_TEMPORAL_INIT_GAIN_CLAMP_GLOSSY = 3.0f;
static const float RESTIR_TEMPORAL_INIT_GAIN_CLAMP_ROUGH  = 12.0f;
static const float RESTIR_TEMPORAL_REUSE_WEIGHT_CLAMP_GLOSSY = 8.0f;
static const float RESTIR_TEMPORAL_REUSE_WEIGHT_CLAMP_ROUGH  = 64.0f;
static const float RESTIR_TEMPORAL_REFLECTION_THRESHOLD_MIN = 0.90f;
static const float RESTIR_TEMPORAL_REFLECTION_THRESHOLD_MAX = 0.995f;

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
    seed_rng(rng, screenPos, g_Frame.frameIndex);

    // Accessing reservoirs bindless
    StructuredBuffer<Reservoir>    prevReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<Reservoir>  currReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, g_Frame, rng, surface, primaryRayT);
    if (!hasPrimaryHit) {
        currReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    // 1. Trace Single Indirect Bounce
    float3 rayDir;
    float3 throughput;
    float pdf;
    bool isDiffuse;
    SampleIndirectRay(surface.normal, surface.viewDir, surface.albedo, surface.metallic, surface.roughness, rng, rayDir, throughput, pdf, isDiffuse, true);
    bool isValidSample = pdf > 0.f && max(throughput.r, max(throughput.g, throughput.b)) > 0.f;
    
    bool hasFirstBounceCandidate = false;
    float3 continuationRadiance = 0.0f;
    float3 hitPos = 0.0f;
    float3 hitNormal = 0.0f;
    float firstBounceHitT = 0.0f;

    if (isValidSample)
    {
        RayDesc ray;
        ray.Origin = surface.worldPos + surface.normal * 0.001f;
        ray.Direction = rayDir;
        ray.TMin = 0.01f;
        ray.TMax = 1000.0f;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
        while (q.Proceed()) {
            PROCESS_ALPHA_MASK(q, rng);
        }

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) 
        {
            Surface hitSurf;
            ResolveHitSurface(ray, q.CommittedRayT(), q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), hitSurf, 0.15f);
            hitPos    = hitSurf.worldPos;
            hitNormal = hitSurf.normal;
            firstBounceHitT = q.CommittedRayT();
            float3 hitViewDir = hitSurf.viewDir;
            hasFirstBounceCandidate = true;

            // Evaluate continuation radiance from the sampled first-bounce candidate.
            // SHaRC provides multi-bounce continuation on a cache hit; on a miss we
            // currently fall back to direct lighting at the candidate only.
            float3 cachedRadiance = float3(0.0f, 0.0f, 0.0f);
            bool useCachedRadiance = false;
            {
                SharcParameters sharcParams;
                sharcParams.gridParameters.cameraPosition   = g_Frame.cameraPosition.xyz;
                sharcParams.gridParameters.logarithmBase    = SHARC_GRID_LOGARITHM_BASE;
                sharcParams.gridParameters.sceneScale       = g_Frame.sharcSceneScale;
                sharcParams.gridParameters.levelBias        = 0.0f;
                sharcParams.hashMapData.capacity            = SHARC_HASH_ENTRIES_NUM;
                sharcParams.hashMapData.hashEntriesBuffer   = ResourceDescriptorHeap[g_Sharc.HashEntriesBufIdx];
                sharcParams.accumulationBuffer              = ResourceDescriptorHeap[g_Sharc.AccumulationBufIdx];
                sharcParams.resolvedBuffer                  = ResourceDescriptorHeap[g_Sharc.ResolvedBufIdx];
                sharcParams.radianceScale                   = 1e3f;
                sharcParams.enableAntiFireflyFilter         = false;

                SharcHitData sharcQuery;
                sharcQuery.positionWorld = hitPos;
                sharcQuery.normalWorld   = hitNormal;

                float pathRoughness = isDiffuse ? 1.0f : surface.roughness;
                useCachedRadiance = IsSharcQueryValid(hitPos, q.CommittedRayT(), pathRoughness, sharcParams)
                    && SharcGetCachedRadiance(sharcParams, sharcQuery, cachedRadiance, false);
            }

            if (useCachedRadiance)
            {
                continuationRadiance = cachedRadiance;
            }
            else
            {
                float3 directLighting = GetDirectLightingHybrid(
                    hitPos, hitNormal, hitViewDir,
                    hitSurf.albedo, hitSurf.metallic, hitSurf.roughness,
                    g_Scene, g_Lights, g_Frame.numLights, g_Frame, true, rng);
                continuationRadiance = directLighting;
            }
        }
    }

    // 2. Create Initial Reservoir
    //Reservoir r = (Reservoir)0;
    Reservoir r;
    r.hitPos = 0; r.hitNormal = 0; r.radiance = 0;
    r.w_sum = 0; r.W = 0; r.M = 0; r.firstBounceHitT = 0; r.historyAge = 0;

    Surface s;
    s.worldPos = surface.worldPos;
    s.normal = surface.normal;
    s.viewDir = surface.viewDir;
    s.albedo = surface.albedo;
    s.metallic = surface.metallic;
    s.roughness = surface.roughness;

    float glossyFactor = GetGlossyFactor(surface.roughness);
    float3 currentReflectionDir = GetSurfaceReflectionDir(surface);

    float selectedPDF = 0.f;
    float debugSourcePdf = 0.0f;
    float debugTargetPdf = 0.0f;
    float debugRisWeight = 0.0f;
    float debugTemporalTargetPdf = 0.0f;
    if (hasFirstBounceCandidate)
    {
        float targetPDF = GetTargetPDF(s, hitPos, continuationRadiance, true);
        float targetShape = GetTargetShape(s, hitPos, true);
        float radianceLuma = max(1e-4f, Luminance(continuationRadiance));

        float proposalGain = (pdf > 0.0f) ? (targetShape / pdf) : 0.0f;

        if (g_Frame.enableReservoirLobeCheck != 0u) {
            float gainClamp = lerp(
                RESTIR_TEMPORAL_INIT_GAIN_CLAMP_ROUGH,
                RESTIR_TEMPORAL_INIT_GAIN_CLAMP_GLOSSY,
                glossyFactor);
            proposalGain = min(proposalGain, gainClamp);
        }

        float risWeight = proposalGain * radianceLuma;
        debugSourcePdf = pdf;
        debugTargetPdf = targetPDF;
        debugRisWeight = proposalGain;
        if (updateReservoir(r, hitPos, hitNormal, continuationRadiance, firstBounceHitT, risWeight, next_float(rng))) {
            selectedPDF = targetPDF;
            // Tag the initial sample with the lobe it was drawn from.
            r.historyAge = ReservoirPackAge(0u, !isDiffuse);
        }
    }

    // 3. Temporal Reuse
    if (g_Frame.frameIndex > 0u) {
        float4 prevClipPos = mul(float4(surface.worldPos, 1.0f), g_Frame.viewProjPrevious);
        prevClipPos /= prevClipPos.w;
        float2 prevUV = prevClipPos.xy * float2(0.5f, -0.5f) + 0.5f;

        if (prevUV.x >= 0.0f && prevUV.x < 1.0f && prevUV.y >= 0.0f && prevUV.y < 1.0f) {
            uint2 prevScreenPos = min((uint2)(prevUV * (float2)launchDims), launchDims - 1);
            Reservoir prevR = prevReservoirs[prevScreenPos.y * launchDims.x + prevScreenPos.x];

            if (prevR.M > 0.0f && ReservoirAge(prevR) < RESTIR_TEMPORAL_MAX_HISTORY_AGE) {
                RNG prevRng;
                seed_rng(prevRng, prevScreenPos, (g_Frame.frameIndex - 1u) + 911u);

                Surface prevSurface;
                float prevRayT;
                bool hasPrevHit = TracePrimarySurface(prevScreenPos, launchDims, g_Frame, prevRng, prevSurface, prevRayT, true);

                float expectedPrevRayT = length(surface.worldPos - g_Frame.prevCameraPosition.xyz);
                bool depthMatch = hasPrevHit
                    && abs(prevRayT - expectedPrevRayT) <= (RESTIR_TEMPORAL_DEPTH_THRESHOLD * max(1.0f, expectedPrevRayT));
                bool normalMatch = hasPrevHit
                    && dot(surface.normal, prevSurface.normal) > RESTIR_TEMPORAL_NORMAL_THRESHOLD;
                bool materialMatch = hasPrevHit
                    && AreMaterialsSimilar(s, prevSurface,
                        RESTIR_TEMPORAL_ALBEDO_THRESHOLD,
                        RESTIR_TEMPORAL_ROUGHNESS_THRESHOLD,
                        RESTIR_TEMPORAL_METALLIC_THRESHOLD);
                float3 prevReflectionDir = GetSurfaceReflectionDir(prevSurface);
                bool reflectionMatch = true;

                if (g_Frame.enableReservoirLobeCheck != 0u && glossyFactor > 0.0f) {
                    float reflectionThreshold = lerp(
                        RESTIR_TEMPORAL_REFLECTION_THRESHOLD_MIN,
                        RESTIR_TEMPORAL_REFLECTION_THRESHOLD_MAX,
                        glossyFactor);
                    reflectionMatch = dot(currentReflectionDir, prevReflectionDir) > reflectionThreshold;
                }

                if (depthMatch && normalMatch && materialMatch && reflectionMatch) {
                    bool prevIsSpecular = ReservoirIsSpecular(prevR);

                    float maxJac;
                    float maxHistory;
                    float historyTargetPDF = GetTargetPDF(s, prevR.hitPos, prevR.radiance, true);
                    if (g_Frame.enableReservoirLobeCheck != 0u) {
                        float glossyReuseFactor = glossyFactor;
                        if (prevIsSpecular) {
                            glossyReuseFactor = max(glossyReuseFactor, 0.5f);
                        }

                        maxJac = lerp(
                            RESTIR_TEMPORAL_MAX_JACOBIAN,
                            RESTIR_TEMPORAL_GLOSSY_MAX_JACOBIAN,
                            glossyReuseFactor);

                        maxHistory = lerp(
                            RESTIR_TEMPORAL_MAX_HISTORY_LENGTH,
                            RESTIR_TEMPORAL_GLOSSY_MAX_HISTORY,
                            glossyReuseFactor);
                    } else {
                        maxJac     = RESTIR_TEMPORAL_MAX_JACOBIAN;
                        maxHistory = RESTIR_TEMPORAL_MAX_HISTORY_LENGTH;
                    }
                    debugTemporalTargetPdf = historyTargetPDF;

                    float jacobian = ComputeJacobian(surface.worldPos, prevSurface.worldPos, prevR.hitPos, prevR.hitNormal);
                    bool jacobianValid = jacobian >= RESTIR_TEMPORAL_MIN_JACOBIAN && jacobian <= maxJac;

                    if (historyTargetPDF > 0.0f && jacobianValid) {
                        float shiftedTargetPDF = historyTargetPDF * jacobian;

                        Reservoir adjustedPrev = prevR;
                        capReservoirHistory(adjustedPrev, maxHistory);
                        adjustedPrev.historyAge = ReservoirPackAge(
                            min(ReservoirAge(prevR) + 1u, RESTIR_TEMPORAL_MAX_HISTORY_AGE),
                            prevIsSpecular);

                        float temporalReuseWeight = shiftedTargetPDF * adjustedPrev.W * adjustedPrev.M;

                        if (g_Frame.enableReservoirLobeCheck != 0u) {
                            float glossyReuseFactor = glossyFactor;
                            if (prevIsSpecular) {
                                glossyReuseFactor = max(glossyReuseFactor, 0.5f);
                            }

                            float reuseClamp = lerp(
                                RESTIR_TEMPORAL_REUSE_WEIGHT_CLAMP_ROUGH,
                                RESTIR_TEMPORAL_REUSE_WEIGHT_CLAMP_GLOSSY,
                                glossyReuseFactor);

                            temporalReuseWeight = min(temporalReuseWeight, reuseClamp);
                        }

                        if (mergeReservoirsWithWeight(r, adjustedPrev, temporalReuseWeight, next_float(rng))) {
                            selectedPDF = historyTargetPDF;
                            debugTemporalTargetPdf = historyTargetPDF * jacobian;
                        }
                    }
                }
            }

            // Apply the roughness-scaled cap to the final merged reservoir.
           float finalMaxHistory = RESTIR_TEMPORAL_MAX_HISTORY_LENGTH;
            if (g_Frame.enableReservoirLobeCheck != 0u) {
                float finalGlossyFactor = glossyFactor;
                if (ReservoirIsSpecular(r)) {
                    finalGlossyFactor = max(finalGlossyFactor, 0.5f);
                }

                finalMaxHistory = lerp(
                    RESTIR_TEMPORAL_MAX_HISTORY_LENGTH,
                    RESTIR_TEMPORAL_GLOSSY_MAX_HISTORY,
                    finalGlossyFactor);
            }
            capReservoirHistory(r, finalMaxHistory);
        }
    }

    // Normalize reservoir weight
    if (r.M > 0.0f && selectedPDF > 0.0f) {
        r.W = r.w_sum / (r.M * selectedPDF);
    } else {
        r.W = 0.0f;
    }

    if (g_Frame.restirReservoirDebugMode >= RESTIR_RESERVOIR_DEBUG_SOURCE_PDF)
    {
        RWTexture2D<float4> debugHeatmap = ResourceDescriptorHeap[g_Indices.OutputIdx1];
        float debugValue = 0.0f;
        switch (g_Frame.restirReservoirDebugMode)
        {
        case RESTIR_RESERVOIR_DEBUG_SOURCE_PDF:
            debugValue = debugSourcePdf;
            break;
        case RESTIR_RESERVOIR_DEBUG_TARGET_PDF:
            debugValue = debugTargetPdf;
            break;
        case RESTIR_RESERVOIR_DEBUG_RIS_WEIGHT:
            debugValue = debugRisWeight;
            break;
        case RESTIR_RESERVOIR_DEBUG_TEMPORAL_TARGET_PDF:
            debugValue = debugTemporalTargetPdf;
            break;
        default:
            debugValue = 0.0f;
            break;
        }

        debugHeatmap[screenPos] = float4(debugValue, 0.0f, 0.0f, 1.0f);
    }

    currReservoirs[pixelIndex] = r;
}
