// RestirGI_Diffuse_Temporal.hlsl — RTDGI stream
// Hemisphere-sampled diffuse indirect GI with long temporal history.
// Writes diffuse candidate data to an intermediate buffer for RTR rough-surface reuse.

#include "SharcCommon.h"
#include "CommonTracing.hlsl"
#include "SHaRC_Integration.hlsl"

ConstantBuffer<FrameConstants>       g_Frame   : register(b0);
ConstantBuffer<BindlessIndices>      g_Indices : register(b1);
ConstantBuffer<SharcBindlessIndices> g_Sharc   : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

// Diffuse-optimized constants
static const float RESTIR_TEMPORAL_DEPTH_THRESHOLD = 0.1f;
static const float RESTIR_TEMPORAL_NORMAL_THRESHOLD = 0.95f;
static const float RESTIR_TEMPORAL_ALBEDO_THRESHOLD = 0.15f;
static const float RESTIR_TEMPORAL_ROUGHNESS_THRESHOLD = 0.15f;
static const float RESTIR_TEMPORAL_METALLIC_THRESHOLD = 0.15f;
static const float RESTIR_TEMPORAL_MAX_HISTORY_LENGTH = 16.0f;
static const uint  RESTIR_TEMPORAL_MAX_HISTORY_AGE = 12u;
static const float RESTIR_TEMPORAL_MAX_JACOBIAN = 10.0f;
static const float RESTIR_TEMPORAL_MIN_JACOBIAN = 0.1f;
static const float RESTIR_TEMPORAL_INIT_GAIN_CLAMP = 12.0f;
static const float RESTIR_TEMPORAL_REUSE_WEIGHT_CLAMP = 64.0f;

// Force diffuse-only sampling: cosine-weighted hemisphere
void SampleDiffuseRay(float3 N, float3 V, float3 baseColor, float metallic, inout RNG rng,
                      out float3 rayDir, out float3 throughput, out float pdf)
{
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float3 nextDirLocal = sample_cosine_weighted(float2(next_float(rng), next_float(rng)));
    rayDir = align_to_normal(nextDirLocal, N);
    float3 H = normalize(V + rayDir);
    float3 F_at_surface = FresnelSchlick(max(dot(V, H), 0.0), F0);
    float3 kD = (1.0 - F_at_surface) * (1.0 - metallic);
    throughput = kD * baseColor;
    pdf = max(dot(N, rayDir), 0.0f) / 3.14159265f;
}

// Diffuse-only target PDF (no specular contribution)
float GetDiffuseTargetPDF(Surface s, float3 samplePos, float3 sampleRadiance)
{
    float3 L = normalize(samplePos - s.worldPos);
    float dotNL = max(0.0f, dot(s.normal, L));
    if (dotNL <= 0) return 0;
    float3 d, spec;
    EvaluateBSDF(s.normal, s.viewDir, L, s.albedo, s.metallic, s.roughness, d, spec);
    float3 reflected = d * sampleRadiance * dotNL;
    return max(0.0f, Luminance(reflected));
}

float GetDiffuseTargetShape(Surface s, float3 samplePos)
{
    float3 L = normalize(samplePos - s.worldPos);
    float dotNL = max(0.0f, dot(s.normal, L));
    if (dotNL <= 0.0f) return 0.0f;
    float3 d, spec;
    EvaluateBSDF(s.normal, s.viewDir, L, s.albedo, s.metallic, s.roughness, d, spec);
    return max(0.0f, Luminance(d * dotNL));
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex);

    // InputIdx0 = previous diffuse reservoirs (SRV)
    // OutputIdx0 = current diffuse reservoirs (UAV)
    // OutputIdx1 = diffuse candidate buffer (UAV) — for RTR reuse
    // OutputIdx2 = debug heatmap (UAV, optional)
    StructuredBuffer<Reservoir>       prevReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<Reservoir>     currReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWStructuredBuffer<DiffuseCandidate> candidateOut = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, g_Frame, rng, surface, primaryRayT);
    if (!hasPrimaryHit) {
        currReservoirs[pixelIndex] = (Reservoir)0;
        DiffuseCandidate invalidCandidate = (DiffuseCandidate)0;
        invalidCandidate.hitT = -1.0f;
        candidateOut[pixelIndex] = invalidCandidate;
        return;
    }

    // 1. Trace diffuse indirect bounce (hemisphere sampling only)
    float3 rayDir;
    float3 throughput;
    float pdf;
    SampleDiffuseRay(surface.normal, surface.viewDir, surface.albedo, surface.metallic, rng, rayDir, throughput, pdf);
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
            hitPos = hitSurf.worldPos;
            hitNormal = hitSurf.normal;
            firstBounceHitT = q.CommittedRayT();
            hasFirstBounceCandidate = true;

            // SHaRC continuation or direct lighting fallback
            float3 cachedRadiance = 0.0f;
            bool useCachedRadiance = false;
            {
                SharcParameters sharcParams;
                sharcParams.gridParameters.cameraPosition = g_Frame.cameraPosition.xyz;
                sharcParams.gridParameters.logarithmBase  = SHARC_GRID_LOGARITHM_BASE;
                sharcParams.gridParameters.sceneScale     = g_Frame.sharcSceneScale;
                sharcParams.gridParameters.levelBias      = 0.0f;
                sharcParams.hashMapData.capacity          = SHARC_HASH_ENTRIES_NUM;
                sharcParams.hashMapData.hashEntriesBuffer = ResourceDescriptorHeap[g_Sharc.HashEntriesBufIdx];
                sharcParams.accumulationBuffer            = ResourceDescriptorHeap[g_Sharc.AccumulationBufIdx];
                sharcParams.resolvedBuffer                = ResourceDescriptorHeap[g_Sharc.ResolvedBufIdx];
                sharcParams.radianceScale                 = 1e3f;
                sharcParams.enableAntiFireflyFilter       = false;

                SharcHitData sharcQuery;
                sharcQuery.positionWorld = hitPos;
                sharcQuery.normalWorld   = hitNormal;

                // Diffuse path roughness = 1.0 (always)
                useCachedRadiance = IsSharcQueryValid(hitPos, q.CommittedRayT(), 1.0f, sharcParams)
                    && SharcGetCachedRadiance(sharcParams, sharcQuery, cachedRadiance, false);
            }

            if (useCachedRadiance)
                continuationRadiance = cachedRadiance;
            else
            {
                continuationRadiance = GetDirectLightingHybrid(
                    hitPos, hitNormal, hitSurf.viewDir,
                    hitSurf.albedo, hitSurf.metallic, hitSurf.roughness,
                    g_Scene, g_Lights, g_Frame.numLights, g_Frame, true, rng);
            }
        }
    }

    // Write diffuse candidate for RTR rough-surface reuse
    DiffuseCandidate candidate;
    if (hasFirstBounceCandidate) {
        candidate.hitPos = hitPos;
        candidate.hitNormal = hitNormal;
        candidate.radiance = continuationRadiance;
        candidate.hitT = firstBounceHitT;
    } else {
        candidate = (DiffuseCandidate)0;
        candidate.hitT = -1.0f;
    }
    candidate._pad0 = 0;
    candidate._pad1 = 0;
    candidateOut[pixelIndex] = candidate;

    // 2. Create initial reservoir
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

    float selectedPDF = 0.f;
    if (hasFirstBounceCandidate)
    {
        float targetPDF = GetDiffuseTargetPDF(s, hitPos, continuationRadiance);
        float targetShape = GetDiffuseTargetShape(s, hitPos);
        float radianceLuma = max(1e-4f, Luminance(continuationRadiance));

        float proposalGain = (pdf > 0.0f) ? (targetShape / pdf) : 0.0f;
        proposalGain = min(proposalGain, RESTIR_TEMPORAL_INIT_GAIN_CLAMP);

        float risWeight = proposalGain * radianceLuma;
        if (updateReservoir(r, hitPos, hitNormal, continuationRadiance, firstBounceHitT, risWeight, next_float(rng))) {
            selectedPDF = targetPDF;
            r.historyAge = ReservoirPackAge(0u, false); // Always diffuse
        }
    }

    // 3. Temporal reuse (diffuse-optimized: no reflection-direction check)
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

                // No reflection-direction check for diffuse — all hemisphere samples are valid

                if (depthMatch && normalMatch && materialMatch) {
                    float historyTargetPDF = GetDiffuseTargetPDF(s, prevR.hitPos, prevR.radiance);

                    float jacobian = ComputeJacobian(surface.worldPos, prevSurface.worldPos, prevR.hitPos, prevR.hitNormal);
                    bool jacobianValid = jacobian >= RESTIR_TEMPORAL_MIN_JACOBIAN && jacobian <= RESTIR_TEMPORAL_MAX_JACOBIAN;

                    if (historyTargetPDF > 0.0f && jacobianValid) {
                        float shiftedTargetPDF = historyTargetPDF * jacobian;

                        Reservoir adjustedPrev = prevR;
                        capReservoirHistory(adjustedPrev, RESTIR_TEMPORAL_MAX_HISTORY_LENGTH);
                        adjustedPrev.historyAge = ReservoirPackAge(
                            min(ReservoirAge(prevR) + 1u, RESTIR_TEMPORAL_MAX_HISTORY_AGE), false);

                        float temporalReuseWeight = shiftedTargetPDF * adjustedPrev.W * adjustedPrev.M;
                        temporalReuseWeight = min(temporalReuseWeight, RESTIR_TEMPORAL_REUSE_WEIGHT_CLAMP);

                        if (mergeReservoirsWithWeight(r, adjustedPrev, temporalReuseWeight, next_float(rng))) {
                            selectedPDF = historyTargetPDF;
                        }
                    }
                }
            }

            capReservoirHistory(r, RESTIR_TEMPORAL_MAX_HISTORY_LENGTH);
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
        RWTexture2D<float4> debugHeatmap = ResourceDescriptorHeap[g_Indices.OutputIdx2];
        debugHeatmap[screenPos] = float4(selectedPDF, 0.0f, 0.0f, 1.0f);
    }

    currReservoirs[pixelIndex] = r;
}
