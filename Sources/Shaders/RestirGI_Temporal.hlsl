#include "CommonTracing.hlsl"

// My notes on Restir
// Temporal pass, intentionally pick brightest sample, and set probability proportional to targetPDF
// Because "biasing" toward bright samples, radiance of winning sample is statically too high
// To fix this, since probability of selecting sample Y is proportional to targetPDF(Y), we just need divide by probability.

// 1/targetpdf is just normalizing probability of selecting that specific sample
// However we need normalize how many other good sample available in the search space
// if you find one bright light in dark room, w_sum will be small
// if you find one bright light in room full of bright light, w_sum will be large
// even you pick same bright sample, final contribution differs because "density" light in that area differs, w_sum/M capture this density information

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    // Accessing texture bindless
    StructuredBuffer<Reservoir> prevReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<Reservoir> currReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Reservoir res;
    res.hitPos = 0; res.hitNormal = 0; res.radiance = 0;
    res.w_sum = 0; res.W = 0; res.M = 0;

    float3 throughput = 1;
    float3 indirectRadianceAccum = 0;
    
    Surface surface;
    bool hasPrimaryHit = false;

    float3 indirectHitPos = 0, indirectHitNormal = 0;
    bool hasIndirectHit = false;
    float3 bounce2HitPos = 0; bool hasBounce2 = false;
    float3 bounce3HitPos = 0; bool hasBounce3 = false;
    float3 temporalSampleHitPos = 0; bool hasTemporalSample = false;
    float firstBouncePDF = 1.0f;
    float3 firstBounceThroughput = 1.0f;
    bool isPathDiffuse = false;

    float primaryRayT;
    hasPrimaryHit = TracePrimarySurface(launchIndex, launchDims, g_Frame, rng, surface, primaryRayT);

    if (hasPrimaryHit) {

        // Generate first indirect bounce
        float3 nextRayDir;
        SampleIndirectRay(surface.normal, surface.viewDir, surface.albedo, surface.metallic, surface.roughness, rng, nextRayDir, firstBounceThroughput, firstBouncePDF, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

        // Reset throughput to 1.0 for incident radiance accumulation
        throughput = 1.0f;
        float3 rayPos = surface.worldPos + surface.normal * 0.001f;
        float3 rayDir = nextRayDir;
        
        // --- Path Tracing for Indirect Light ---
        for (int bounce = 1; bounce < 4; bounce++) {
            RayDesc ray;
            ray.Origin = rayPos; ray.Direction = rayDir;
            ray.TMin = 0.001f; ray.TMax = 10000.0f;

            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
            while (q.Proceed()) {
                PROCESS_ALPHA_MASK(q, rng);
            }

            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                Surface hitSurf;
                // 0.15f minRoughness for indirect stability
                ResolveHitSurface(ray, q.CommittedRayT(), q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), hitSurf, 0.15f);

                if (bounce == 1) {
                    indirectHitPos = hitSurf.worldPos; indirectHitNormal = hitSurf.normal; hasIndirectHit = true;
                }
                if (bounce == 2) { bounce2HitPos = hitSurf.worldPos; hasBounce2 = true; }
                if (bounce == 3) { bounce3HitPos = hitSurf.worldPos; hasBounce3 = true; }

                // NEE: RIS light sampling — 4 candidates on first bounce, 1 on deeper bounces.
                // Only 1 shadow ray fired for the winner; O(1) cost regardless of light count.
                uint risCandidates = (bounce == 1) ? 4 : 1;
                indirectRadianceAccum += GetDirectLightingRIS(
                    hitSurf.worldPos, hitSurf.normal, hitSurf.viewDir,
                    hitSurf.albedo, hitSurf.metallic, hitSurf.roughness,
                    g_Scene, g_Lights, g_Frame.numLights,
                    g_Frame, rng, isPathDiffuse, risCandidates) * throughput;

                // Path continuation
                float3 nextDir;
                float3 nextThroughput;
                float next_pdf;
                SampleIndirectRay(hitSurf.normal, hitSurf.viewDir, hitSurf.albedo, hitSurf.metallic, hitSurf.roughness, rng, nextDir, nextThroughput, next_pdf, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

                throughput *= nextThroughput;
                rayPos = hitSurf.worldPos + hitSurf.normal * 0.001f;
                rayDir = nextDir;
                
                // Russian Roulette
                if (bounce > 2) {
                    float p = max(throughput.r, max(throughput.g, throughput.b));
                    if (next_float(rng) > p) break;
                    throughput /= p;
                }
            } else {
                float3 skyRadiance = float3(0.5f, 0.7f, 1.0f) * 0.2f;
                if (bounce == 1) {
                    indirectHitPos = ray.Origin + ray.Direction * 1000.0f;
                    indirectHitNormal = -ray.Direction;
                    hasIndirectHit = true;
                }
                if (bounce == 2) { bounce2HitPos = ray.Origin + ray.Direction * 1000.0f; hasBounce2 = true; }
                if (bounce == 3) { bounce3HitPos = ray.Origin + ray.Direction * 1000.0f; hasBounce3 = true; }
                indirectRadianceAccum += skyRadiance * throughput;
                break;
            }
        }
    }

    // --- Temporal Merging ---
    float selectedTargetPdf = 0;
    if (hasPrimaryHit) {
        if (hasIndirectHit) {            
            // Radiance is now incident radiance by design, no division needed
            float3 L_in = indirectRadianceAccum;
            
            // Use GetTargetPDF to determine weight for initial sample
            float targetPDF = GetTargetPDF(surface, indirectHitPos, L_in);
            float risWeight = (firstBouncePDF > 0.0f) ? (targetPDF / firstBouncePDF) : 0.0f;
            if (updateReservoir(res, indirectHitPos, indirectHitNormal, L_in, risWeight, next_float(rng))) {
                selectedTargetPdf = targetPDF;
            }
        }

        if (g_Frame.frameIndex > 1) {
            float4 clipPos = mul(float4(surface.worldPos, 1.0f), g_Frame.viewProjPrevious);
            float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f; prevUV.y = 1.0f - prevUV.y;
            if (prevUV.x >= 0 && prevUV.x <= 1 && prevUV.y >= 0 && prevUV.y <= 1) {
                uint2 prevIndex = (uint2)(prevUV * (float2)launchDims);
                Reservoir prevRes = prevReservoirs[prevIndex.y * launchDims.x + prevIndex.x];
                
                if (prevRes.M > 0) {
                    // Capture previous sample position before merging (for path viz)
                    temporalSampleHitPos = prevRes.hitPos; hasTemporalSample = true;

                    // Re-calculate target PDF of previous sample relative to CURRENT surface
                    float currentTargetPDF = GetTargetPDF(surface, prevRes.hitPos, prevRes.radiance);

                     if (currentTargetPDF > 0) {
                         // Apply Jacobian to temporal reservoir's weight (matches RTXDI)
                         // This corrects for solid-angle change when reprojecting from previous to current pixel
                        //  float3 prevWorldPos = ReconstructWorldPos(prevUV, 
                        //      g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, prevUV, 0).r,
                        //      g_Frame.projectionInverse, g_Frame.viewInversePrevious);
                        //  float jacobian = ComputeJacobian(surface.worldPos, prevWorldPos, prevRes.hitPos, prevRes.hitNormal);
                        //  if (jacobian <= 0 || jacobian > 64.0f || isnan(jacobian) || isinf(jacobian)) {
                        //      // Reject sample with extreme Jacobian (matches RTXDI RAB_ValidateGISampleWithJacobian)
                        //  } else {
                             // Clamp M before combine (matches RTXDI: cap history length)
                             //prevRes.M = min(prevRes.M, 30.0f);
                             //prevRes.w_sum *= jacobian;

                             if (mergeReservoirs(res, prevRes, currentTargetPDF, next_float(rng)))
                             {
                                 selectedTargetPdf = currentTargetPDF;
                             }
                         //}
                     }
                    
                     if (res.M > 30.0f) { 
                         res.w_sum *= (30.0f / res.M); 
                         res.M = 30.0f; 
                     }
                }
            }
        }

        // Final Normalization: Re-evaluate target PDF for the winning sample
         if (selectedTargetPdf > 0 && res.M > 0) {
             res.W = res.w_sum / (res.M * selectedTargetPdf);
         } else {
             res.W = 0;
         }
    }

    currReservoirs[launchIndex.y * launchDims.x + launchIndex.x] = res;

    // --- Path Visualization Recording ---
    // Triggered for exactly one frame by the left-click; writes world-space line segments
    // into a small fixed-slot buffer that the PathVizLines VS reads each frame.
    if (g_Frame.pathVizEnabled != 0 &&
        launchIndex.x == g_Frame.mouseSelectedPixelX &&
        launchIndex.y == g_Frame.mouseSelectedPixelY)
    {
        RWStructuredBuffer<PathVizLine> g_PathVizLines = ResourceDescriptorHeap[g_Indices.PathVizLineBufferIdx];

        // Invalidate all slots so stale data from a previous click doesn't bleed through
        for (int s = 0; s < MAX_PATH_VIZ_LINES; s++) {
            PathVizLine blank;
            blank.start = float3(0, 0, 0); blank.typeAndValid = 0;
            blank.end   = float3(0, 0, 0); blank._pad = 0;
            g_PathVizLines[s] = blank;
        }

        PathVizLine debugLine;
        debugLine._pad = 0;

        // Line 0: Surface normal at primary hit
        if (hasPrimaryHit) {
            debugLine.start = surface.worldPos;
            debugLine.end   = surface.worldPos + surface.normal * 0.3f;
            debugLine.typeAndValid = PATHVIZ_TYPE_PRIMARY | (1u << 4);
            g_PathVizLines[0] = debugLine;
        }
        // Line 1: First indirect bounce — first surface to bounce-1 hit
        if (hasIndirectHit) {
            debugLine.start = surface.worldPos;
            debugLine.end   = indirectHitPos;
            debugLine.typeAndValid = PATHVIZ_TYPE_BOUNCE1 | (1u << 4);
            g_PathVizLines[1] = debugLine;
        }
        // Line 2: Second bounce
        if (hasBounce2) {
            debugLine.start = indirectHitPos;
            debugLine.end   = bounce2HitPos;
            debugLine.typeAndValid = PATHVIZ_TYPE_BOUNCE2 | (1u << 4);
            g_PathVizLines[2] = debugLine;
        }
        // Line 3: Third bounce
        if (hasBounce3) {
            debugLine.start = bounce2HitPos;
            debugLine.end   = bounce3HitPos;
            debugLine.typeAndValid = PATHVIZ_TYPE_BOUNCE3 | (1u << 4);
            g_PathVizLines[3] = debugLine;
        }
        // Line 4: Temporal reuse — current surface to the temporally reused sample
        if (hasTemporalSample) {
            debugLine.start = surface.worldPos;
            debugLine.end   = temporalSampleHitPos;
            debugLine.typeAndValid = PATHVIZ_TYPE_TEMPORAL | (1u << 4);
            g_PathVizLines[4] = debugLine;
        }
    }
}
