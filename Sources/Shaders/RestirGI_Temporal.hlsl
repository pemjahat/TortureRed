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

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);

RWStructuredBuffer<Reservoir> g_ReservoirCurrent : register(u2);  // Temporal output (ping-pong with Previous)
RWStructuredBuffer<Reservoir> g_ReservoirPrevious : register(u3); // Previous frame's temporal output

ConstantBuffer<FrameConstants> g_Frame : register(b0);
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_AccumulationBuffer.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    float2 uv = ((float2)launchIndex + 0.5f) / (float2)launchDims;
    float depth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, uv, 0).r;

    Reservoir res;
    res.hitPos = 0; res.hitNormal = 0; res.radiance = 0; res.targetPDF = 0;
    res.w_sum = 0; res.M = 0;

    float3 throughput = 1;
    float3 indirectRadianceAccum = 0;
    
    Surface surface;
    bool hasPrimaryHit = false;

    float3 indirectHitPos = 0, indirectHitNormal = 0;
    bool hasIndirectHit = false;
    float firstBouncePDF = 1.0f;
    float3 firstBounceThroughput = 1.0f;
    bool isPathDiffuse = false;

    if (depth < 1.0f) {
        surface.worldPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInverse);
        surface.normal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);
        surface.viewDir = normalize(g_Frame.cameraPosition.xyz - surface.worldPos);
        
        hasPrimaryHit = true;
        surface.albedo = g_Textures[g_Frame.albedoIndex].SampleLevel(g_LinearSampler, uv, 0).rgb;
        float4 materialProps = g_Textures[g_Frame.materialIndex].SampleLevel(g_LinearSampler, uv, 0);
        surface.roughness = max(0.01f, materialProps.r);
        surface.metallic = materialProps.g;

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
                uint instanceIdx = q.CommittedInstanceID();
                uint triIdx = q.CommittedPrimitiveIndex();
                float2 barys = q.CommittedTriangleBarycentrics();
                DrawNodeData nodeData = g_DrawNodeBuffer[instanceIdx];
                MaterialConstants mat = g_Materials[nodeData.materialID];

                uint i0 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 0];
                uint i1 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 1];
                uint i2 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 2];
                GLTFVertex v0 = g_GlobalVertices[nodeData.vertexOffset + i0];
                GLTFVertex v1 = g_GlobalVertices[nodeData.vertexOffset + i1];
                GLTFVertex v2 = g_GlobalVertices[nodeData.vertexOffset + i2];

                float3 worldNormal = normalize(mul(v0.normal * (1.0f-barys.x-barys.y) + v1.normal * barys.x + v2.normal * barys.y, (float3x3)nodeData.world));
                float2 uv_hit = v0.texCoord * (1.0f-barys.x-barys.y) + v1.texCoord * barys.x + v2.texCoord * barys.y;
                float3 hitPos = ray.Origin + ray.Direction * q.CommittedRayT();

                float4 albedo_hit = mat.baseColorFactor;
                if (mat.baseColorTextureIndex >= 0) albedo_hit *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, uv_hit, 0);

                float metallic_hit = mat.metallicFactor;
                float roughness_hit = mat.roughnessFactor;
                if (mat.metallicRoughnessTextureIndex >= 0) {
                    float4 mrSample = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, uv_hit, 0);
                    roughness_hit *= mrSample.g; metallic_hit *= mrSample.b;
                }
                roughness_hit = max(0.15f, roughness_hit); // Indirect stability

                float3 V_hit = -ray.Direction;

                if (bounce == 1) {
                    indirectHitPos = hitPos; indirectHitNormal = worldNormal; hasIndirectHit = true;
                }

                // NEE: RIS light sampling — 4 candidates on first bounce, 1 on deeper bounces.
                // Only 1 shadow ray fired for the winner; O(1) cost regardless of light count.
                uint risCandidates = (bounce == 1) ? 4 : 1;
                indirectRadianceAccum += GetDirectLightingRIS(
                    hitPos, worldNormal, V_hit,
                    albedo_hit.rgb, metallic_hit, roughness_hit,
                    g_Scene, g_Lights, g_Frame.numLights,
                    g_Frame, rng, isPathDiffuse, risCandidates) * throughput;

                // Path continuation
                float3 nextDir;
                float3 nextThroughput;
                float next_pdf;
                SampleIndirectRay(worldNormal, V_hit, albedo_hit.rgb, metallic_hit, roughness_hit, rng, nextDir, nextThroughput, next_pdf, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

                throughput *= nextThroughput;
                rayPos = hitPos + worldNormal * 0.001f;
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
            if (updateReservoir(res, indirectHitPos, indirectHitNormal, L_in, targetPDF, firstBouncePDF, next_float(rng))) {
                selectedTargetPdf = targetPDF;
            }
        }

        if (g_Frame.frameIndex > 1) {
            float4 clipPos = mul(float4(surface.worldPos, 1.0f), g_Frame.viewProjPrevious);
            float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f; prevUV.y = 1.0f - prevUV.y;
            if (prevUV.x >= 0 && prevUV.x <= 1 && prevUV.y >= 0 && prevUV.y <= 1) {
                uint2 prevIndex = (uint2)(prevUV * (float2)launchDims);
                Reservoir prevRes = g_ReservoirPrevious[prevIndex.y * launchDims.x + prevIndex.x];
                
                if (prevRes.M > 0) {
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
             res.w_sum = res.w_sum / (res.M * selectedTargetPdf);
         } else {
             res.w_sum = 0;
         }
    }

    g_ReservoirCurrent[launchIndex.y * launchDims.x + launchIndex.x] = res;
}
