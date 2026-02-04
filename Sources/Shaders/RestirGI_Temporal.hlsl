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
Texture2D g_Textures[] : register(t0, space0);

RWStructuredBuffer<Reservoir> g_ReservoirIntermediate : register(u2);
RWStructuredBuffer<Reservoir> g_ReservoirPrevious : register(u3);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<LightConstants> g_Light : register(b1);

SamplerState g_LinearSampler : register(s0);

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
    res.w_sum = 0; res.M = 0; res.W = 0;

    float3 throughput = 1;
    float3 indirectRadianceAccum = 0;
    
    float3 primaryHitPos = 0, primaryHitNormal = 0, primaryV = 0, primaryAlbedo = 0;
    float primaryRoughness = 0, primaryMetallic = 0;
    bool hasPrimaryHit = false;

    float3 indirectHitPos = 0, indirectHitNormal = 0;
    bool hasIndirectHit = false;
    float firstBouncePDF = 1.0f;
    float3 firstBounceThroughput = 1.0f;
    bool isPathDiffuse = false;

    if (depth < 1.0f) {
        primaryHitPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInverse);
        primaryHitNormal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);
        primaryV = normalize(g_Frame.cameraPosition.xyz - primaryHitPos);
        
        hasPrimaryHit = true;
        primaryAlbedo = g_Textures[g_Frame.albedoIndex].SampleLevel(g_LinearSampler, uv, 0).rgb;
        float4 materialProps = g_Textures[g_Frame.materialIndex].SampleLevel(g_LinearSampler, uv, 0);
        primaryRoughness = max(0.01f, materialProps.r);
        primaryMetallic = materialProps.g;

        // Generate first indirect bounce
        float3 nextRayDir;
        SampleIndirectRay(primaryHitNormal, primaryV, primaryAlbedo, primaryMetallic, primaryRoughness, rng, nextRayDir, firstBounceThroughput, firstBouncePDF, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

        throughput *= firstBounceThroughput;
        float3 rayPos = primaryHitPos + primaryHitNormal * 0.001f;
        float3 rayDir = nextRayDir;
        
        // --- Path Tracing for Indirect Light ---
        for (int bounce = 1; bounce < 4; bounce++) {
            RayDesc ray;
            ray.Origin = rayPos; ray.Direction = rayDir;
            ray.TMin = 0.001f; ray.TMax = 10000.0f;

            RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
            q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
            q.Proceed();

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

                // NEE
                indirectRadianceAccum += GetDirectLighting(hitPos, worldNormal, V_hit, albedo_hit.rgb, metallic_hit, roughness_hit, g_Scene, g_Light, g_Frame, isPathDiffuse) * throughput;

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
    if (hasPrimaryHit) {
        if (hasPrimaryHit) {            
            // We want res.radiance * res.W to be an unbiased estimator of L_in
            // throughput = (BRDF * Cos) / PDF. 
            // L_total = L_in * throughput
            // Therefore, L_in = L_total / throughput            
            float3 L_in_unbiased = indirectRadianceAccum / max(0.0001f, firstBounceThroughput);
            
            float targetPDF = Luminance(L_in_unbiased);
            float weight = targetPDF / max(1e-6f, firstBouncePDF); 
            
            updateReservoir(res, indirectHitPos, indirectHitNormal, L_in_unbiased, targetPDF, weight, next_float(rng));
        }

        if (g_Frame.frameIndex > 1) {
            float4 clipPos = mul(float4(primaryHitPos, 1.0f), g_Frame.viewProjPrevious);
            float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f; prevUV.y = 1.0f - prevUV.y;
            if (prevUV.x >= 0 && prevUV.x <= 1 && prevUV.y >= 0 && prevUV.y <= 1) {
                uint2 prevIndex = (uint2)(prevUV * (float2)launchDims);
                Reservoir prevRes = g_ReservoirPrevious[prevIndex.y * launchDims.x + prevIndex.x];
                
                if (prevRes.M > 0 && prevRes.targetPDF > 0) {
                    // Merging with confidence-weighted RIS
                    float weight = prevRes.W * prevRes.M * prevRes.targetPDF;
                    mergeReservoirs(res, prevRes, prevRes.targetPDF, weight, next_float(rng));
                    
                    if (res.M > 30.0f) { 
                        res.w_sum *= (30.0f / res.M); 
                        res.M = 30.0f; 
                    }
                }
            }
        }

        // Final Normalization with Bias Fix (avoid using 1.0 as epsilon)
        if (res.targetPDF > 0) 
            res.W = res.w_sum / max(1e-6f, res.M * res.targetPDF); 
        else 
            res.W = 0;
    }

    g_ReservoirIntermediate[launchIndex.y * launchDims.x + launchIndex.x] = res;
}
