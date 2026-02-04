#include "CommonTracing.hlsl"

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);
Texture2D g_Textures[] : register(t0, space0);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<LightConstants> g_Light : register(b1);

SamplerState g_LinearSampler : register(s0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    float2 uv = ((float2)launchIndex + 0.5f) / (float2)launchDims;
    float depth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, uv, 0).r;

    float3 accumulatedColor = 0;
    float3 indirectRadianceAccum = 0;
    float3 throughput = 1;

    if (depth >= 1.0f) {
        accumulatedColor = float3(0.5f, 0.7f, 1.0f) * 0.2f;
    } else {
        float3 primaryHitPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInverse);
        float3 primaryNormal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);
        float4 primaryAlbedo = g_Textures[g_Frame.albedoIndex].SampleLevel(g_LinearSampler, uv, 0);
        float4 primaryMaterial = g_Textures[g_Frame.materialIndex].SampleLevel(g_LinearSampler, uv, 0);
        float primaryRoughness = max(0.01f, primaryMaterial.r);
        float primaryMetallic = primaryMaterial.g;

        float3 V = normalize(g_Frame.cameraPosition.xyz - primaryHitPos);
        bool isPathDiffuse = false;

        // --- Step 1: Direct Lighting for Primary Hit (Raster Surface) ---
        accumulatedColor = GetDirectLighting(primaryHitPos, primaryNormal, V, primaryAlbedo.rgb, primaryMetallic, primaryRoughness, g_Scene, g_Light, g_Frame);

        // --- Step 2: Sample First Indirect Ray from Primary Hit ---
        float3 rayDir;
        float3 firstThroughput;
        float firstPDF;
        SampleIndirectRay(primaryNormal, V, primaryAlbedo.rgb, primaryMetallic, primaryRoughness, rng, rayDir, firstThroughput, firstPDF, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

        throughput *= firstThroughput;
        float3 rayPos = primaryHitPos + primaryNormal * 0.001f;

        // --- Step 3: Indirect Bounces ---
        for (int bounce = 1; bounce < 4; bounce++) {
            if (all(throughput <= 0.0f)) break;

            RayDesc ray;
            ray.Origin = rayPos;
            ray.Direction = rayDir;
            ray.TMin = 0.001f;
            ray.TMax = 10000.0f;

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

                float3 worldNormal = normalize(mul(v0.normal * (1.0f - barys.x - barys.y) + v1.normal * barys.x + v2.normal * barys.y, (float3x3)nodeData.world));
                float2 hitUv = v0.texCoord * (1.0f - barys.x - barys.y) + v1.texCoord * barys.x + v2.texCoord * barys.y;
                float3 hitPos = ray.Origin + ray.Direction * q.CommittedRayT();

                float4 albedo_hit = mat.baseColorFactor;
                if (mat.baseColorTextureIndex >= 0) albedo_hit *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);

                float metallic_hit = mat.metallicFactor;
                float roughness_hit = mat.roughnessFactor;
                if (mat.metallicRoughnessTextureIndex >= 0) {
                    float4 mrSample = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
                    roughness_hit *= mrSample.g; metallic_hit *= mrSample.b;
                }
                
                roughness_hit = max(0.15f, roughness_hit);
                float3 V_hit = -ray.Direction;

                // NEE
                indirectRadianceAccum += GetDirectLighting(hitPos, worldNormal, V_hit, albedo_hit.rgb, metallic_hit, roughness_hit, g_Scene, g_Light, g_Frame, isPathDiffuse) * throughput;

                // Sample next bounce
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
                indirectRadianceAccum += skyRadiance * throughput;
                break;
            }
        }
    }

    accumulatedColor += indirectRadianceAccum;

    // Progressive accumulation
    if (g_Frame.frameIndex <= 1) {
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = g_AccumulationBuffer[launchIndex].rgb;
        float n = (float)g_Frame.frameIndex;
        float lerpFactor = min( (n - 1.0f) / min(n, 2000.0f), 1.0f );  // Clamp to <= 1
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    float3 exposedColor = accumulatedColor * g_Frame.exposure;
    g_Output[launchIndex] = float4(exposedColor / (exposedColor + 1.0f), 1.0f);
}
