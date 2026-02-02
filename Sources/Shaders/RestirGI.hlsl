#include "CommonTracing.hlsl"

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);
RaytracingAccelerationStructure g_Scene : register(t2, space1);
StructuredBuffer<DrawNodeData> g_DrawNodeBuffer : register(t1, space1);
StructuredBuffer<MaterialConstants> g_Materials : register(t0, space1);
StructuredBuffer<GLTFVertex> g_GlobalVertices : register(t4, space1);
StructuredBuffer<uint> g_GlobalIndices : register(t3, space1);
ByteAddressBuffer g_Buffers[] : register(t0, space2);
Texture2D g_Textures[] : register(t0, space0);

RWStructuredBuffer<Reservoir> g_ReservoirCurrent : register(u2);
RWStructuredBuffer<Reservoir> g_ReservoirPrevious : register(u3);

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

    float2 subpixel = float2(next_float(rng), next_float(rng));
    float2 d = (((float2)launchIndex + subpixel) / (float2)launchDims) * 2.0f - 1.0f;
    d.y = -d.y;

    float4 targetRay = mul(float4(d, 1.0f, 1.0f), g_Frame.projectionInverse);
    float3 rayDir = mul(float4(normalize(targetRay.xyz / targetRay.w), 0.0f), g_Frame.viewInverse).xyz;
    float3 rayPos = g_Frame.cameraPosition.xyz;

    Reservoir res;
    res.hitPos = 0; res.hitNormal = 0; res.radiance = 0; res.targetPDF = 0;
    res.w_sum = 0; res.M = 0; res.W = 0; res.primaryPos = 0; res.primaryNormal = 0;

    float3 accumulatedColor = 0;
    float3 throughput = 1;
    float3 indirectRadianceAccum = 0;
    
    float3 primaryHitPos = 0, primaryHitNormal = 0, primaryV = 0, primaryBaseColor = 0;
    float primaryMetallic = 0, primaryRoughness = 1.0f;
    bool hasPrimaryHit = false;

    float3 indirectHitPos = 0, indirectHitNormal = 0;
    bool hasIndirectHit = false;
    float firstBouncePDF = 1.0f;
    float3 firstBounceThroughput = 1.0f;
    bool isPathDiffuse = false;

    for (int bounce = 0; bounce < 4; bounce++) {
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
            float2 uv = v0.texCoord * (1.0f-barys.x-barys.y) + v1.texCoord * barys.x + v2.texCoord * barys.y;
            float3 hitPos = ray.Origin + ray.Direction * q.CommittedRayT();

            float4 baseColor = mat.baseColorFactor;
            if (mat.baseColorTextureIndex >= 0) baseColor *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, uv, 0);

            float metallic = mat.metallicFactor;
            float roughness = mat.roughnessFactor;
            if (mat.metallicRoughnessTextureIndex >= 0) {
                float4 mrSample = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, uv, 0);
                roughness *= mrSample.g; metallic *= mrSample.b;
            }
            roughness = (bounce > 0) ? max(0.15f, roughness) : max(0.01f, roughness);

            float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
            float3 V = -ray.Direction;

            if (bounce == 0) {
                primaryHitPos = hitPos; primaryHitNormal = worldNormal; primaryV = V;
                primaryBaseColor = baseColor.rgb; primaryMetallic = metallic; primaryRoughness = roughness;
                hasPrimaryHit = true;
            } else if (bounce == 1) {
                indirectHitPos = hitPos; indirectHitNormal = worldNormal; hasIndirectHit = true;
            }

            // NEE
            {
                float3 L_light = -normalize(g_Light.direction.xyz);
                float ndotl = max(0.0001f, dot(worldNormal, L_light));
                if (ndotl > 0) {
                    RayDesc shadowRay;
                    shadowRay.Origin = hitPos + worldNormal * 0.001f;
                    shadowRay.Direction = L_light;
                    shadowRay.TMin = 0.001f; shadowRay.TMax = 10000.0f;
                    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
                    sq.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, shadowRay);
                    sq.Proceed();

                    if (sq.CommittedStatus() == COMMITTED_NOTHING) {
                        float3 diffuse, specular;
                        EvaluateBSDF(worldNormal, V, L_light, baseColor.rgb, metallic, roughness, diffuse, specular);
                        if ((bounce > 0 && !g_Frame.enableIndirectSpecular) || (isPathDiffuse && g_Frame.enableAvoidCaustics)) specular = 0;
                        float3 directLight = (diffuse + specular) * g_Light.color.rgb * g_Light.intensity * ndotl;
                        if (bounce == 0) accumulatedColor += directLight;
                        else indirectRadianceAccum += directLight * throughput;
                    }
                }
            }

            // Scattering
            float3 F_prob = FresnelSchlick(max(dot(worldNormal, V), 0.0), F0);
            float probSpecular = clamp(max(F_prob.r, max(F_prob.g, F_prob.b)), 0.1, 0.9);
            float rndCont = next_float(rng);
            float3 throughputFactor = 1.0f;
            float samplePDF = 1.0f;

            if (rndCont < probSpecular) {
                if ((bounce > 0 && !g_Frame.enableIndirectSpecular) || (isPathDiffuse && g_Frame.enableAvoidCaustics)) {
                    throughputFactor = 0;
                } else {
                    float3 H = ImportanceSampleGGX(float2(next_float(rng), next_float(rng)), worldNormal, roughness);
                    rayDir = reflect(-V, H);
                    float VdotH = max(dot(V, H), 0.0);
                    float NdotV = max(dot(worldNormal, V), 0.0001);
                    float NdotH = max(dot(worldNormal, H), 0.0001);
                    float G = GeometrySmith(worldNormal, V, rayDir, roughness);
                    float D = DistributionGGX(worldNormal, H, roughness);
                    float3 F_spec = FresnelSchlick(VdotH, F0);
                    throughputFactor = (F_spec * G * VdotH) / (NdotV * NdotH * probSpecular);
                    samplePDF = (D * NdotH) / (4.0f * VdotH + 0.0001f) * probSpecular;
                }
            } else {
                float3 nextDirLocal = sample_cosine_weighted(float2(next_float(rng), next_float(rng)));
                rayDir = align_to_normal(nextDirLocal, worldNormal);
                float3 H_diff = normalize(V + rayDir);
                float3 F_at_surface = FresnelSchlick(max(dot(V, H_diff), 0.0), F0);
                float3 kD = (1.0 - F_at_surface) * (1.0 - metallic);
                throughputFactor = (kD * baseColor.rgb) / (1.0 - probSpecular);
                samplePDF = (max(dot(worldNormal, rayDir), 0.0f) / 3.14159265f) * (1.0 - probSpecular);
                isPathDiffuse = true;
            }

            if (bounce == 0) {
                firstBounceThroughput = throughputFactor;
                firstBouncePDF = samplePDF;
            }

            throughput *= throughputFactor;
            rayPos = hitPos + worldNormal * 0.001f;
            if (bounce > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (next_float(rng) > p) break;
                throughput /= p;
            }
        } else {
            float3 skyRadiance = float3(0.5f, 0.7f, 1.0f) * 0.2f;
            if (bounce == 0) accumulatedColor += skyRadiance;
            else {
                if (bounce == 1) {
                    indirectHitPos = ray.Origin + ray.Direction * 1000.0f;
                    indirectHitNormal = -ray.Direction;
                    hasIndirectHit = true;
                }
                indirectRadianceAccum += skyRadiance * throughput;
            }
            break;
        }
    }

    if (hasPrimaryHit) {
        res.primaryPos = primaryHitPos; res.primaryNormal = primaryHitNormal;
        if (hasIndirectHit) {
            float3 incomingLight = indirectRadianceAccum / max(0.0001f, firstBounceThroughput);
            float targetPDF = Luminance(incomingLight);
            float weight = targetPDF / max(0.00001f, firstBouncePDF);
            updateReservoir(res, indirectHitPos, indirectHitNormal, incomingLight, targetPDF, weight, next_float(rng));
        }

        // Temporal
        if (g_Frame.frameIndex > 1) {
            float4 clipPos = mul(float4(primaryHitPos, 1.0f), g_Frame.viewProjPrevious);
            float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f; prevUV.y = 1.0f - prevUV.y;
            if (prevUV.x >= 0 && prevUV.x <= 1 && prevUV.y >= 0 && prevUV.y <= 1) {
                uint2 prevIndex = (uint2)(prevUV * (float2)launchDims);
                Reservoir prevRes = g_ReservoirPrevious[prevIndex.y * launchDims.x + prevIndex.x];
                if (prevRes.M > 0 && prevRes.targetPDF > 0) {
                    mergeReservoirs(res, prevRes, prevRes.targetPDF, prevRes.targetPDF * prevRes.W * prevRes.M, next_float(rng));
                    if (res.M > 10.0f) { res.w_sum *= 10.0f / res.M; res.M = 10.0f; }
                }
            }
        }

        if (res.targetPDF > 0) res.W = res.w_sum / max(1.f, res.M * res.targetPDF); else res.W = 0;

        if (res.W > 0) {
            float3 L_res = normalize(res.hitPos - primaryHitPos);
            float ndotl_res = max(0.0f, dot(primaryHitNormal, L_res));
            if (ndotl_res > 0) {
                float3 diffuse, specular;
                EvaluateBSDF(primaryHitNormal, primaryV, L_res, primaryBaseColor, primaryMetallic, primaryRoughness, diffuse, specular);
                if (!g_Frame.enableIndirectSpecular || (g_Frame.enableAvoidCaustics && primaryMetallic < 0.5f)) specular = 0;
                accumulatedColor += (diffuse + specular) * res.radiance * res.W * ndotl_res;
            }
        }
    }

    g_ReservoirCurrent[launchIndex.y * launchDims.x + launchIndex.x] = res;

    if (g_Frame.frameIndex <= 1) {
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = g_AccumulationBuffer[launchIndex].rgb;
        float n = (float)g_Frame.frameIndex;
        float lerpFactor = (n - 1.0f) / min(n, 2000.0f);
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    float3 exposedColor = accumulatedColor * g_Frame.exposure;
    g_Output[launchIndex] = float4(exposedColor / (exposedColor + 1.0f), 1.0f);
}
