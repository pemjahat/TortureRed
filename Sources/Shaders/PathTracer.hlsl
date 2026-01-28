
#include "Common.hlsl"
#include "PBR.hlsl"

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

float Luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// Random number generator (PCG)
struct RNG {
    uint state;
    uint inc;
};

uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float next_float(inout RNG rng) {
    rng.state = rng.state * 747796405u + 1u;
    uint res = pcg_hash(rng.state);
    return float(res) / 4294967296.0f;
}

float3 sample_cosine_weighted(float2 u) {
    float phi = 2.0f * 3.14159265f * u.x;
    float sinTheta = sqrt(u.y);
    float cosTheta = sqrt(1.0f - u.y);
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

void seed_rng(out RNG rng, uint2 screenPos, uint frameIndex) {
    rng.state = pcg_hash(screenPos.y * 65536 + screenPos.x + pcg_hash(frameIndex));
    rng.inc = 1;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    // Camera ray generation with jitter for anti-aliasing
    float2 subpixel = float2(next_float(rng), next_float(rng));
    float2 d = (((float2)launchIndex + subpixel) / (float2)launchDims) * 2.0f - 1.0f;
    // Screen space to NDC
    d.y = -d.y;

    // NDC to world space
    float4 target = mul(float4(d, 1.0f, 1.0f), g_Frame.projectionInverse);
    float3 rayDir = mul(float4(normalize(target.xyz / target.w), 0.0f), g_Frame.viewInverse).xyz;
    float3 rayPos = g_Frame.cameraPosition.xyz;

    Reservoir res;
    res.hitPos = 0;
    res.hitNormal = 0;
    res.radiance = 0;
    res.w_sum = 0;
    res.M = 0;
    res.W = 0;
    res.primaryPos = 0;
    res.primaryNormal = 0;

    float3 accumulatedColor = 0;    // store total light energy (radiance) reaches camera along path
    float3 throughput = 1;  // represent percentage light "survives" after bounce surface along path
    float3 incomingThroughput = 1; // throughput starting from the first indirect hit
    float3 indirectRadianceAccum = 0;

    // ReSTIR Primary Hit Data
    float3 primaryHitPos = 0;
    float3 primaryHitNormal = 0;
    float3 primaryV = 0;
    float3 primaryBaseColor = 0;
    float primaryMetallic = 0;
    float primaryRoughness = 1.0f;
    bool hasPrimaryHit = false; //  if first bounce hit sky (no need reservoir update)

    // ReSTIR Indirect Sample Data (Virtual Point Light)
    float3 indirectHitPos = 0;
    float3 indirectHitNormal = 0;
    bool hasIndirectHit = false;

    for (int bounce = 0; bounce < 4; bounce++) {
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

            // Fetch vertices and interpolate
            uint i0 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 0];
            uint i1 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 1];
            uint i2 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 2];

            GLTFVertex v0 = g_GlobalVertices[nodeData.vertexOffset + i0];
            GLTFVertex v1 = g_GlobalVertices[nodeData.vertexOffset + i1];
            GLTFVertex v2 = g_GlobalVertices[nodeData.vertexOffset + i2];

            // Reconstruct attributes
            float3 worldNormal = normalize(mul(v0.normal * (1.0f - barys.x - barys.y) + v1.normal * barys.x + v2.normal * barys.y, (float3x3)nodeData.world));
            float2 uv = v0.texCoord * (1.0f - barys.x - barys.y) + v1.texCoord * barys.x + v2.texCoord * barys.y;
            float3 hitPos = ray.Origin + ray.Direction * q.CommittedRayT();

            float4 baseColor = mat.baseColorFactor;
            if (mat.baseColorTextureIndex >= 0) {
                baseColor *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, uv, 0);
            }

            float metallic = mat.metallicFactor;
            float roughness = mat.roughnessFactor;

            if (mat.metallicRoughnessTextureIndex >= 0) {
                float4 mrSample = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, uv, 0);
                roughness *= mrSample.g;
                metallic *= mrSample.b;
            }
            roughness = max(0.01f, roughness);

            float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
            float3 V = -ray.Direction;

            if (bounce == 0) {
                primaryHitPos = hitPos;
                primaryHitNormal = worldNormal;
                primaryV = V;
                primaryBaseColor = baseColor.rgb;
                primaryMetallic = metallic;
                primaryRoughness = roughness;
                hasPrimaryHit = true;
            } else if (bounce == 1) {
                indirectHitPos = hitPos;
                indirectHitNormal = worldNormal;
                hasIndirectHit = true;
            }

            // NEE for Directional Light
            float3 directLight = 0;
            {
                float3 L = -normalize(g_Light.direction.xyz);
                float3 H = normalize(V + L);
                float ndotl = max(0.0001f, dot(worldNormal, L));
                float ndotv = max(0.0001f, dot(worldNormal, V));

                if (ndotl > 0) {
                    RayDesc shadowRay;
                    shadowRay.Origin = hitPos + worldNormal * 0.001f;
                    shadowRay.Direction = L;
                    shadowRay.TMin = 0.001f;
                    shadowRay.TMax = 10000.0f;

                    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
                    sq.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, shadowRay);
                    sq.Proceed();

                    if (sq.CommittedStatus() == COMMITTED_NOTHING) {
                        float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
                        float D = DistributionGGX(worldNormal, H, roughness);
                        float G = GeometrySmith(worldNormal, V, L, roughness);
                        
                        float3 numerator = D * G * F;
                        float denominator = 4.0 * ndotv * ndotl + 0.0001;
                        float3 specular = numerator / denominator;
                        
                        float3 kD = (1.0 - F) * (1.0 - metallic);
                        float3 diffuse = kD * baseColor.rgb / 3.14159265;
                        
                        directLight = (diffuse + specular) * g_Light.color.rgb * g_Light.intensity * ndotl;
                    }
                }
            }

            if (bounce == 0) {
                accumulatedColor += directLight;
            } else {
                indirectRadianceAccum += directLight * incomingThroughput;
            }

            // Path continuation (stochastic selection between diffuse and specular)
            float3 F_prob = FresnelSchlick(max(dot(worldNormal, V), 0.0), F0);
            float probSpecular = clamp(max(F_prob.r, max(F_prob.g, F_prob.b)), 0.1, 0.9);
            float rnd = next_float(rng);

            float3 throughputFactor = 1.0f;
            if (rnd < probSpecular) {
                float3 H = ImportanceSampleGGX(float2(next_float(rng), next_float(rng)), worldNormal, roughness);
                rayDir = reflect(-V, H);
                float VdotH = max(dot(V, H), 0.0);
                float NdotV = max(dot(worldNormal, V), 0.0001);
                float NdotH = max(dot(worldNormal, H), 0.0001);
                float G = GeometrySmith(worldNormal, V, rayDir, roughness);
                float3 F_spec = FresnelSchlick(VdotH, F0);
                throughputFactor = (F_spec * G * VdotH) / (NdotV * NdotH * probSpecular);
            } else {
                float3 nextDirLocal = sample_cosine_weighted(float2(next_float(rng), next_float(rng)));
                rayDir = align_to_normal(nextDirLocal, worldNormal);
                float3 F_at_surface = FresnelSchlick(max(dot(worldNormal, V), 0.0), F0);
                float3 kD = (1.0 - F_at_surface) * (1.0 - metallic);
                throughputFactor = (kD * baseColor.rgb) / (1.0 - probSpecular);
            }

            throughput *= throughputFactor;
            if (bounce > 0) incomingThroughput *= throughputFactor;

            rayPos = hitPos + worldNormal * 0.001f;

            if (bounce > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (next_float(rng) > p) break;
                throughput /= p;
                incomingThroughput /= p;
            }
        } else {
            float3 skyRadiance = float3(0.5f, 0.7f, 1.0f) * 0.2f;
            if (bounce == 0) {
                accumulatedColor += skyRadiance;
            } else {
                if (bounce == 1) {
                    indirectHitPos = ray.Origin + ray.Direction * 1000.0f;
                    indirectHitNormal = -ray.Direction;
                    hasIndirectHit = true;
                }
                indirectRadianceAccum += skyRadiance * incomingThroughput;
            }
            break;
        }
    }

    // Initial reservoir update after multi-bounce path
    if (hasPrimaryHit) {
        res.primaryPos = primaryHitPos;
        res.primaryNormal = primaryHitNormal;
        
        if (hasIndirectHit) {
            float targetPDF = Luminance(indirectRadianceAccum);
            updateReservoir(res, indirectHitPos, indirectHitNormal, indirectRadianceAccum, targetPDF, next_float(rng));
        }
    }

    // Temporal Resampling
    if (g_Frame.enableTemporal && hasPrimaryHit && g_Frame.frameIndex > 1) {
        float4 clipPos = mul(float4(primaryHitPos, 1.0f), g_Frame.viewProjPrevious);
        float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f;
        prevUV.y = 1.0f - prevUV.y;
        
        if (prevUV.x >= 0 && prevUV.x <= 1 && prevUV.y >= 0 && prevUV.y <= 1) {
            uint2 prevIndex = (uint2)(prevUV * (float2)launchDims);
            uint prevPixelIdx = prevIndex.y * launchDims.x + prevIndex.x;
            Reservoir prevRes = g_ReservoirPrevious[prevPixelIdx];
            
            // Re-validate and merge
            float dotNormal = dot(res.primaryNormal, prevRes.primaryNormal);
            float distPos = distance(res.primaryPos, prevRes.primaryPos);

            float targetPDF = Luminance(prevRes.radiance);
            if (prevRes.M > 0 && targetPDF > 0 && dotNormal > 0.95f && distPos < 0.5f) {
                float weight = targetPDF * prevRes.W * prevRes.M;
                mergeReservoirs(res, prevRes, weight, next_float(rng));
                if (res.M > 60.0f) { // Slightly higher history for temporal
                    res.w_sum *= 60.0f / res.M;
                    res.M = 60.0f;
                }
            }
        }
    }

    // Spatial Resampling
    if (g_Frame.enableSpatial && hasPrimaryHit && g_Frame.frameIndex > 1) {
        for (int i = 0; i < 2; i++) {
            float2 offset = float2(next_float(rng), next_float(rng)) * 2.0f - 1.0f;
            int2 neighborIndex = (int2)launchIndex + (int2)(offset * 15.0f); // increased range slightly
            
            if (neighborIndex.x >= 0 && neighborIndex.x < (int)launchDims.x && 
                neighborIndex.y >= 0 && neighborIndex.y < (int)launchDims.y) {
                
                uint neighborPixelIdx = neighborIndex.y * launchDims.x + neighborIndex.x;
                Reservoir neighborRes = g_ReservoirCurrent[neighborPixelIdx];
                
                // Similarity check: Normal and Position
                float dotNormal = dot(res.primaryNormal, neighborRes.primaryNormal);
                float distPos = distance(res.primaryPos, neighborRes.primaryPos);
                float targetPDF = Luminance(neighborRes.radiance);

                // Only merge if the neighbor is on a similar surface
                if (neighborRes.M > 0 && targetPDF > 0 && dotNormal > 0.95f && distPos < 0.5f) {
                    
                    float weight = targetPDF * neighborRes.W * neighborRes.M;
                    mergeReservoirs(res, neighborRes, weight, next_float(rng));
                }
            }
        }
    }

    // Final reservoir weight normalization
    float targetPDF = Luminance(res.radiance);
    if (targetPDF > 0) {
        // Weight ~ how much does this sample represent average of all samples
        res.W = res.w_sum / (res.M * targetPDF);
    } else {
        res.W = 0;
    }

    // Apply Indirect contribution from Reservoir
    if (hasPrimaryHit && res.W > 0) {
        float3 L = normalize(res.hitPos - primaryHitPos);
        float3 V = primaryV;
        float3 N = primaryHitNormal;
        
        float ndotl = max(0.0f, dot(N, L));
        float ndotv = max(0.0001f, dot(N, V));

        if (ndotl > 0) {
            float3 H = normalize(V + L);
            float3 F0 = lerp(float3(0.04, 0.04, 0.04), primaryBaseColor, primaryMetallic);
            
            float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
            float D = DistributionGGX(N, H, primaryRoughness);
            float G = GeometrySmith(N, V, L, primaryRoughness);
            
            float3 specular = (D * G * F) / (4.0 * ndotv * ndotl + 0.0001);
            float3 kD = (1.0 - F) * (1.0 - primaryMetallic);
            float3 diffuse = kD * primaryBaseColor / 3.14159265;
            
            // Weight accounts for the probability of selecting this specific light path
            accumulatedColor += (diffuse + specular) * res.radiance * res.W * ndotl;
        }
    }

    // Store reservoir for next frame
    uint pixelIndex = launchIndex.y * launchDims.x + launchIndex.x;
    g_ReservoirCurrent[pixelIndex] = res;

    if (g_Frame.frameIndex <= 1) {
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
        
        float3 exposedColor = accumulatedColor * g_Frame.exposure;
        float3 ldrColor = exposedColor / (exposedColor + 1.0f);
        g_Output[launchIndex] = float4(ldrColor, 1.0f);
    } else {
        float3 prevColor = g_AccumulationBuffer[launchIndex].rgb;
        
        // Progressive accumulation logic similar to DXRPathTracer
        float n = (float)g_Frame.frameIndex;
        // Limit n to prevent floating point precision issues at very high frame counts
        float lerpFactor = (n - 1.0f) / n;
        if (n > 2000.0f) lerpFactor = 1999.0f / 2000.0f;

        float3 finalColor = lerp(accumulatedColor, prevColor, lerpFactor);
        
        g_AccumulationBuffer[launchIndex] = float4(finalColor, 1.0f);
        
        // Basic Tone Mapping for output
        float3 exposedColor = finalColor * g_Frame.exposure;
        float3 ldrColor = exposedColor / (exposedColor + 1.0f);
        
        g_Output[launchIndex] = float4(ldrColor, 1.0f);
    }
}
