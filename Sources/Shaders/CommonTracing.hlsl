#ifndef COMMON_TRACING_HLSL
#define COMMON_TRACING_HLSL

#include "Common.hlsl"
#include "PBR.hlsl"

// Global Raytracing Resources (Space 1)
Texture2D g_Textures[] : register(t0, space0);
RaytracingAccelerationStructure g_Scene : register(t2, space1);
StructuredBuffer<DrawNodeData> g_DrawNodeBuffer : register(t1, space1);
StructuredBuffer<MaterialConstants> g_Materials : register(t0, space1);
StructuredBuffer<GLTFVertex> g_GlobalVertices : register(t4, space1);
StructuredBuffer<uint> g_GlobalIndices : register(t3, space1);
ByteAddressBuffer g_LightLUT : register(t1, space2);

SamplerState g_LinearSampler : register(s0);

#define PROCESS_ALPHA_MASK(q, rng) \
    if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) { \
        uint instanceIdx = q.CandidateInstanceID(); \
        uint triIdx = q.CandidatePrimitiveIndex(); \
        float2 barys = q.CandidateTriangleBarycentrics(); \
        DrawNodeData nodeData = g_DrawNodeBuffer[instanceIdx]; \
        MaterialConstants mat = g_Materials[nodeData.materialID]; \
        if (mat.alphaMode > 0) { \
            uint i0 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 0]; \
            uint i1 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 1]; \
            uint i2 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 2]; \
            GLTFVertex v0 = g_GlobalVertices[nodeData.vertexOffset + i0]; \
            GLTFVertex v1 = g_GlobalVertices[nodeData.vertexOffset + i1]; \
            GLTFVertex v2 = g_GlobalVertices[nodeData.vertexOffset + i2]; \
            float2 hitUv = v0.texCoord * (1.0f - barys.x - barys.y) + v1.texCoord * barys.x + v2.texCoord * barys.y; \
            float alpha = mat.baseColorFactor.a; \
            if (mat.baseColorTextureIndex >= 0) { \
                alpha *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0).a; \
            } \
            alpha = (mat.alphaMode == 1) ? ((alpha >= mat.alphaCutoff) ? 1.0f : 0.0f) : saturate(alpha); \
            if (next_float(rng) < alpha) { \
                q.CommitNonOpaqueTriangleHit(); \
            } \
        } else { \
            q.CommitNonOpaqueTriangleHit(); \
        } \
    }

float Luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

struct Surface {
    float3 worldPos;
    float3 normal;
    float3 viewDir;
    float3 albedo;
    float metallic;
    float roughness;
};

float GetTargetPDF(Surface s, float3 samplePos, float3 sampleRadiance) {
    float3 L = normalize(samplePos - s.worldPos);
    float dotNL = max(0.0f, dot(s.normal, L));
    if (dotNL <= 0) return 0;
    
    float3 d, spec;
    EvaluateBSDF(s.normal, s.viewDir, L, s.albedo, s.metallic, s.roughness, d, spec);
    float3 reflected = (d + spec) * sampleRadiance * dotNL;
    return max(0.0f, Luminance(reflected));
}

// Compute the Jacobian for a GI shift (point-to-point solid angle ratio)
float ComputeJacobian(float3 primaryPos, float3 neighborPrimaryPos, float3 sampleHitPos, float3 sampleHitNormal) {
    float3 diffP = sampleHitPos - primaryPos;
    float distSqP = max(0.0001f, dot(diffP, diffP));
    float cosP = max(0.0001f, abs(dot(sampleHitNormal, diffP / sqrt(distSqP))));
    
    float3 diffQ = sampleHitPos - neighborPrimaryPos;
    float distSqQ = max(0.0001f, dot(diffQ, diffQ));
    float cosQ = max(0.0001f, abs(dot(sampleHitNormal, diffQ / sqrt(distSqQ))));
    
    // Solid angle at P / Solid angle at Q
    return (cosP * distSqQ) / (max(0.00001f, cosQ * distSqP));    
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

// Sample an indirect ray based on surface properties
// Throughput is cumulative weight ~ (BSDF * cos) / PDF
void SampleIndirectRay(float3 N, float3 V, float3 baseColor, float metallic, float roughness, inout RNG rng, out float3 rayDir, out float3 throughput, out float pdf, out bool isDiffuse, bool enableSpecular = true) {
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float3 F_prob = FresnelSchlick(max(dot(N, V), 0.0), F0);
    float probSpecular = clamp(max(F_prob.r, max(F_prob.g, F_prob.b)), 0.1, 0.9);
    float rnd = next_float(rng);

    if (rnd < probSpecular) {
        if (!enableSpecular) {
            throughput = 0; rayDir = 0; pdf = 1; isDiffuse = false;
            return;
        }
        float3 H = ImportanceSampleGGX(float2(next_float(rng), next_float(rng)), N, roughness);
        rayDir = reflect(-V, H);
        float VdotH = max(dot(V, H), 0.0);
        float NdotV = max(dot(N, V), 0.0001);
        float NdotH = max(dot(N, H), 0.0001);
        float G = GeometrySmith(N, V, rayDir, roughness);
        float D = DistributionGGX(N, H, roughness);
        float3 F_spec = FresnelSchlick(VdotH, F0);
        throughput = (F_spec * G * VdotH) / (NdotV * NdotH * (probSpecular + 0.0001f));
        pdf = (D * NdotH) / (4.0f * VdotH + 0.0001f) * probSpecular;
        isDiffuse = false;
    } else {
        float3 nextDirLocal = sample_cosine_weighted(float2(next_float(rng), next_float(rng)));
        rayDir = align_to_normal(nextDirLocal, N);
        float3 H = normalize(V + rayDir);
        float3 F_at_surface = FresnelSchlick(max(dot(V, H), 0.0), F0);
        float3 kD = (1.0 - F_at_surface) * (1.0 - metallic);
        throughput = (kD * baseColor) / (1.0 - probSpecular + 0.0001f);
        pdf = (max(dot(N, rayDir), 0.0f) / 3.14159265f) * (1.0 - probSpecular);
        isDiffuse = true;
    }
}

float3 GetDirectLighting(float3 P, float3 N, float3 V, float3 albedo, float metallic, float roughness, 
                        RaytracingAccelerationStructure scene, LightConstants light, FrameConstants frame,
                        inout RNG rng, bool isDiffuse = false) {
    float3 L_light = -normalize(light.direction.xyz);
    float ndotl = max(0.0001f, dot(N, L_light));
    if (ndotl > 0) {
        RayDesc shadowRay;
        shadowRay.Origin = P + N * 0.001f;
        shadowRay.Direction = L_light;
        shadowRay.TMin = 0.001f; shadowRay.TMax = 10000.0f;
        RayQuery<RAY_FLAG_NONE> sq;
        sq.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
        while (sq.Proceed()) {
            PROCESS_ALPHA_MASK(sq, rng);
        }

        if (sq.CommittedStatus() == COMMITTED_NOTHING) {
            float3 d, s;
            EvaluateBSDF(N, V, L_light, albedo, metallic, roughness, d, s);
            if (!frame.enableIndirectSpecular || (isDiffuse && frame.enableAvoidCaustics)) s = 0;
            return (d + s) * light.color.rgb * light.intensity * ndotl;
        }
    }
    return 0;
}

// Evaluate lighting from a single local light (point/spot)
float3 EvaluateLocalLight(float3 P, float3 N, float3 V, float3 albedo, float metallic, float roughness,
                         LightConstants light, RaytracingAccelerationStructure scene, FrameConstants frame,
                         inout RNG rng, bool isDiffuse) {
    float3 d = light.position.xyz - P;
    float dist = length(d);
    float3 L_light = normalize(d);
    
    float ndotl = max(0.0001f, dot(N, L_light));
    if (ndotl <= 0.0) return 0.0;
    
    // Attenuation: 1.0 / (1.0 + 0.1*dist + 0.01*dist*dist)
    float attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
    
    // Spot light cone attenuation
    // light.direction.w stores cosOuter, light.padding[0] stores cosInner
    float cosAngle = dot(-L_light, normalize(light.direction.xyz));
    float cosOuter = light.direction.w;
    float cosInner = asfloat(light.padding[0]);
    float spotEffect = smoothstep(cosOuter, cosInner, cosAngle);
    
    // Shadow ray with TMax = distance - epsilon
    RayDesc shadowRay;
    shadowRay.Origin = P + N * 0.001f;
    shadowRay.Direction = L_light;
    shadowRay.TMin = 0.001f;
    shadowRay.TMax = dist - 0.002f;
    
    RayQuery<RAY_FLAG_NONE> sq;
    sq.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
    while (sq.Proceed()) {
        PROCESS_ALPHA_MASK(sq, rng);
    }
    
    if (sq.CommittedStatus() != COMMITTED_NOTHING) return 0.0;
    
    float3 diff, spec;
    EvaluateBSDF(N, V, L_light, albedo, metallic, roughness, diff, spec);
    if (!frame.enableIndirectSpecular || (isDiffuse && frame.enableAvoidCaustics)) spec = 0;
    
    return (diff + spec) * light.color.rgb * light.intensity * ndotl * attenuation * spotEffect;
}

// Multi-light direct lighting function
// Iterates through all lights and accumulates their contributions
float3 GetDirectLightingMultiLights(float3 P, float3 N, float3 V, float3 albedo, float metallic, float roughness,
                                   RaytracingAccelerationStructure scene, 
                                   StructuredBuffer<LightConstants> lights, uint numLights,
                                   FrameConstants frame, inout RNG rng, bool isDiffuse = false) {
    float3 totalLighting = 0.0;
    
    for (uint i = 0; i < numLights; ++i) {
        LightConstants light = lights[i];
        
        // Determine light type: position.w == 0 for directional, > 0.5 for point/spot
        if (light.direction.w < 0.5f) {
            // Directional light - use existing single light logic
            // But we need to handle it carefully since GetDirectLighting expects a single light
            float3 L_light = -normalize(light.direction.xyz);
            float ndotl = max(0.0001f, dot(N, L_light));
            if (ndotl > 0) {
                RayDesc shadowRay;
                shadowRay.Origin = P + N * 0.001f;
                shadowRay.Direction = L_light;
                shadowRay.TMin = 0.001f; 
                shadowRay.TMax = 10000.0f;
                RayQuery<RAY_FLAG_NONE> sq;
                sq.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
                while (sq.Proceed()) {
                    PROCESS_ALPHA_MASK(sq, rng);
                }

                if (sq.CommittedStatus() == COMMITTED_NOTHING) {
                    float3 d, s;
                    EvaluateBSDF(N, V, L_light, albedo, metallic, roughness, d, s);
                    if (!frame.enableIndirectSpecular || (isDiffuse && frame.enableAvoidCaustics)) s = 0;
                    totalLighting += (d + s) * light.color.rgb * light.intensity * ndotl;
                }
            }
        } else {
            // Point/spot light
            totalLighting += EvaluateLocalLight(P, N, V, albedo, metallic, roughness, light, scene, frame, rng, isDiffuse);
        }
    }
    
    return totalLighting;
}

bool CheckVisibility(float3 P, float3 N, float3 samplePos, inout RNG rng) {
    float3 L = samplePos - P;
    float dist = length(L);
    L /= dist;

    RayDesc ray;
    ray.Origin = P + N * 0.001f;
    ray.Direction = L;
    ray.TMin = 0.001f;
    ray.TMax = dist - 0.002f;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    while (q.Proceed()) {
        PROCESS_ALPHA_MASK(q, rng);
    }

    return q.CommittedStatus() == COMMITTED_NOTHING;
}

// ============================================================================
// LIGHT SAMPLING LUT - O(1) LOOKUP SAMPLING
// ============================================================================
// Uses a 1D texture LUT to map a random value U=[0,1) directly to a light index.
// LUT resolution is fixed at 256.
// Max lights: 256 (hard cap, matches LUT resolution for simplicity).
// 
// Build: On CPU, map each CDF bucket [i/256, (i+1)/256) to a light index.
// Shader: Single lookup: uint lightIdx = LUT[uint(U * 256)];
// ============================================================================

#define LIGHT_LUT_RESOLUTION 256
#define MAX_LIGHTS 256

// LightSampleResult struct for LUT sampling
struct LightSampleResult {
    uint lightIndex;
    float pdf;
    LightConstants light;
};

// Sample a single light using LUT - O(1) lookup
LightSampleResult SampleSingleLightLUT(
    float rngSample,                       // Random sample in [0,1)
    StructuredBuffer<LightConstants> lights,
    ByteAddressBuffer lightLUTBuffer,       // LUT: uint[256] light indices
    uint numLights) {
    
    LightSampleResult result;
    
    if (numLights <= 1) {
        result.lightIndex = 0;
        result.pdf = 1.0;
    } else {
        // Clamp and scale U to LUT resolution
        float u = clamp(rngSample, 0.0f, 0.999999f);
        uint lutIndex = uint(u * float(LIGHT_LUT_RESOLUTION));
        lutIndex = min(lutIndex, LIGHT_LUT_RESOLUTION - 1);
        
        // Direct LUT lookup: load uint at offset lutIndex * sizeof(uint)
        result.lightIndex = lightLUTBuffer.Load(lutIndex * sizeof(uint));
        
        // Clamp to valid range (defensive)
        result.lightIndex = min(result.lightIndex, numLights - 1);
        
        // Use the importance PDF stored in the light constants
        // Note: For numLights > 1, the LUT encodes the distribution.
        // The selection PDF is stored in the light buffer itself.
        result.pdf = max(0.00001f, lights[result.lightIndex].selectionPDF);
    }
    
    result.light = lights[result.lightIndex];
    return result;
}

// Convenience wrapper using frame constants to get LUT buffer
LightSampleResult SampleSingleLight(
    float rngSample,
    StructuredBuffer<LightConstants> lights,
    uint numLights,
    FrameConstants frame) {
    
    return SampleSingleLightLUT(rngSample, lights, g_LightLUT, numLights);
}

// Legacy CDF functions removed - LUT is O(1) and sufficient for max 256 lights

// Sample a single light uniformly - each light has equal probability 1/numLights
LightSampleResult SampleSingleLightUniform(
    float rngSample,                        // Random value in [0, 1)
    StructuredBuffer<LightConstants> lights,
    uint numLights) {
    
    LightSampleResult result;
    
    if (numLights <= 1) {
        result.lightIndex = 0;
        result.pdf = 1.0f;
    } else {
        uint idx = min(uint(rngSample * float(numLights)), numLights - 1);
        result.lightIndex = idx;
        result.pdf = 1.0f / float(numLights);
    }
    
    result.light = lights[result.lightIndex];
    return result;
}

// Evaluate lighting from a sampled light with proper weighting
// For indirect bounces: uses stochastic light sampling + MIS with BSDF
float3 EvaluateSingleLightWithMIS(
    float3 P, float3 N, float3 V,
    float3 albedo, float metallic, float roughness,
    LightSampleResult lightSample,
    RaytracingAccelerationStructure scene,
    FrameConstants frame,
    inout RNG rng,
    bool isDiffuse) {
    
    LightConstants light = lightSample.light;
    float3 result = 0.0f;
    
    // Directional lights (position.w < 0.5) are handled specially
    if (light.direction.w < 0.5f) {
        float3 L_light = -normalize(light.direction.xyz);
        float ndotl = max(0.0001f, dot(N, L_light));
        
        if (ndotl > 0) {
            RayDesc shadowRay;
            shadowRay.Origin = P + N * 0.001f;
            shadowRay.Direction = L_light;
            shadowRay.TMin = 0.001f;
            shadowRay.TMax = 10000.0f;
            
            RayQuery<RAY_FLAG_NONE> sq;
            sq.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
            while (sq.Proceed()) {
                PROCESS_ALPHA_MASK(sq, rng);
            }

            if (sq.CommittedStatus() == COMMITTED_NOTHING) {
                float3 d, s;
                EvaluateBSDF(N, V, L_light, albedo, metallic, roughness, d, s);
                if (!frame.enableIndirectSpecular || (isDiffuse && frame.enableAvoidCaustics)) s = 0;
                result = (d + s) * light.color.rgb * light.intensity * ndotl;
            }
        }
    } else {
        // Point/spot light
        result = EvaluateLocalLight(P, N, V, albedo, metallic, roughness, light, scene, frame, rng, isDiffuse);
    }
    
    // Divide by PDF to get unbiased estimate: L / p
    return result / lightSample.pdf;
}

// Convert BSDF PDF (solid angle) to light sampling PDF
float BSDFPDFToLightPDF(float bsdfPDF, float3 P, float3 sampleDir, float3 lightPos) {
    float3 toLight = lightPos - P;
    float distSq = dot(toLight, toLight);
    float dist = sqrt(distSq);
    float cosTheta = abs(dot(sampleDir, toLight / dist));
    return bsdfPDF * cosTheta / max(distSq, 0.0001f);
}

// Stochastic NEE for indirect bounces with MIS
// Branches based on frame.lightSamplingMode:
//   0 = Uniform: each light selected with equal probability (1/numLights)
//   1 = Importance (CDF): LUT-based selection weighted by intensity
float3 GetDirectLightingStochastic(
    float3 P, float3 N, float3 V,
    float3 albedo, float metallic, float roughness,
    RaytracingAccelerationStructure scene,
    StructuredBuffer<LightConstants> lights,
    uint numLights,
    FrameConstants frame,
    inout RNG rng,
    bool isDiffuse) {
    
    if (numLights == 0) return 0.0f;
    
    LightSampleResult lightSample;
    
    if (frame.lightSamplingMode == 0) {
        // Uniform sampling: pick any light with equal probability
        lightSample = SampleSingleLightUniform(next_float(rng), lights, numLights);
    } else {
        // Importance (CDF) sampling: LUT-based, weighted by intensity
        // Requires a valid LUT buffer
        if (frame.lightLUTBufferIndex == 0xFFFFFFFF) return 0.0f;
        lightSample = SampleSingleLight(next_float(rng), lights, numLights, frame);
    }
    
    // Evaluate the sampled light (result is already divided by PDF)
    return EvaluateSingleLightWithMIS(
        P, N, V, albedo, metallic, roughness,
        lightSample, scene, frame, rng, isDiffuse);
}

// ============================================================================
// RIS (Resampled Importance Sampling) for indirect bounce light selection.
// Selects 1 winning light from M candidates using an unshadowed BSDF-weighted
// target PDF. Only 1 shadow ray is fired (for the winner), giving O(1) cost.
// ============================================================================

// Unshadowed target PDF: p̂(l) = luminance( BSDF * NdotL * L_color * intensity * attenuation )
float RIS_TargetPDF(
    float3 P, float3 N, float3 V,
    float3 albedo, float metallic, float roughness,
    LightConstants light)
{
    float3 L;
    float  attenuation = 1.0f;
    float  spotEffect  = 1.0f;

    if (light.direction.w < 0.5f)
    {
        // Directional
        L = -normalize(light.direction.xyz);
    }
    else
    {
        // Point / Spot
        float3 diff = light.position.xyz - P;
        float  dist = length(diff);
        if (dist < 0.0001f) return 0.0f;
        L           = diff / dist;
        attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
        float cosAngle = dot(-L, normalize(light.direction.xyz));
        float cosOuter = light.direction.w;
        float cosInner = asfloat(light.padding[0]);
        spotEffect = smoothstep(cosOuter, cosInner, cosAngle);
    }

    float NdotL = dot(N, L);
    if (NdotL <= 0.0f) return 0.0f;

    float3 diffBRDF, specBRDF;
    EvaluateBSDF(N, V, L, albedo, metallic, roughness, diffBRDF, specBRDF);
    return max(0.0f, Luminance((diffBRDF + specBRDF) * NdotL
                               * light.color.rgb * light.intensity
                               * attenuation * spotEffect));
}

// RIS light selection: M candidates → 1 winner → 1 shadow ray.
// Returns an unbiased estimate of direct lighting summed over all local lights.
// numCandidates: tune 4 for first indirect bounce, 1 for deeper bounces.
float3 GetDirectLightingRIS(
    float3 P, float3 N, float3 V,
    float3 albedo, float metallic, float roughness,
    RaytracingAccelerationStructure scene,
    StructuredBuffer<LightConstants> lights,
    uint numLights,
    FrameConstants frame,
    inout RNG rng,
    bool isDiffuse,
    uint numCandidates = 4)
{
    if (numLights == 0) return 0.0f;

    // --- Phase 1: weighted reservoir sampling over M candidates (no shadow) ---
    uint  selectedIndex = 0;
    float weightSum     = 0.0f;

    for (uint i = 0; i < numCandidates; ++i)
    {
        float2 u = float2(next_float(rng), next_float(rng));
        LightSampleResult lsr = SampleSingleLight(u, lights, numLights, frame);

        float targetPDF = RIS_TargetPDF(P, N, V, albedo, metallic, roughness, lsr.light);
        float risWeight = targetPDF / max(lsr.pdf, 1e-6f);

        weightSum += risWeight;
        if (next_float(rng) < (risWeight / max(weightSum, 1e-6f)))
            selectedIndex = lsr.lightIndex;
    }

    if (weightSum <= 0.0f) return 0.0f;

    // --- Phase 2: evaluate winner with 1 shadow ray ---
    LightConstants winner = lights[selectedIndex];
    float winnerTarget = RIS_TargetPDF(P, N, V, albedo, metallic, roughness, winner);
    if (winnerTarget <= 0.0f) return 0.0f;

    float3 L_winner = 0.0f;
    if (winner.direction.w < 0.5f)
    {
        // Directional
        float3 L = -normalize(winner.direction.xyz);
        float NdotL = max(0.0001f, dot(N, L));
        RayDesc sr;
        sr.Origin = P + N * 0.001f;
        sr.Direction = L;
        sr.TMin = 0.001f;
        sr.TMax = 10000.0f;
        RayQuery<RAY_FLAG_NONE> sq;
        sq.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sr);
        while (sq.Proceed()) {
            PROCESS_ALPHA_MASK(sq, rng);
        }
        if (sq.CommittedStatus() == COMMITTED_NOTHING)
        {
            float3 d, s;
            EvaluateBSDF(N, V, L, albedo, metallic, roughness, d, s);
            if (!frame.enableIndirectSpecular || (isDiffuse && frame.enableAvoidCaustics)) s = 0;
            L_winner = (d + s) * winner.color.rgb * winner.intensity * NdotL;
        }
    }
    else
    {
        // Point / Spot — EvaluateLocalLight already traces the shadow ray
        L_winner = EvaluateLocalLight(P, N, V, albedo, metallic, roughness,
                                      winner, scene, frame, rng, isDiffuse);
    }

    // Unbiased RIS contribution weight: W = weightSum / (M * p̂(x*))
    float W = weightSum / (float(numCandidates) * winnerTarget);
    return L_winner * W;
}

// Direct lighting dispatch based on frame.lightSamplingMode.
// Applied uniformly to both primary and indirect hits:
//   0 = Uniform     : 1 shadow ray,  equal probability per light
//   1 = ImportancePDF: 1 shadow ray,  LUT-weighted by light intensity
//   2 = Brute Force : N shadow rays, evaluates every light
float3 GetDirectLightingHybrid(
    float3 P, float3 N, float3 V,
    float3 albedo, float metallic, float roughness,
    RaytracingAccelerationStructure scene,
    StructuredBuffer<LightConstants> lights,
    uint numLights,
    FrameConstants frame,
    bool isDiffuse,
    inout RNG rng) { // Only used for stochastic modes (mode 0 or 1)) {
    
    if (numLights > 1 && frame.lightSamplingMode != 2) {
        // Stochastic light sampling (Uniform or Importance CDF)
        return GetDirectLightingStochastic(P, N, V, albedo, metallic, roughness,
                                          scene, lights, numLights, frame,
                                          rng, isDiffuse);
    } else {
        // Brute force: evaluate every light (primary hit or mode 2)
        return GetDirectLightingMultiLights(P, N, V, albedo, metallic, roughness,
                                           scene, lights, numLights, frame, rng, isDiffuse);
    }
}

#endif // COMMON_TRACING_HLSL
