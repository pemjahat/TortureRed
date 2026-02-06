#ifndef RTXDI_BRIDGE_HLSLI
#define RTXDI_BRIDGE_HLSLI

#include "CommonTracing.hlsl"
#include "Rtxdi/Utils/Math.hlsli"

// Bridge that need to be implemented
/*
// Shared with Restir DI
RAB_EmptySurface()
RAB_GetGBufferSurface(...)
RAB_GetNextRandom(...)
RAB_GetSurfaceLinearDepth(...)
RAB_GetSurfaceNormal(...)
RAB_GetSurfaceWorldPos(...)
RAB_IsSurfaceValid(...)
RAB_RandomSamplerState { ... }
RAB_Surface { ... }

// Exclusive for RestirGI
RAB_GetGISampleTargetPdfForSurface(...)
RAB_ValidateGISampleWithJacobian(...)
RAB_GetConservativeVisibility(surface, samplePosition)
RAB_GetTemporalConservativeVisibility(currentSurface, previousSurface, samplePosition)
*/

struct RAB_Surface {
    float3 worldPos;
    float3 normal;
    float linearDepth;
    float3 albedo;
    float roughness;
    float metallic;
    float3 viewDir;
    bool valid;
};

struct RAB_Material {
    float3 albedo;
    float roughness;
    float metallic;
};

#define RAB_RandomSamplerState RNG

RAB_RandomSamplerState RAB_InitRandomSampler(uint2 pixelPos, uint frameIndex) {
    RNG rng;
    seed_rng(rng, pixelPos, frameIndex);
    return rng;
}

float RAB_GetNextRandom(inout RAB_RandomSamplerState rng) {
    return next_float(rng);
}

RAB_Surface RAB_EmptySurface() {
    RAB_Surface s;
    s.worldPos = 0; s.normal = 0; s.linearDepth = 1.0f; s.albedo = 0;
    s.roughness = 0; s.metallic = 0; s.viewDir = 0; s.valid = false;
    return s;
}

bool RAB_IsSurfaceValid(RAB_Surface surface) {
    return surface.valid;
}

float3 RAB_GetSurfaceWorldPos(RAB_Surface surface) { return surface.worldPos; }
float3 RAB_GetSurfaceNormal(RAB_Surface surface) { return surface.normal; }
float RAB_GetSurfaceLinearDepth(RAB_Surface surface) { return surface.linearDepth; }

RAB_Material RAB_GetMaterial(RAB_Surface s) {
    RAB_Material m;
    m.albedo = s.albedo;
    m.roughness = s.roughness;
    m.metallic = s.metallic;
    return m;
}

bool RAB_AreMaterialsSimilar(RAB_Material m1, RAB_Material m2) {
    return all(abs(m1.albedo - m2.albedo) < 0.1f) && (abs(m1.roughness - m2.roughness) < 0.1f);
}

bool RAB_ValidateGISampleWithJacobian(float jacobian) {
    return jacobian > 0.0f && jacobian < 64.0f;
}

// Assuming these are globally available where bridge is included
// Texture2D g_Textures[] : register(t0, space0);
// ConstantBuffer<FrameConstants> g_Frame : register(b0);
// SamplerState g_LinearSampler : register(s0);

RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame) {
    uint2 launchDims;
    g_Textures[g_Frame.depthIndex].GetDimensions(launchDims.x, launchDims.y);
    
    if (any(pixelPosition < 0) || any(pixelPosition >= (int2)launchDims)) {
        return RAB_EmptySurface();
    }

    float2 uv = ((float2)pixelPosition + 0.5f) / (float2)launchDims;
    float depth = g_Textures[g_Frame.depthIndex].SampleLevel(g_LinearSampler, uv, 0).r;
    
    if (depth >= 1.0f) return RAB_EmptySurface();

    RAB_Surface s;
    s.valid = true;
    s.linearDepth = depth;
    
    if (previousFrame) {
        s.worldPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInversePrevious);
        s.normal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);
        s.viewDir = normalize(g_Frame.prevCameraPosition.xyz - s.worldPos);
    } else {
        s.worldPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverse, g_Frame.viewInverse);
        s.normal = normalize(g_Textures[g_Frame.normalIndex].SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0f - 1.0f);
        s.viewDir = normalize(g_Frame.cameraPosition.xyz - s.worldPos);
    }
    
    s.albedo = g_Textures[g_Frame.albedoIndex].SampleLevel(g_LinearSampler, uv, 0).rgb;
    float4 mat = g_Textures[g_Frame.materialIndex].SampleLevel(g_LinearSampler, uv, 0);
    s.roughness = max(0.01f, mat.r);
    s.metallic = mat.g;
    
    return s;
}

float RAB_GetGISampleTargetPdfForSurface(float3 samplePos, float3 sampleRadiance, RAB_Surface surface) {
    if (!surface.valid) return 0;
    
    float3 L = normalize(samplePos - surface.worldPos);
    float3 N = surface.normal;
    float3 V = surface.viewDir;
    
    float dotNL = dot(N, L);
    if (dotNL <= 0) return 0;

    float3 diffuseBRDF, specularBRDF;
    // EvaluateBSDF from PBR.hlsl (included via CommonTracing.hlsl)
    EvaluateBSDF(N, V, L, surface.albedo, surface.metallic, surface.roughness, diffuseBRDF, specularBRDF);
    
    // Target PDF is Luminance of reflected radiance.
    // We multiply by dotNL and PI is already handled inside EvaluateBSDF for diffuse.
    float3 reflected = (diffuseBRDF + specularBRDF) * sampleRadiance * dotNL;
    return max(0.0f, Luminance(reflected));
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition) {
    return true; // Simplified for now, ReSTIR GI handles visibility during resampling or resolve
}

bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, float3 samplePosition) {
    return true;
}

int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool previousFrame) {
    uint2 dims;
    g_Textures[g_Frame.depthIndex].GetDimensions(dims.x, dims.y);
    return clamp(pixelPosition, int2(0, 0), int2(dims) - 1);
}

// RTXDI GI functions might need more bridge functions depending on version
// For now, these should cover the basics.

#endif // RTXDI_BRIDGE_HLSLI
