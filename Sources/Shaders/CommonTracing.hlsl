#ifndef COMMON_TRACING_HLSL
#define COMMON_TRACING_HLSL

#include "Common.hlsl"
#include "PBR.hlsl"

float Luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
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

#endif // COMMON_TRACING_HLSL
