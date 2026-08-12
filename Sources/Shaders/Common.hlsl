#ifndef COMMON_HLSL
#define COMMON_HLSL

#include "Shared/SharedTypes.h"

// FP16Scale/FP16Max are defined in Shared/SharedTypes.h (shared CPU/GPU section).
// FP16Scale = 2^-10 pre-scales values so they fit in fp16 representable range.
// FP16Max   = 65504.0f is the maximum representable fp16 value.

#define SHARC_HASH_ENTRIES_NUM (4 * 1024 * 1024)

float3 ReconstructWorldPos(float2 uv, float depth, float4x4 projectionInverse, float4x4 viewInverse) {
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, projectionInverse);
    viewPos /= viewPos.w;
    float4 worldPos = mul(viewPos, viewInverse);
    return worldPos.xyz;
}

struct RayPayload {
    float4 color;
};

struct GLTFVertex {
    float3 position;
    float3 normal;
    float2 texCoord;
};

// Weighted Reservoir Sampling helper
// Returns true if the new sample was selected
bool updateReservoir(inout Reservoir r, float3 hitPos, float3 hitNormal, float3 radiance, float firstBounceHitT, float risWeight, float rnd) {
    r.w_sum += risWeight;
    r.M += 1.0f;

    if (rnd * r.w_sum <= risWeight) {
        r.hitPos = hitPos;
        r.hitNormal = hitNormal;
        r.radiance = radiance;
        r.firstBounceHitT = firstBounceHitT;
        return true;
    }
    return false;
}

bool updateReservoir(inout Reservoir r, float3 hitPos, float3 hitNormal, float3 radiance, float risWeight, float rnd) {
    return updateReservoir(r, hitPos, hitNormal, radiance, 0.0f, risWeight, rnd);
}

// Merge two reservoirs with a shifted target PDF
// Returns true if the reservoir was updated with the new sample
bool mergeReservoirsWithWeight(inout Reservoir curRes, Reservoir neighbourRes, float risWeight, float rnd) {
    curRes.w_sum += risWeight;
    curRes.M += neighbourRes.M;

    if (rnd * curRes.w_sum <= risWeight) {
        curRes.hitPos = neighbourRes.hitPos;
        curRes.hitNormal = neighbourRes.hitNormal;
        curRes.radiance = neighbourRes.radiance;
        curRes.firstBounceHitT = neighbourRes.firstBounceHitT;
        curRes.historyAge = neighbourRes.historyAge;
        return true;
    }
    return false;
}

bool mergeReservoirs(inout Reservoir curRes, Reservoir neighbourRes, float shiftedTargetPDF, float rnd) {
    float risWeight = shiftedTargetPDF * neighbourRes.W * neighbourRes.M;
    return mergeReservoirsWithWeight(curRes, neighbourRes, risWeight, rnd);
}

void capReservoirHistory(inout Reservoir r, float maxHistoryLength) {
    if (r.M > maxHistoryLength) {
        float scale = maxHistoryLength / r.M;
        r.w_sum *= scale;
        r.M = maxHistoryLength;
    }
}

#endif // COMMON_HLSL
