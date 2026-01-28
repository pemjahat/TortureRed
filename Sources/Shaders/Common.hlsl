#ifndef COMMON_HLSL
#define COMMON_HLSL

// Alignment matched with C++ structures in GraphicsTypes.h

struct FrameConstants {
    row_major float4x4 viewProj;
    row_major float4x4 viewInverse;
    row_major float4x4 projectionInverse;
    row_major float4x4 viewProjPrevious;   // Added for ReSTIR/Temporal
    row_major float4x4 viewInversePrevious; // Added for ReSTIR/Temporal
    float4 cameraPosition;
    float4 prevCameraPosition;              // Added for ReSTIR/Temporal
    uint frameIndex;
    int albedoIndex;    // RT GBuffer albedo indices
    int normalIndex;    // RT GBuffer normal indices
    int materialIndex;  // RT GBuffer material indices
    int depthIndex;     // RT GBuffer depth indices
    int shadowMapIndex;
    float exposure;
    uint enableTemporal;
    uint enableSpatial;
    uint padding;
};

struct LightConstants {
    row_major float4x4 viewProj;
    float4 position;
    float4 color;
    float4 direction;
    float intensity;
    uint32_t padding[3];
};

struct RayPayload {
    float4 color;
};

struct GLTFVertex {
    float3 position;
    float3 normal;
    float2 texCoord;
};

struct MaterialConstants {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureIndex;
    int normalTextureIndex;
    int metallicRoughnessTextureIndex;
    uint padding[1];
};

struct DrawNodeData {
    row_major float4x4 world;
    uint vertexOffset;
    uint indexOffset;
    uint materialID;
    uint padding;
};

struct Reservoir {
    float3 hitPos;     // Position of the indirect light hit
    float3 hitNormal;  // Normal at the hit point
    float3 radiance;   // Radiance from the hit point
    float w_sum;       // Sum of weights
    float M;           // Number of samples
    float W;           // Normalization weight
    float3 primaryNormal; // For similarity checks
    float3 primaryPos;    // For similarity checks
};

// Weighted Reservoir Sampling helper
// Returns true if the new sample was selected
bool updateReservoir(inout Reservoir r, float3 hitPos, float3 hitNormal, float3 radiance, float weight, float rnd) {
    r.w_sum += weight;
    r.M += 1.0f;
    // luminance as weight, close to 1.0 for bright samples (very likely to be selected)
    if (rnd < weight / r.w_sum) {
        r.hitPos = hitPos;
        r.hitNormal = hitNormal;
        r.radiance = radiance;
        return true;
    }
    return false;
}

// Merge two reservoirs
void mergeReservoirs(inout Reservoir r, Reservoir r2, float weight, float rnd) {
    float M_before = r.M;
    r.w_sum += weight;
    r.M += r2.M;
    if (rnd < weight / r.w_sum) {
        r.hitPos = r2.hitPos;
        r.hitNormal = r2.hitNormal;
        r.radiance = r2.radiance;
    }
}

#endif // COMMON_HLSL
