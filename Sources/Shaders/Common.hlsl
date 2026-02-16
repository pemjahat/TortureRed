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
    float exposure;
    uint enableRestir;
    uint enableAvoidCaustics;
    uint enableIndirectSpecular;
    uint useRTXDI;
    uint numLights;
    uint lightSamplingMode; // 0=uniform, 1=importance sampling (indirect only)
    uint lightLUTBufferIndex; // Index into light LUT buffer
};

float3 ReconstructWorldPos(float2 uv, float depth, float4x4 projectionInverse, float4x4 viewInverse) {
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, projectionInverse);
    viewPos /= viewPos.w;
    float4 worldPos = mul(viewPos, viewInverse);
    return worldPos.xyz;
}

struct LightConstants {
    row_major float4x4 viewProj;
    float4 position;
    float4 color;
    float4 direction;
    float intensity;
    float selectionPDF;
    uint32_t padding[2];
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
    float3 radiance;   // Folded radiance from the hit point (includes albedo)
    float targetPDF;   // Demodulated target PDF for selection
    float w_sum;       // Sum of weights and Normalization weight (at the end of spatial / temporal pass)
    float M;           // Number of samples
};

// Weighted Reservoir Sampling helper
// Returns true if the new sample was selected
bool updateReservoir(inout Reservoir r, float3 hitPos, float3 hitNormal, float3 radiance, float targetPDF, float samplePDF, float rnd) {
    float risWeight = (samplePDF > 0) ? (targetPDF / samplePDF) : 0;
    r.w_sum += risWeight;
    r.M += 1.0f;

    if (rnd * r.w_sum <= risWeight) {
        r.hitPos = hitPos;
        r.hitNormal = hitNormal;
        r.radiance = radiance;
        return true;
    }
    return false;
}

// Merge two reservoirs with a shifted target PDF
// Returns true if the reservoir was updated with the new sample
bool mergeReservoirs(inout Reservoir curRes, Reservoir neighbourRes, float targetPDF, float rnd) {
    float risWeight = targetPDF * neighbourRes.w_sum * neighbourRes.M;

    curRes.w_sum += risWeight;
    curRes.M += neighbourRes.M;

    if (rnd * curRes.w_sum <= risWeight) {
        curRes.hitPos = neighbourRes.hitPos;
        curRes.hitNormal = neighbourRes.hitNormal;
        curRes.radiance = neighbourRes.radiance;
        return true;
    }
    return false;
}

#endif // COMMON_HLSL
