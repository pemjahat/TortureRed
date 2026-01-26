#ifndef COMMON_HLSL
#define COMMON_HLSL

// Alignment matched with C++ structures in GraphicsTypes.h

struct FrameConstants {
    row_major float4x4 viewProj;
    row_major float4x4 viewInverse;
    row_major float4x4 projectionInverse;
    float4 cameraPosition;
    uint frameIndex;
    int albedoIndex;    // RT GBuffer albedo indices
    int normalIndex;    // RT GBuffer normal indices
    int materialIndex;  // RT GBuffer material indices
    int depthIndex;     // RT GBuffer depth indices
    int shadowMapIndex;
    uint32_t padding[2];
};

struct LightConstants {
    row_major float4x4 viewProj;
    float4 position;
    float4 color;
    float4 direction;
};

struct PrimitiveData {
    int vertexBufferIndex;
    int indexBufferIndex;
    uint32_t materialIndex;
    uint32_t padding;
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
};

struct DrawNodeData {
    row_major float4x4 world;
    uint vertexOffset;
    uint indexOffset;
    uint materialID;
    uint padding;
};

#endif // COMMON_HLSL
