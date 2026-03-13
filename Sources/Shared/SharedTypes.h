#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#ifdef __cplusplus
    #include <DirectXMath.h>
    #include <cstdint>
    using float4   = DirectX::XMFLOAT4;
    using float4x4 = DirectX::XMFLOAT4X4;
    using float3   = DirectX::XMFLOAT3;
    using float2   = DirectX::XMFLOAT2;
    using uint     = uint32_t;
    #define ROW_MAJOR 
#else
    #define ROW_MAJOR row_major
#endif

// Core Frame Constants
struct FrameConstants {
    ROW_MAJOR float4x4 viewProj;
    ROW_MAJOR float4x4 viewInverse;
    ROW_MAJOR float4x4 projectionInverse;
    ROW_MAJOR float4x4 viewProjPrevious;
    ROW_MAJOR float4x4 viewInversePrevious;
    float4 cameraPosition;
    float4 prevCameraPosition;
    uint frameIndex;
    int albedoIndex;
    int normalIndex;
    int materialIndex;
    int depthIndex;
    float exposure;
    uint enableRestir;
    uint enableAvoidCaustics;
    uint enableIndirectSpecular;
    uint enableRasterIndirectGI;
    uint useRTXDI;
    uint numLights;
    uint lightSamplingMode;
    uint lightLUTBufferIndex;
    uint screenWidth;
    uint screenHeight;
    float sharcSceneScale;
    uint sharcAccumulationFrameNum;
    uint sharcStaleFrameNum;
    uint sharcDebug;
};

struct BindlessIndices {
    uint InputIdx0;
    uint InputIdx1;
    uint OutputIdx0;
    uint OutputIdx1;
};

struct LightConstants {
    ROW_MAJOR float4x4 viewProj;
    float4 position;
    float4 color;
    float4 direction;
    float intensity;
    float selectionPDF;
    uint padding[2];
};

struct MaterialConstants {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureIndex;
    int normalTextureIndex;
    int metallicRoughnessTextureIndex;
    int alphaMode;
    float alphaCutoff;
};

struct DrawNodeData {
    ROW_MAJOR float4x4 world;
    uint vertexOffset;
    uint indexOffset;
    uint materialID;
    uint padding;
};

struct Reservoir {
    float3 hitPos;
    float3 hitNormal;
    float3 radiance;
    float w_sum;
    float W;
    float M;
    uint historyAge;
};

struct SharcBindlessIndices {
    uint HashEntriesBufIdx;
    uint AccumulationBufIdx;
    uint ResolvedBufIdx;
};

struct IrCacheBindlessIndices {
    uint MetaBufIdx;
    uint PoolBufIdx;
    uint GridMetaBufIdx;
    uint EntryCellBufIdx;
    uint IrradianceBufIdx;
    uint LifeBufIdx;
    uint IndirectionBufIdx;
    uint TraceArgsBufIdx;
    uint PosBufIdx;
    uint RepropBufIdx;
    uint ReproposalCountBufIdx;
};

#endif // SHARED_TYPES_H
