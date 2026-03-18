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

#define RESTIR_RESERVOIR_DEBUG_OFF        0u
#define RESTIR_RESERVOIR_DEBUG_POSITION   1u
#define RESTIR_RESERVOIR_DEBUG_NORMAL     2u
#define RESTIR_RESERVOIR_DEBUG_RADIANCE   3u
#define RESTIR_RESERVOIR_DEBUG_WEIGHTSUM  4u

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
    uint restirReservoirDebugMode;
    uint mouseSelectedPixelX;
    uint mouseSelectedPixelY;
    uint pathVizEnabled;  // 1 for exactly one frame after left-click
    uint _fcPad3;
};

struct BindlessIndices {
    uint InputIdx0;
    uint InputIdx1;
    uint OutputIdx0;
    uint OutputIdx1;
    uint PathVizLineBufferIdx; // UAV (CS write) or SRV (VS read) index for path viz lines
    uint _bPad;
};

// Path visualization line types (bits[3:0] of typeAndValid)
// bit[4] = valid flag
#define PATHVIZ_TYPE_PRIMARY    0u // yellow:  camera -> first surface
#define PATHVIZ_TYPE_BOUNCE1    1u // white:   first surface -> bounce 1
#define PATHVIZ_TYPE_BOUNCE2    2u // cyan:    bounce 1 -> bounce 2
#define PATHVIZ_TYPE_BOUNCE3    3u // blue:    bounce 2 -> bounce 3
#define PATHVIZ_TYPE_TEMPORAL   4u // orange:  center pixel -> temporal sample
#define PATHVIZ_TYPE_SPATIAL    5u // magenta: center pixel -> spatial winner
#define MAX_PATH_VIZ_LINES  16

struct PathVizLine {
    float3 start;
    uint typeAndValid; // bits[3:0] = type, bit[4] = valid flag
    float3 end;
    uint _pad;
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
