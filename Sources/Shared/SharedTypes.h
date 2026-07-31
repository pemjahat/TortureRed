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

// Anti-aliasing / post-processing modes
#define AA_MODE_NONE          0u  // No AA — single sample, no accumulation, no TAA
#define AA_MODE_ACCUMULATION  1u  // Progressive accumulation (path tracer converges over time)
#define AA_MODE_TAA           2u  // Temporal Anti-Aliasing with sub-pixel jitter

#define RESTIR_RESERVOIR_DEBUG_OFF        0u
#define RESTIR_RESERVOIR_DEBUG_POSITION   1u
#define RESTIR_RESERVOIR_DEBUG_NORMAL     2u
#define RESTIR_RESERVOIR_DEBUG_RADIANCE   3u
#define RESTIR_RESERVOIR_DEBUG_WEIGHTSUM  4u
#define RESTIR_RESERVOIR_DEBUG_SOURCE_PDF 5u
#define RESTIR_RESERVOIR_DEBUG_TARGET_PDF 6u
#define RESTIR_RESERVOIR_DEBUG_RIS_WEIGHT 7u
#define RESTIR_RESERVOIR_DEBUG_TEMPORAL_TARGET_PDF 8u
#define RESTIR_RESERVOIR_DEBUG_SPATIAL_SHIFTED_TARGET_PDF 9u
#define RESTIR_RESERVOIR_DEBUG_W         10u

// ReSTIR DI debug modes
#define RESTIR_DI_DEBUG_OFF            0u
#define RESTIR_DI_DEBUG_LIGHT_INDEX    1u
#define RESTIR_DI_DEBUG_M_COUNT        2u
#define RESTIR_DI_DEBUG_WEIGHT         3u
#define RESTIR_DI_DEBUG_VISIBILITY_AGE 4u

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
    uint enableReservoirLobeCheck; // 1 = lobe-aware temporal/spatial reuse (lobe-matched PDF, roughness-scaled Jacobian/history)
    uint enableNrdRelax;
    uint enableNrdValidation;
    float rtrRoughReuseThreshold; // Roughness above which specular reuses diffuse candidate (default 0.6)

    // TAA / Temporal Super-Resolution
    float2 taaJitter;           // Sub-pixel jitter in pixel units of internal resolution
    uint   taaEnabled;          // 1 = TAA active
    uint   internalWidth;       // Internal render resolution width
    uint   internalHeight;      // Internal render resolution height
    uint   outputWidth;         // Output (swap chain) resolution width (1920)
    uint   outputHeight;        // Output (swap chain) resolution height (1080)
    uint   taaHistoryIndex;     // Ping-pong index for TAA history (0 or 1)
    float  taaUpsamplingFactor; // Temporal upsampling factor (1.0 - 4.0)
    uint   taaFrameCounter;    // Monotonically increasing counter for TAA jitter sequence (never reset)
    uint   _pad0;              // 8-byte padding (2x uint) to align projectionInverseUnjittered to 16 bytes (offset 512)
    uint   _pad1;
    ROW_MAJOR float4x4 projectionInverseUnjittered; // Unjittered projection inverse for motion vectors

    // ReSTIR DI (Direct Illumination) flags
    uint enableRestirDI;              // 1 = ReSTIR DI active (direct lighting from local lights)
    uint restirDIDebugMode;           // RESTIR_DI_DEBUG_* enum
    uint _diPad0;
    uint _diPad1;
};

struct BindlessIndices {
    uint InputIdx0;
    uint InputIdx1;
    uint InputIdx2;
    uint OutputIdx0;
    uint OutputIdx1;
    uint OutputIdx2;
    uint PathVizLineBufferIdx; // UAV (CS write) or SRV (VS read) index for path viz lines
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

// Meshlet rasterization PSO bin — determines which PSO permutation renders a meshlet.
// Must match PipelineBin enum in Renderer.h and the shader's GetBin() function.
#define RASTER_BIN_OPAQUE       0u  // Standard opaque: back-face cull, no alpha test
#define RASTER_BIN_ALPHA_MASKED 1u  // Alpha-tested: no cull, discard in PS

struct MaterialConstants {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureIndex;
    int normalTextureIndex;
    int metallicRoughnessTextureIndex;
    int alphaMode;
    float alphaCutoff;
    uint RasterBin;  // RASTER_BIN_OPAQUE or RASTER_BIN_ALPHA_MASKED
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
    float firstBounceHitT;
    // Bit 31 of historyAge encodes lobe type: 0 = diffuse, 1 = specular.
    // Bits [0-30] store the actual age. MAX_HISTORY_AGE (12) fits comfortably.
    uint historyAge;
};

// Use these helpers (defined in CommonTracing.hlsl) to access historyAge safely.
#define RESERVOIR_SPECULAR_FLAG 0x80000000u

// Per-pixel DI reservoir — stores the winning local light index from ReSTIR DI.
// 32 bytes total (8 x uint32).
struct DIRreservoir {
    float w_sum;              // Running sum of RIS weights
    float W;                  // Normalized unbiased weight W = w_sum / (M * p_hat_selected)
    float M;                  // Effective sample count (history length)
    float targetPdf;          // Target PDF p_hat of the currently selected light
    uint  selectedLightIndex; // Index into LightsBuffer of the winning light (bits[0-15])
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

// Per-pixel diffuse candidate data written by RTDGI temporal, read by RTR temporal
struct DiffuseCandidate {
    float3 hitPos;
    float  hitT;
    float3 hitNormal;
    float  _pad0;
    float3 radiance;
    float  _pad1;
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

// =============================================================================
// Meshlet Pipeline Types (shared CPU/GPU)
// =============================================================================

// Maximum meshlet sizes (compile-time constants, must match bin cache embedding)
static const uint MESHLET_MAX_VERTICES  = 64;
static const uint MESHLET_MAX_TRIANGLES = 124;

struct Meshlet {
    uint VertexOffset;      // Offset into MeshletVertexBuffer (uint32 indices)
    uint TriangleOffset;    // Offset into MeshletTriangleBuffer (packed triangles)
    uint VertexCount;       // Number of unique vertices (max 64)
    uint TriangleCount;     // Number of triangles (max 124)
};

struct MeshletTriangle {
    uint V0 : 10;
    uint V1 : 10;
    uint V2 : 10;
    uint    : 2;  // padding
};

struct MeshletBounds {
    float3 LocalCenter;
    float3 LocalExtents;
};

// Replaces DrawNodeData for the meshlet path
// Each *Offset field is an element offset into the corresponding global StructuredBuffer<T>
struct MeshData {
    uint PositionOffset;        // Element offset into GlobalPositions[]  (float3)
    uint NormalOffset;          // Element offset into GlobalNormals[]    (uint, RGB10A2_SNORM)
    uint UVOffset;              // Element offset into GlobalUVs[]        (uint, RG16_FLOAT)
    uint MeshletOffset;         // Element offset into GlobalMeshlets[]   (Meshlet)
    uint MeshletVertexOffset;   // Element offset into GlobalMeshletVertices[] (uint)
    uint MeshletTriangleOffset; // Element offset into GlobalMeshletTriangles[] (MeshletTriangle)
    uint MeshletBoundsOffset;   // Element offset into GlobalMeshletBounds[]   (MeshletBounds)
    uint MeshletCount;
    uint MaterialIndex;
    uint GlobalMeshletStart;   // Global meshlet index where this entry's meshlets begin (CPU prefix-sum)
    uint _pad1;
    uint _pad2;
};

struct InstanceData {
    ROW_MAJOR float4x4 LocalToWorld;
    uint MeshDataIndex;         // Index into global MeshData[] buffer
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Culling token — the unit of work through the cull pipeline
struct MeshletCandidate {
    uint InstanceID;    // Index into InstanceData[]
    uint MeshletIndex;  // Per-mesh meshlet index
};

struct MeshletCandidateDebug {
    uint VertexCount;
    uint TriangleCount;
};

// =============================================================================
// Meshlet Binning / Rasterize Params (shared CPU/GPU, passed via root constants b1)
// =============================================================================

// Passed to all 4 binning CS passes via root param 12 (b1).
// Each pass uses only the fields it needs; unused fields are zero.
struct BinningParams {
    uint NumBins;                       // NUM_RASTER_BINS (2)
    uint VisibleMeshletsIdx;            // SRV index of VisibleMeshlets[]
    uint VisibleMeshletsCounterIdx;     // SRV index of VisibleMeshletsCounter
    uint MeshletCountsIdx;              // SRV index of MeshletCounts[] (for AllocateBinRanges)
    uint RWMeshletCountsIdx;            // UAV index of MeshletCounts[]
    uint RWMeshletOffsetAndCountsIdx;   // UAV index of MeshletOffsetAndCounts[]
    uint RWBinnedMeshletsIdx;           // UAV index of BinnedMeshlets[]
    uint RWGlobalMeshletCounterIdx;     // UAV index of GlobalMeshletCounter
    uint RWDispatchArgumentsIdx;        // UAV index of ClassifyDispatchArgs
    uint RWDispatchMeshArgsIdx;         // UAV index of DispatchMeshArgs[] — uint3[NUM_BINS], used as INDIRECT_ARGUMENT
};

// Passed to the Mesh Shader rasterize pass via root param 12 (b1).
struct RasterParams {
    uint BinIndex;              // Which bin this DispatchMesh call is for
    uint VisibleMeshletsIdx;    // SRV index of VisibleMeshlets[]
    uint BinnedMeshletsIdx;     // SRV index of BinnedMeshlets[]
    uint MeshletBinDataIdx;     // SRV index of MeshletOffsetAndCounts[]
};

// Passed to the debug mesh shader (MeshletRasterizeDebugMS.hlsl) via root param 12 (b1).
// CPU iterates all meshlets and issues one DispatchMesh(1,1,1) per meshlet,
// setting InstanceID and MeshletIndex directly — no binning indirection.
struct DebugRasterParams {
    uint InstanceID;    // Index into GlobalInstanceData[]
    uint MeshletIndex;  // Per-instance local meshlet index (into md.MeshletOffset + MeshletIndex)
    uint _pad0;
    uint _pad1;
};

// =============================================================================
// HZB (Hierarchical Z-Buffer) construction
// Built via AMD FidelityFX SPD (fetched via CMake FetchContent, see CMakeLists.txt). Bound as a root CBV
// (b0) on MeshletPass's dedicated m_HZBRootSignature. All texture/buffer access
// inside HZB.hlsl goes through ResourceDescriptorHeap[idx] bindless indices —
// no members here are arrays, to avoid HLSL cbuffer array-packing (16 bytes per
// element) mismatching this struct's plain C++ layout. Per-mip HZB UAV indices
// instead live in a small StructuredBuffer<uint> (MipIndicesSRVIdx), which packs
// tightly on both sides.
struct HZBConstants {
    uint  DepthSRVIdx;       // SRV index of the source depth buffer (GBuffer.depth)
    uint  MipIndicesSRVIdx;  // SRV index of StructuredBuffer<uint> — one HZB mip UAV index per element
    uint  SpdCounterUAVIdx;  // UAV index of the SPD global atomic counter (RWStructuredBuffer<uint>[1])
    uint  NumMips;           // Number of HZB mip levels (<= 12, SPD's max)
    uint  NumWorkGroups;     // SpdSetup() output — total thread groups dispatched (X*Y)
    uint  WorkGroupOffsetX;  // SpdSetup() output — always 0 (HZB has no sub-rect offset)
    uint  WorkGroupOffsetY;
    uint  Width;             // HZB mip 0 width, in texels
    uint  Height;            // HZB mip 0 height, in texels
    float DimensionsInvX;    // 1 / Width  — used by HZBInitCS to compute the source-depth UV
    float DimensionsInvY;    // 1 / Height
    uint  _pad0;
};

// Debug visualization of the raw HZB mip chain (Sources/Shaders/HZBDebugView.hlsl),
// bound at the same root param 13 (b2) slot VisibilityDebugView.hlsl's DebugParams
// uses — reused across debug-only CS PSOs, never bound simultaneously.
struct HZBDebugParams {
    uint HZBSRVIdx;    // SRV index of the HZB texture (all mips)
    uint MipLevel;     // Which mip to visualize
    uint OutputUAVIdx; // UAV index of FullScreenDebugTex (R16G16B16A16_FLOAT)
    uint Width;        // Output (FullScreenDebugTex) width
    uint Height;       // Output (FullScreenDebugTex) height
};

#endif // SHARED_TYPES_H
