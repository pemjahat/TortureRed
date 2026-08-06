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

// alphaMode values (shared CPU/GPU — must match cgltf alpha mode enum)
// 0 = Opaque, 1 = Mask (alpha-test), 2 = Blend (alpha-blended; rejected by meshlet pipeline)
#define ALPHA_MODE_OPAQUE 0u
#define ALPHA_MODE_MASK   1u
#define ALPHA_MODE_BLEND  2u

struct MaterialConstants {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureIndex;
    int normalTextureIndex;
    int metallicRoughnessTextureIndex;
    int alphaMode;      // ALPHA_MODE_OPAQUE / ALPHA_MODE_MASK / ALPHA_MODE_BLEND
    float alphaCutoff;
    uint _pad;
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

// Bounding SPHERE (not AABB) — chosen over AABB because a sphere transforms
// trivially under any affine LocalToWorld (center = mul(center,M), radius scaled
// by the matrix's max axis scale), whereas an AABB must be fully recomputed from
// its 8 corners on every transform. Computed once at load time via meshoptimizer's
// meshopt_computeMeshletBounds (tight cluster bounding sphere), consumed by
// FrustumCullMeshletSphere/HZBCull in MeshletCommon.hlsli / MeshletTwoPassCull.hlsl.
struct MeshletBounds {
    float3 LocalCenter;
    float  LocalRadius;
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
    uint AlphaMode;            // ALPHA_MODE_OPAQUE / ALPHA_MODE_MASK / ALPHA_MODE_BLEND
    uint _pad;
};

struct InstanceData {
    ROW_MAJOR float4x4 LocalToWorld;
    uint MeshDataIndex;         // Index into global MeshData[] buffer
    uint BoundsIndex;           // Index into InstanceBounds[] — currently equal to MeshDataIndex
    uint _pad1;
    uint _pad2;
};

// LOCAL-space bounding SPHERE of a SubMesh — the enclosing sphere of the union
// of that SubMesh's meshlet bounding spheres (meshopt_computeSphereBounds over
// prim.meshletBounds), computed once on CPU in Model::BuildMeshlets().
// One entry per MeshData, built once on CPU in Model::CreateMeshletResources()
// Pass 1 (primitive iteration) — bounds are a GEOMETRY property, not an
// instance property, so shared meshes share one entry.
// Cull shaders transform this by the instance's PER-FRAME LocalToWorld
// (D3D12_Research pattern: local bounds × current matrix), so culling
// follows animated/moving instances correctly.
struct InstanceBounds {
    float3 BoundsCenter;   // Local-space center
    float  BoundsRadius;   // Local-space bounding-sphere radius
};

// Culling token — the unit of work through the cull pipeline
struct MeshletCandidate {
    uint InstanceID;    // Index into InstanceData[]
    uint MeshletIndex;  // Per-mesh meshlet index
};

// =============================================================================
// Meshlet Dispatch / Rasterize Params (shared CPU/GPU, passed via root constants b1)
// =============================================================================

// Passed to the Mesh Shader rasterize pass via root param 12 (b1).
// With no binning, the mesh shader directly indexes VisibleMeshlets[SV_GroupID].
struct RasterParams {
    uint VisibleMeshletsIdx;         // SRV index of VisibleMeshlets[] (for rasterize MS)
    uint DispatchMeshArgsIdx;        // UAV index of DispatchMeshArgs (for BuildDispatchMeshArgsCS)
    uint VisibleMeshletsCounterIdx;  // SRV index of VisibleMeshletsCounter (for BuildDispatchMeshArgsCS)
    uint _pad;
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

// -----------------------------------------------------------------------------
// Occluded-rect debug recording (task007 mode 1) — Sources/Shaders/OccludedRectDebug.hlsl
//
// Cull shaders (MeshletTwoPassCull.hlsl) append one record per instance/meshlet
// rejected by HZBCull; the draw pass rasterizes the recorded NDC rects over a
// dimmed scene-albedo background into FullScreenDebugTex.
// -----------------------------------------------------------------------------
#define MAX_OCCLUDED_RECT_DEBUG 16384

struct OccludedRectDebug {
    float2 RectMinNDC;   // Clamped NDC rect (HZBCull step 2/4 output)
    float2 RectMaxNDC;
    float  NearestDepth; // Object side of the step-7 compare (reverse-Z: larger = closer)
    float  HZBDepth;     // HZB side of the compare
    uint   Mip;          // HZB mip the test used (step 5)
    uint   Phase;        // TWO_PASS_PHASE_* of the cull pass that occluded it
    uint   Kind;         // 0 = meshlet, 1 = instance
    uint   _pad0;
    float2 SampleMinNDC; // NDC rect spanned by the 4 HZB sample texels (step 5)
    float2 SampleMaxNDC; // — compare against RectMin/MaxNDC for the texel:object ratio
};

// Meshlet debug view mode that tints surviving meshlets by the HZB mip their
// occlusion test used (task007 mode 3). Matches the combo in Application::RenderImGui
// and ViewMode in VisibilityDebugView.hlsl. 0=Off, 1=Instance, 2=Meshlet, 3=Primitive.
#define MESHLET_DEBUG_MIP_TINT 4

// Draw-pass parameters — root constants at main-root-signature param 13 (b2),
// same slot as HZBDebugParams (debug-only PSOs, never bound simultaneously).
struct OccludedRectDrawParams {
    uint RectsSRVIdx;      // Bindless SRV of the OccludedRectDebug buffer
    uint RectsCountSRVIdx; // Bindless SRV of the record counter
    uint OutputUAVIdx;     // UAV index of FullScreenDebugTex
    uint Width;            // Output (FullScreenDebugTex) width
    uint Height;           // Output height
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// =============================================================================
// GPU on-screen debug text/lines — Sources/Shaders/DebugTextRender.hlsli/.hlsl
//
// Any shader appends packed text/line instances to one RWByteAddressBuffer via
// the producer API; at end of frame a 1-thread CS builds two indirect draw args
// (resetting the counters) and two ExecuteIndirect draws rasterize glyphs+lines
// onto the backbuffer. Port of D3D12_Research ShaderDebugRenderer, bindless variant.
// =============================================================================
#define DEBUG_TEXT_MAX_CHARS 8192
#define DEBUG_TEXT_MAX_LINES 32768

// RWByteAddressBuffer layout offsets (bytes) — must match DebugTextRenderer::Data
#define DEBUG_TEXT_COUNTER_OFFSET   0
#define DEBUG_LINE_COUNTER_OFFSET   4
#define DEBUG_TEXT_COUNTERS_SIZE    16
#define DEBUG_TEXT_INSTANCES_OFFSET DEBUG_TEXT_COUNTERS_SIZE
#define DEBUG_LINE_INSTANCES_OFFSET (DEBUG_TEXT_INSTANCES_OFFSET + DEBUG_TEXT_MAX_CHARS * 32)

// One glyph of the debug font atlas (built from an ImFontAtlas)
struct DebugGlyph {
    float2 MinUV;
    float2 MaxUV;
    float2 Dimensions;   // Quad size in pixels at scale 1
    float2 Offset;       // Offset from cursor to quad top-left
    float  AdvanceX;     // Cursor advance in pixels
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

// Unpacked instances (32 bytes each) — pack later if bandwidth matters
struct DebugCharInstance {
    float2 Position;     // Pixels, top-left of glyph quad
    uint   Character;    // ASCII codepoint
    float  Scale;        // Size multiplier (1 = native font size)
    float4 Color;
};

struct DebugLineInstance {
    float3 A;
    uint   ColorA;       // RGBA8; LSB of ColorA = screen-space ([0,1]) flag
    float3 B;
    uint   ColorB;       // RGBA8; world-space endpoints need FrameCB in the line VS
};

// End-of-frame pass parameters — root constants b1 (main root signature param 12,
// shared by the args-builder CS and the glyph/line raster PSOs)
struct DebugTextRenderParams {
    uint  DataSRVIdx;      // ByteAddressBuffer SRV (instances)
    uint  DataUAVIdx;      // RWByteAddressBuffer UAV (args builder resets counters)
    uint  ArgsUAVIdx;      // RWStructuredBuffer<uint4> indirect draw args (2 entries)
    uint  GlyphSRVIdx;     // StructuredBuffer<DebugGlyph>
    uint  FontAtlasSRVIdx; // Texture2D<float4> font atlas
    float TargetWidth;     // Backbuffer width in pixels
    float TargetHeight;    // Backbuffer height in pixels
    uint  _pad0;
};

// Depth-readout producer (mode 5) parameters — root constants b2 (param 13)
struct DepthReadoutParams {
    uint  RectsSRVIdx;      // Bindless SRV of the OccludedRectDebug buffer
    uint  RectsCountSRVIdx; // Bindless SRV of the record counter
    uint  DataUAVIdx;       // Debug text render-data UAV
    uint  GlyphSRVIdx;      // StructuredBuffer<DebugGlyph>
    float FontSize;         // Native font size in pixels
    float BackbufferWidth;
    float BackbufferHeight;
    uint  MaxLabels;        // Cap on labels emitted per frame
};

// =============================================================================
// Two-Pass Occlusion Culling — constants shared by CullInstancesCS / CullMeshletsCS
// =============================================================================

// Phase selectors (passed as root CBV fields, not shader defines)
#define TWO_PASS_PHASE_FIRST   0u
#define TWO_PASS_PHASE_SECOND  1u

struct TwoPassCullConstants {
    uint  NumInstances;              // Total instances in InstanceData[]
    uint  NumMeshlets;               // Total meshlets across all instances
    uint  HZBSRVIdx;                 // Bindless SRV index of the HZB texture (all mips)
    uint  HZBMipCount;               // Number of HZB mip levels
    uint  HZBWidth;                  // HZB mip 0 width
    uint  HZBHeight;                 // HZB mip 0 height
    uint  CandidateMeshletsCounterIdx; // UAV bindless index of CandidateMeshletsCounter
    uint  CandidateMeshletsUAVIdx;     // UAV bindless index of CandidateMeshlets[]
    uint  OccludedInstancesCounterIdx; // UAV bindless index of OccludedInstancesCounter
    uint  OccludedInstancesUAVIdx;     // UAV bindless index of OccludedInstances[] (write)
    uint  OccludedInstancesSRVIdx;     // Bindless SRV index of OccludedInstances[] (read by Phase 2)
    uint  VisibleMeshletsUAVIdx;       // UAV bindless index of VisibleMeshlets[]
    uint  VisibleMeshletsCounterUAVIdx;// UAV bindless index of VisibleMeshletsCounter
    uint  Phase;                       // TWO_PASS_PHASE_FIRST / SECOND / FRUSTUM_ONLY
    uint  EnableOcclusion;             // 0 = frustum-only, 1 = occlusion culling active
    // Occluded-rect debug recording (task007 mode 1). When DebugRecordOccluded is 0
    // the cull shaders never touch the debug buffers (zero overhead).
    uint  DebugRecordOccluded;         // 1 = append OccludedRectDebug records for HZB-rejected candidates
    uint  OccludedRectsUAVIdx;         // UAV bindless index of OccludedRectDebug[]
    uint  OccludedRectsCounterUAVIdx;  // UAV bindless index of the record counter

    // Mip-selection tint (task007 mode 3): one mip value per visible-meshlet slot,
    // read back by the debug overlay via the vis token's candidateIndex.
    uint  DebugRecordMip;              // 1 = write the occlusion test's mip per surviving meshlet
    uint  VisibleMeshletMipsUAVIdx;    // UAV bindless index of VisibleMeshletMips[]
};

// =============================================================================
// CullStats — per-phase culling counters copied from the functional counters
// into a dedicated debug buffer after each phase completes. Consumed by
// CullStatsCS for on-screen overlay.
// =============================================================================

// CullStatsBuffer layout (RWStructuredBuffer<uint>[16]):
#define CULL_STATS_P1_CANDIDATE_MESHLETS   0  // CandidateMeshletsCounter after Phase 1
#define CULL_STATS_P1_VISIBLE_MESHLETS     1  // VisibleMeshletsCounter after Phase 1
#define CULL_STATS_P1_OCCLUDED_INSTANCES   2  // OccludedInstancesCounter (deferred) after Phase 1
#define CULL_STATS_P2_CANDIDATE_MESHLETS   3  // CandidateMeshletsCounter after Phase 2
#define CULL_STATS_P2_VISIBLE_MESHLETS     4  // VisibleMeshletsCounter after Phase 2
#define CULL_STATS_P2_OCCLUDED_INSTANCES   5  // OccludedInstances in Phase 2
#define CULL_STATS_COUNT                   8

// Passed to CopyCullStatsCS via root constants.
// For P1: BaseSlot=0 copies Candidate[0],Visible[0],Occluded[0] → Stats[0..2]
// For P2: BaseSlot=4 copies Candidate[0],Visible[0] → Stats[4..5], Occluded[0]→Stats[7]
//         plus P2 OccludedInstancesP2SRVIdx → Stats[6]
struct CullStatsCopyParams {
    uint CandidateCounterSRVIdx;     // SRV heap index of CandidateMeshletsCounter (1-element per-phase)
    uint VisibleCounterSRVIdx;       // SRV heap index of VisibleMeshletsCounter
    uint OccludedCounterSRVIdx;      // SRV heap index of OccludedInstancesCounter (P1=deferred, P2=occluded-P2)
    uint StatsBufferUAVIdx;          // UAV heap index of CullStatsBuffer
    uint BaseSlot;                   // 0 for Phase 1, 4 for Phase 2
    uint _pad[3];
};

// Passed to CullStatsCS via root constants — reads the debug stats buffer and
// renders an on-screen table via the GPU debug-text system.
struct CullStatsParams {
    uint StatsBufferSRVIdx;          // SRV heap index of CullStatsBuffer
    uint DataUAVIdx;                 // Debug text render-data UAV index
    uint GlyphSRVIdx;                // Debug glyph atlas SRV index
    float FontSize;                  // Native font line height in pixels
    uint TotalInstances;             // CPU-provided: total instances in scene
    uint TotalMeshlets;              // CPU-provided: total meshlets across all instances
    uint BackbufferWidth;
    uint BackbufferHeight;
    float StartX;                    // Top-left pixel X
    float StartY;                    // Top-left pixel Y
};

#endif // SHARED_TYPES_H
