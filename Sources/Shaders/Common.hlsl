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
    float4 irCacheCameraPosition;           // frozen debug pos; == cameraPosition when not frozen
    uint frameIndex;
    int albedoIndex;    // RT GBuffer albedo indices
    int normalIndex;    // RT GBuffer normal indices
    int materialIndex;  // RT GBuffer material indices
    int depthIndex;     // RT GBuffer depth indices
    float exposure;
    uint enableRestir;
    uint enableAvoidCaustics;
    uint enableIndirectSpecular;
    uint enableRasterIndirectGI;
    uint useRTXDI;
    uint numLights;
    uint lightSamplingMode; // 0=uniform, 1=importance sampling (indirect only)
    uint lightLUTBufferIndex; // Index into light LUT buffer
    uint debugIrCache;
    int  debugIrCacheCascadeFilter; // -1 = all cascades, 0-7 = specific cascade
    uint screenWidth;
    uint screenHeight;
    // SHaRC (Spatial Hash Radiance Cache) parameters
    float sharcSceneScale;
    uint  sharcAccumulationFrameNum;
    uint  sharcStaleFrameNum;
    uint  sharcDebug;          // 0=off, 1=SHaRC output, 2=bounce heatmap
};

struct BindlessIndices
{
    uint InputIdx0;
    uint InputIdx1;
    uint OutputIdx0;
    uint OutputIdx1;
};

// Irradiance Cache buffer
// WorldPos -> CellIndex = cascade * 32^3 + z * 32^2 + y * 32 + x

// Meta - track global count (total, alive, queue for tracing, pool)
// Pool - list index available entries, probe allocated - decrement pool top atomic, probe expired - push stack
// GridMeta - map by cell index, give back entry index and flags (occupied, just allocated)
// EntryCell - reverse map of GridMeta, map by entry index, give back cell index
// Irradiance - actual storage of irradiance data (in: entry index)
// Life - storage of aging (in: entry index), when is not lookup, age + 1
// Indirection - list of live entries, rebuilt by Age pass, used by Update pass to iterate only valid probes
// TraceArg - read live probe from Meta, IrCacheUpdate use this for indirect dispatch, each thread handle one probe

// Irradiance Cache Pass
// PoolInit (once)
// PrepareAge - Age (indirect) - PrepareTrace - Update (indirect)

// Bindless heap indices for all spatial irradiance cache buffers.
// Bound as root constants at b2.  All indices refer to UAV descriptors so
// shaders can freely read or write each resource via RW types.
struct IrCacheBindlessIndices
{
    uint MetaBufIdx;              // RWByteAddressBuffer       — 4 × uint meta counters
    uint PoolBufIdx;              // RWStructuredBuffer<uint>  — free-entry pool
    uint GridMetaBufIdx;          // RWByteAddressBuffer       — uint per cell (entryIdx<<3 | flags)
    uint EntryCellBufIdx;         // RWStructuredBuffer<uint>  — entry → cell back-link
    uint IrradianceBufIdx;        // RWStructuredBuffer<Reservoir> — probe reservoir payload per entry
    uint LifeBufIdx;              // RWByteAddressBuffer       — life uint per entry
    uint IndirectionBufIdx;       // RWStructuredBuffer<uint>  — compact live-entry list
    uint TraceArgsBufIdx;         // RWByteAddressBuffer       — 3 × uint indirect dispatch args
    uint PosBufIdx;               // RWStructuredBuffer<float4> — applied probe positions (w=1 if valid)
    uint RepropBufIdx;            // RWStructuredBuffer<float4> — voted proposal positions (written during Update)
    uint ReproposalCountBufIdx;   // RWByteAddressBuffer       — per-entry vote counters
};

// SHaRC (Spatial Hash Radiance Cache) bindless buffer indices.
// Bound as root constants at b2 when SHaRC passes are active.
#define SHARC_HASH_ENTRIES_NUM (4 * 1024 * 1024)
struct SharcBindlessIndices
{
    uint HashEntriesBufIdx;    // RWStructuredBuffer<uint64_t>              — hash table  (32 MB)
    uint AccumulationBufIdx;   // RWStructuredBuffer<SharcAccumulationData> — per-frame accumulation (64 MB)
    uint ResolvedBufIdx;       // RWStructuredBuffer<SharcPackedData>       — temporal EMA output  (64 MB)
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
    int alphaMode;
    float alphaCutoff;
};

struct DrawNodeData {
    row_major float4x4 world;
    uint vertexOffset;
    uint indexOffset;
    uint materialID;
    uint padding;
};

struct Reservoir {
    float3 hitPos;     // Position of the sampled first-bounce candidate
    float3 hitNormal;  // Surface normal at the sampled candidate
    float3 radiance;   // Continuation radiance transported from that candidate
    float w_sum;       // Sum of RIS weights
    float W;           // Normalization weight used at resolve
    float M;           // Number of samples
};

// Weighted Reservoir Sampling helper
// Returns true if the new sample was selected
bool updateReservoir(inout Reservoir r, float3 hitPos, float3 hitNormal, float3 radiance, float risWeight, float rnd) {
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
bool mergeReservoirs(inout Reservoir curRes, Reservoir neighbourRes, float shiftedTargetPDF, float rnd) {
    float risWeight = shiftedTargetPDF * neighbourRes.W * neighbourRes.M;

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
