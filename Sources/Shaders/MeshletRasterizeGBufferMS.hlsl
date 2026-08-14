#include "MeshletCommon.hlsli"
#include "VisibilityBuffer.hlsli"

/*
    Mesh Shader + Pixel Shader for GPU-driven meshlet rasterization —
    DIRECT-TO-GBUFFER variant.

    Used when the Visibility Buffer pipeline is disabled. Unlike the Visibility
    Buffer variant (MeshletRasterizeMS.hlsl), this shader does the full material
    sampling / GBuffer write inline in the pixel shader — one shade per rasterized
    fragment.

    MSMain: one thread group per visible meshlet (SV_GroupID → MeshletCandidate directly,
            no bin indirection). 32 threads.
    PSMain: writes GBuffers directly — albedo (R8G8B8A8_UNORM), normal (R16G16B16A16_FLOAT),
            roughness|metallic (R8G8B8A8_UNORM), plus visibility token (R32_UINT).

    No ALPHA_MASK permutation — single combined PSO handles both opaque and alpha-masked.
    Alpha-blended instances are rejected in CullInstancesCS.
*/

#define NUM_MESHLET_THREADS 32

// --- Bindless resource declarations ---
// Meshlet stream buffers are looked up individually via ResourceDescriptorHeap
// from gRasterParams' flat bindless index fields.

// Material buffer (root SRV param 1, t0 space1)
StructuredBuffer<MaterialConstants> MaterialBuffer : register(t0, space1);

// Bindless textures (space0)
Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0);

// Per-frame constants
ConstantBuffer<FrameConstants> FrameCB : register(b0);

// Raster params (root constants b1)
ConstantBuffer<RasterParams> gRasterParams : register(b1);

// --- Per-primitive output ---
struct PrimitiveAttribute
{
    uint PrimitiveID    : SV_PrimitiveID;
    uint CandidateIndex : CANDIDATE_INDEX;
};

// --- Per-vertex output (full surface data, needed for inline GBuffer shading) ---
struct VertexAttribute
{
    float4 Position : SV_Position;
    float3 WorldPos : WORLD_POS;
    float3 Normal   : NORMAL;
    float2 UV       : TEXCOORD;
    nointerpolation uint MaterialID : MATERIAL_ID;
};

// --- Mesh Shader Entry Point ---
// No bin indirection — groupID directly indexes VisibleMeshlets[].
[outputtopology("triangle")]
[numthreads(NUM_MESHLET_THREADS, 1, 1)]
void MSMain(
    in  uint groupThreadID : SV_GroupIndex,
    in  uint groupID       : SV_GroupID,
    out vertices  VertexAttribute  verts[MESHLET_MAX_VERTICES],
    out indices   uint3            triangles[MESHLET_MAX_TRIANGLES],
    out primitives PrimitiveAttribute primitives[MESHLET_MAX_TRIANGLES])
{
    // Direct indexing: no bins — VisibleMeshlets[candidateIndex] is the meshlet candidate.
    // Phase 2's own meshlets start AFTER Phase 1's range (VisibleMeshletsCounter[TWO_PASS_PHASE_FIRST]
    // is Phase 1's final count) since the counter is only cleared once per frame. 
    // Phase 1 dispatches groupID in [0, C1); Phase 2
    // dispatches groupID in [0, C2) but must offset by C1 to land in its own slice [C1, C1+C2).
    uint candidateIndex = groupID;
    if (gRasterParams.Phase == TWO_PASS_PHASE_SECOND)
    {
        StructuredBuffer<uint> visibleMeshletsCounter = ResourceDescriptorHeap[gRasterParams.VisibleMeshletsCounterIdx];
        candidateIndex += visibleMeshletsCounter[TWO_PASS_PHASE_FIRST];
    }
    StructuredBuffer<MeshletCandidate> visibleMeshlets = ResourceDescriptorHeap[gRasterParams.VisibleMeshletsIdx];
    MeshletCandidate cand = visibleMeshlets[candidateIndex];

    // Meshlet stream buffers — bindless heap lookups
    StructuredBuffer<InstanceData>    globalInstanceData     = ResourceDescriptorHeap[gRasterParams.InstanceDataSRVIdx];
    StructuredBuffer<MeshData>        globalMeshData         = ResourceDescriptorHeap[gRasterParams.MeshDataSRVIdx];
    StructuredBuffer<Meshlet>         globalMeshlets         = ResourceDescriptorHeap[gRasterParams.GlobalMeshletsSRVIdx];
    StructuredBuffer<uint>            globalMeshletVertices  = ResourceDescriptorHeap[gRasterParams.GlobalMeshletVerticesSRVIdx];
    StructuredBuffer<MeshletTriangle> globalMeshletTriangles = ResourceDescriptorHeap[gRasterParams.GlobalMeshletTrianglesSRVIdx];
    StructuredBuffer<float3>          globalPositions        = ResourceDescriptorHeap[gRasterParams.GlobalPositionsSRVIdx];
    StructuredBuffer<uint>            globalNormals          = ResourceDescriptorHeap[gRasterParams.GlobalNormalsSRVIdx];
    StructuredBuffer<uint>            globalUVs              = ResourceDescriptorHeap[gRasterParams.GlobalUVsSRVIdx];

    InstanceData inst     = globalInstanceData[cand.InstanceID];
    MeshData md           = globalMeshData[inst.MeshDataIndex];
    Meshlet m             = globalMeshlets[md.MeshletOffset + cand.MeshletIndex];

    SetMeshOutputCounts(m.VertexCount, m.TriangleCount);

    // Output vertices (strided loop over up to MESHLET_MAX_VERTICES)
    for (uint i = groupThreadID; i < m.VertexCount; i += NUM_MESHLET_THREADS)
    {
        uint globalVtxIdx = globalMeshletVertices[md.MeshletVertexOffset + m.VertexOffset + i];
        float3 localPos   = globalPositions[md.PositionOffset + globalVtxIdx];
        float4 worldPos   = mul(float4(localPos, 1.0), inst.LocalToWorld);
        float4 clipPos    = mul(worldPos, FrameCB.viewProj);
        float3 localNormal = UnpackNormalRGB10A2(globalNormals, md.NormalOffset, globalVtxIdx);
        float3 worldNormal = mul(localNormal, (float3x3)inst.LocalToWorld);

        VertexAttribute v;
        v.Position   = clipPos;
        v.WorldPos   = worldPos.xyz;
        v.Normal     = worldNormal;
        v.UV         = UnpackUVRG16(globalUVs, md.UVOffset, globalVtxIdx);
        v.MaterialID = md.MaterialIndex;
        verts[i] = v;
    }

    // Output primitives (strided loop over up to MESHLET_MAX_TRIANGLES)
    for (uint i = groupThreadID; i < m.TriangleCount; i += NUM_MESHLET_THREADS)
    {
        MeshletTriangle tri = globalMeshletTriangles[md.MeshletTriangleOffset + m.TriangleOffset + i];
        triangles[i] = uint3(tri.V0, tri.V1, tri.V2);

        PrimitiveAttribute pri;
        pri.PrimitiveID    = i;
        pri.CandidateIndex = candidateIndex; // absolute VisibleMeshlets[] index (phase-offset applied above)
        primitives[i] = pri;
    }
}

// --- Pixel Shader ---
// Outputs 4 render targets:
//   SV_Target0: R8G8B8A8_UNORM       — albedo
//   SV_Target1: R16G16B16A16_FLOAT   — packed normal (world-space, [0,1])
//   SV_Target2: R8G8B8A8_UNORM       — roughness | metallic
//   SV_Target3: R32_UINT             — visibility token
// Unconditional alpha discard — correct for both Opaque and Masked.
struct GBufferOutput
{
    float4 albedo   : SV_Target0;
    float4 normal   : SV_Target1;
    float4 material : SV_Target2;
    uint   visToken : SV_Target3;
};

GBufferOutput PSMain(
    VertexAttribute vertexData,
    PrimitiveAttribute primitiveData)
{
    MaterialConstants matConstants = MaterialBuffer[vertexData.MaterialID];

    // --- Albedo ---
    float4 albedo = matConstants.baseColorFactor;
    if (matConstants.baseColorTextureIndex >= 0)
        albedo *= g_Textures[matConstants.baseColorTextureIndex].Sample(g_LinearSampler, vertexData.UV);

    // --- Alpha discard (unconditional, handles Opaque + Mask) ---
    // For opaque (alphaMode==0), alphaCutoff is 0 so this is always a no-op.
    // For masked (alphaMode==1), performs actual alpha-test.
    if (matConstants.alphaMode == 1 && albedo.a < matConstants.alphaCutoff)
        discard;

    // --- Roughness / Metallic ---
    float roughness = matConstants.roughnessFactor;
    float metallic  = matConstants.metallicFactor;
    if (matConstants.metallicRoughnessTextureIndex >= 0)
    {
        float4 mr = g_Textures[matConstants.metallicRoughnessTextureIndex].Sample(g_LinearSampler, vertexData.UV);
        roughness *= mr.g;
        metallic  *= mr.b;
    }

    GBufferOutput output;
    output.albedo   = albedo;
    output.normal   = float4(normalize(vertexData.Normal) * 0.5f + 0.5f, 1.0f);
    output.material = float4(roughness, metallic, 0.0f, 1.0f);
    output.visToken = PackVisBuffer(primitiveData.CandidateIndex, primitiveData.PrimitiveID);
    return output;
}
