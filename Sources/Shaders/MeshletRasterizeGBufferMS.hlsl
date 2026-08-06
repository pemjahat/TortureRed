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
// Meshlet stream buffers (contiguous in heap, bound via root param 14 descriptor table t0-t8 space3)
StructuredBuffer<float3>           GlobalPositions         : register(t0, space3);
StructuredBuffer<uint>             GlobalNormals           : register(t1, space3);
StructuredBuffer<uint>             GlobalUVs               : register(t2, space3);
StructuredBuffer<Meshlet>          GlobalMeshlets          : register(t3, space3);
StructuredBuffer<uint>             GlobalMeshletVertices   : register(t4, space3);
StructuredBuffer<MeshletTriangle>  GlobalMeshletTriangles  : register(t5, space3);
StructuredBuffer<MeshletBounds>    GlobalMeshletBounds     : register(t6, space3); // unused here
StructuredBuffer<MeshData>         GlobalMeshData          : register(t7, space3);
StructuredBuffer<InstanceData>     GlobalInstanceData      : register(t8, space3);

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
    // Direct indexing: no bins — VisibleMeshlets[groupID] is the meshlet candidate
    StructuredBuffer<MeshletCandidate> visibleMeshlets = ResourceDescriptorHeap[gRasterParams.VisibleMeshletsIdx];
    MeshletCandidate cand = visibleMeshlets[groupID];

    InstanceData inst     = GlobalInstanceData[cand.InstanceID];
    MeshData md           = GlobalMeshData[inst.MeshDataIndex];
    Meshlet m             = GlobalMeshlets[md.MeshletOffset + cand.MeshletIndex];

    SetMeshOutputCounts(m.VertexCount, m.TriangleCount);

    // Output vertices (strided loop over up to MESHLET_MAX_VERTICES)
    for (uint i = groupThreadID; i < m.VertexCount; i += NUM_MESHLET_THREADS)
    {
        uint globalVtxIdx = GlobalMeshletVertices[md.MeshletVertexOffset + m.VertexOffset + i];
        float3 localPos   = GlobalPositions[md.PositionOffset + globalVtxIdx];
        float4 worldPos   = mul(float4(localPos, 1.0), inst.LocalToWorld);
        float4 clipPos    = mul(worldPos, FrameCB.viewProj);
        float3 localNormal = UnpackNormalRGB10A2(GlobalNormals, md.NormalOffset, globalVtxIdx);
        float3 worldNormal = mul(localNormal, (float3x3)inst.LocalToWorld);

        VertexAttribute v;
        v.Position   = clipPos;
        v.WorldPos   = worldPos.xyz;
        v.Normal     = worldNormal;
        v.UV         = UnpackUVRG16(GlobalUVs, md.UVOffset, globalVtxIdx);
        v.MaterialID = md.MaterialIndex;
        verts[i] = v;
    }

    // Output primitives (strided loop over up to MESHLET_MAX_TRIANGLES)
    for (uint i = groupThreadID; i < m.TriangleCount; i += NUM_MESHLET_THREADS)
    {
        MeshletTriangle tri = GlobalMeshletTriangles[md.MeshletTriangleOffset + m.TriangleOffset + i];
        triangles[i] = uint3(tri.V0, tri.V1, tri.V2);

        PrimitiveAttribute pri;
        pri.PrimitiveID    = i;
        pri.CandidateIndex = groupID;
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
    if (albedo.a < matConstants.alphaCutoff)
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
