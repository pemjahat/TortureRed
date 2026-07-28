#include "MeshletCommon.hlsli"
#include "VisibilityBuffer.hlsli"

/*
    Debug Mesh Shader — CPU-driven per-meshlet dispatch (no GPU culling / binning).

    The CPU iterates every meshlet and calls DispatchMesh(1, 1, 1) once per meshlet,
    writing InstanceID + MeshletIndex directly into DebugRasterParams (root param 12, b1).
    This bypasses VisibleMeshlets / BinnedMeshlets entirely, making it easy to isolate
    and verify individual meshlet data without the full GPU-driven pipeline.

    MSMain: SV_GroupID is always 0 — InstanceID and MeshletIndex come from root constants.
    PSMain: identical GBuffer output to MeshletRasterizeMS.hlsl.
*/

#ifndef ALPHA_MASK
#define ALPHA_MASK 0
#endif

#define NUM_MESHLET_THREADS 32

// --- Bindless resource declarations (same slots as MeshletRasterizeMS.hlsl) ---
StructuredBuffer<float3>           GlobalPositions         : register(t0, space3);
StructuredBuffer<uint>             GlobalNormals           : register(t1, space3);
StructuredBuffer<uint>             GlobalUVs               : register(t2, space3);
StructuredBuffer<Meshlet>          GlobalMeshlets          : register(t3, space3);
StructuredBuffer<uint>             GlobalMeshletVertices   : register(t4, space3);
StructuredBuffer<MeshletTriangle>  GlobalMeshletTriangles  : register(t5, space3);
StructuredBuffer<MeshletBounds>    GlobalMeshletBounds     : register(t6, space3);
StructuredBuffer<MeshData>         GlobalMeshData          : register(t7, space3);
StructuredBuffer<InstanceData>     GlobalInstanceData      : register(t8, space3);

// Material buffer (root SRV param 1, t0 space1)
StructuredBuffer<MaterialConstants> MaterialBuffer : register(t0, space1);

// Bindless textures (space0)
Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0);

// Per-frame constants
ConstantBuffer<FrameConstants> FrameCB : register(b0);

// Per-dispatch debug params: InstanceID + MeshletIndex set by CPU per DispatchMesh call
ConstantBuffer<DebugRasterParams> gDebugParams : register(b1);

// --- Per-primitive output ---
struct PrimitiveAttribute
{
    uint PrimitiveID    : SV_PrimitiveID;
    uint CandidateIndex : CANDIDATE_INDEX;
};

// --- Per-vertex output ---
struct VertexAttribute
{
    float4 Position : SV_Position;
    float3 WorldPos : WORLD_POS;
    float3 Normal   : NORMAL;
    float2 UV       : TEXCOORD;
    nointerpolation uint MaterialID : MATERIAL_ID;
};

// --- Mesh Shader Entry Point ---
[outputtopology("triangle")]
[numthreads(NUM_MESHLET_THREADS, 1, 1)]
void MSMain(
    in  uint groupThreadID : SV_GroupIndex,
    in  uint groupID       : SV_GroupID,   // always 0 — one group per DispatchMesh(1,1,1)
    out vertices  VertexAttribute  verts[MESHLET_MAX_VERTICES],
    out indices   uint3            triangles[MESHLET_MAX_TRIANGLES],
    out primitives PrimitiveAttribute primitives[MESHLET_MAX_TRIANGLES])
{
    // Resolve directly from root constants — no binning indirection
    InstanceData inst = GlobalInstanceData[gDebugParams.InstanceID];
    MeshData md       = GlobalMeshData[inst.MeshDataIndex];
    Meshlet m         = GlobalMeshlets[md.MeshletOffset + gDebugParams.MeshletIndex];

    SetMeshOutputCounts(m.VertexCount, m.TriangleCount);

    // Output vertices
    for (uint i = groupThreadID; i < m.VertexCount; i += NUM_MESHLET_THREADS)
    {
        uint globalVtxIdx  = GlobalMeshletVertices[md.MeshletVertexOffset + m.VertexOffset + i];
        float3 localPos    = GlobalPositions[md.PositionOffset + globalVtxIdx];
        float4 worldPos    = mul(float4(localPos, 1.0), inst.LocalToWorld);
        float4 clipPos     = mul(worldPos, FrameCB.viewProj);
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

    // Output primitives
    for (uint i = groupThreadID; i < m.TriangleCount; i += NUM_MESHLET_THREADS)
    {
        MeshletTriangle tri = GlobalMeshletTriangles[md.MeshletTriangleOffset + m.TriangleOffset + i];
        triangles[i] = uint3(tri.V0, tri.V1, tri.V2);

        PrimitiveAttribute pri;
        pri.PrimitiveID    = i;
        pri.CandidateIndex = gDebugParams.InstanceID * 10000u + gDebugParams.MeshletIndex; // debug token
        primitives[i] = pri;
    }
}

// --- Pixel Shader (identical GBuffer output to MeshletRasterizeMS.hlsl) ---
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

    float4 albedo = matConstants.baseColorFactor;
    if (matConstants.baseColorTextureIndex >= 0)
        albedo *= g_Textures[matConstants.baseColorTextureIndex].Sample(g_LinearSampler, vertexData.UV);

#if ALPHA_MASK
    if (matConstants.alphaMode == 1 && albedo.a < matConstants.alphaCutoff)
        discard;
#endif

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
