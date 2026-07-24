#include "MeshletCommon.hlsli"
#include "VisibilityBuffer.hlsli"

/*
    Mesh Shader + Pixel Shader for GPU-driven meshlet rasterization.
    Replaces the old VS+PS path (MeshletRasterize.hlsl).

    MSMain: one thread group per visible meshlet (SV_GroupID → bin indirection → MeshletCandidate).
            32 threads process up to MESHLET_MAX_VERTICES vertices and MESHLET_MAX_TRIANGLES triangles.
    PSMain: writes visibility buffer token (candidateIndex + primitiveID) to SV_TARGET1.
            SV_TARGET0 is unused (cleared to 0) — full shading is deferred to a lighting pass
            that reads the visibility buffer.

    Compile permutations:
        ALPHA_MASK=0  — Opaque bin (back-face cull)
        ALPHA_MASK=1  — AlphaMasked bin (no cull, alpha discard in PS)
        ENABLE_DEBUG_DATA=1 — writes extra debug info (used by VisibilityDebugView.hlsl)
*/

#ifndef ALPHA_MASK
#define ALPHA_MASK 0
#endif

#ifndef ENABLE_DEBUG_DATA
#define ENABLE_DEBUG_DATA 0
#endif

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

// Per-bin raster params (root constants b1)
ConstantBuffer<RasterParams> gRasterParams : register(b1);

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
#if ALPHA_MASK
    float2 UV       : TEXCOORD;
    nointerpolation uint MaterialID : MATERIAL_ID;
#endif
};

// --- Mesh Shader Entry Point ---
[outputtopology("triangle")]
[numthreads(NUM_MESHLET_THREADS, 1, 1)]
void MSMain(
    in  uint groupThreadID : SV_GroupIndex,
    in  uint groupID       : SV_GroupID,
    out vertices  VertexAttribute  verts[MESHLET_MAX_VERTICES],
    out indices   uint3            triangles[MESHLET_MAX_TRIANGLES],
    out primitives PrimitiveAttribute primitives[MESHLET_MAX_TRIANGLES])
{
    // Resolve meshlet index via bin indirection:
    //   BinnedMeshlets[groupID + binOffset] → index into VisibleMeshlets[]
    StructuredBuffer<uint4>            binData        = ResourceDescriptorHeap[gRasterParams.MeshletBinDataIdx];
    StructuredBuffer<uint>             binnedMeshlets = ResourceDescriptorHeap[gRasterParams.BinnedMeshletsIdx];
    StructuredBuffer<MeshletCandidate> visibleMeshlets = ResourceDescriptorHeap[gRasterParams.VisibleMeshletsIdx];

    uint binOffset    = binData[gRasterParams.BinIndex].w;
    uint meshletIndex = binnedMeshlets[binOffset + groupID];

    MeshletCandidate cand = visibleMeshlets[meshletIndex];
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

        VertexAttribute v;
        v.Position = clipPos;
#if ALPHA_MASK
        v.UV         = UnpackUVRG16(GlobalUVs, md.UVOffset, globalVtxIdx);
        v.MaterialID = md.MaterialIndex;
#endif
        verts[i] = v;
    }

    // Output primitives (strided loop over up to MESHLET_MAX_TRIANGLES)
    for (uint i = groupThreadID; i < m.TriangleCount; i += NUM_MESHLET_THREADS)
    {
        MeshletTriangle tri = GlobalMeshletTriangles[md.MeshletTriangleOffset + m.TriangleOffset + i];
        triangles[i] = uint3(tri.V0, tri.V1, tri.V2);

        PrimitiveAttribute pri;
        pri.PrimitiveID    = i;
        pri.CandidateIndex = meshletIndex;
        primitives[i] = pri;
    }
}

// --- Pixel Shader ---
// Outputs:
//   SV_TARGET0: R16G16B16A16_FLOAT — cleared to 0 (full shading deferred to lighting pass)
//   SV_TARGET1: R32_UINT           — visibility buffer token (candidateIndex + primitiveID)
struct PSOutput
{
    float4 color     : SV_TARGET0;
    uint   visBuffer : SV_TARGET1;
};

PSOutput PSMain(
    VertexAttribute vertexData,
    PrimitiveAttribute primitiveData)
{
#if ALPHA_MASK
    // Alpha discard for alpha-masked materials
    MaterialConstants material = MaterialBuffer[vertexData.MaterialID];
    float alpha = material.baseColorFactor.a;
    if (material.baseColorTextureIndex >= 0)
        alpha *= g_Textures[material.baseColorTextureIndex].Sample(g_LinearSampler, vertexData.UV).a;
    if (alpha < material.alphaCutoff)
        discard;
#endif

    PSOutput output;
    output.color     = float4(0, 0, 0, 0); // Deferred shading — lighting pass reads visibility buffer
    output.visBuffer = PackVisBuffer(primitiveData.CandidateIndex, primitiveData.PrimitiveID);
    return output;
}
