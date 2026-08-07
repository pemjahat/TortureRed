#include "MeshletCommon.hlsli"
#include "VisibilityBuffer.hlsli"

/*
    Mesh Shader + Pixel Shader for GPU-driven meshlet rasterization — Visibility Buffer variant.

    Proper Visibility Buffer split (mirrors D3D12_Research's MeshletRasterize.hlsl):
    this pass ONLY rasterizes geometry and writes the compact per-pixel visibility
    token (candidate index + primitive ID) plus depth. No material sampling and no
    GBuffer output happens here — that work is deferred to the full-screen
    VisibilityGBuffer.hlsl resolve pass, which runs once per screen pixel instead of
    once per rasterized fragment.

    MSMain: one thread group per visible meshlet (SV_GroupID → MeshletCandidate directly,
            no bin indirection). 32 threads process up to MESHLET_MAX_VERTICES vertices
            and MESHLET_MAX_TRIANGLES triangles.
    PSMain: writes ONLY the visibility token (R32_UINT) to SV_Target0.
            UNCONDITIONAL alpha discard: for alpha-masked materials, the base color
            texture is sampled to alpha-test before the token is written (so invisible
            fragments never occlude geometry behind them). For opaque materials,
            the discard test is a no-op (alphaCutoff = 0, alpha >= 0 always passes).
            Alpha-blended instances are already rejected in CullInstancesCS and never
            reach this shader.

    No ALPHA_MASK permutation — single combined PSO handles both opaque and alpha-masked
    meshlets (Adria-style no-binning design).
*/

#define NUM_MESHLET_THREADS 32

// --- Bindless resource declarations ---
// Meshlet stream buffers (contiguous in heap, bound via root param 14 descriptor table t0-t8 space3)
StructuredBuffer<float3>           GlobalPositions         : register(t0, space3);
StructuredBuffer<uint>             GlobalNormals           : register(t1, space3); // unused here (moved to VisibilityGBuffer.hlsl)
StructuredBuffer<uint>             GlobalUVs               : register(t2, space3);
StructuredBuffer<Meshlet>          GlobalMeshlets          : register(t3, space3);
StructuredBuffer<uint>             GlobalMeshletVertices   : register(t4, space3);
StructuredBuffer<MeshletTriangle>  GlobalMeshletTriangles  : register(t5, space3);
StructuredBuffer<MeshletBounds>    GlobalMeshletBounds     : register(t6, space3); // unused here
StructuredBuffer<MeshData>         GlobalMeshData          : register(t7, space3);
StructuredBuffer<InstanceData>     GlobalInstanceData      : register(t8, space3);

// Material buffer (root SRV param 1, t0 space1) — always needed for alpha discard
StructuredBuffer<MaterialConstants> MaterialBuffer : register(t0, space1);

// Bindless textures (space0) — always needed for alpha discard
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

// --- Per-vertex output ---
// Always includes UV + MaterialID for unconditional alpha discard (no ALPHA_MASK permutation).
struct VertexAttribute
{
    float4 Position : SV_Position;
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
    // is Phase 1's final count) since the counter is only cleared once per frame 
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
        v.Position   = clipPos;
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
        pri.CandidateIndex = candidateIndex; // absolute VisibleMeshlets[] index (phase-offset applied above)
        primitives[i] = pri;
    }
}

// --- Pixel Shader ---
// Outputs a single render target:
//   SV_Target0: R32_UINT — visibility token (candidate index + primitive ID)
// Unconditional alpha discard (no ALPHA_MASK permutation):
//   - alphaMode == ALPHA_MODE_MASK (1): discard if alpha < cutoff
//   - alphaMode == ALPHA_MODE_OPAQUE (0): alphaCutoff = 0, never discards
//   - alphaMode == ALPHA_MODE_BLEND (2): rejected by culling, never reaches here
struct VisOutput
{
    uint visToken : SV_Target0;
};

VisOutput PSMain(
    VertexAttribute vertexData,
    PrimitiveAttribute primitiveData)
{
    MaterialConstants matConstants = MaterialBuffer[vertexData.MaterialID];

    // Alpha discard — unconditional, correct for both Opaque and Alpha-Masked materials
    // For opaque (alphaMode==0), alphaCutoff is 0 so this is always a no-op.
    // For masked (alphaMode==1), performs actual alpha-test.
    // For blend (alphaMode==2), rejected by culling — never reaches this shader.
    float4 albedo = matConstants.baseColorFactor;
    if (matConstants.baseColorTextureIndex >= 0)
        albedo *= g_Textures[matConstants.baseColorTextureIndex].Sample(g_LinearSampler, vertexData.UV);
    if (matConstants.alphaMode == 1 && albedo.a < matConstants.alphaCutoff)
        discard;

    VisOutput output;
    output.visToken = PackVisBuffer(primitiveData.CandidateIndex, primitiveData.PrimitiveID);
    return output;
}
