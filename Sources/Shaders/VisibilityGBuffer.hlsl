// VisibilityGBuffer.hlsl
//
// Full-screen resolve pass for the meshlet Visibility Buffer pipeline.
// MeshletRasterizeMS.hlsl only rasterizes geometry and writes a compact per-pixel
// visibility token (candidate index + primitive ID). This pass runs once per
// screen pixel, decodes that token, reconstructs the triangle's vertex attributes
// via analytic barycentrics (VisibilityBuffer.hlsli::GetVertexAttributes), samples
// the material, and writes the same GBuffer layout as the legacy Gbuffer.hlsl /
// former MeshletRasterizeMS PSMain (albedo / normal / roughness|metallic).
//
// Mirrors D3D12_Research's VisibilityGBuffer.hlsl (ShadePS).
//
// Sky / background pixels (UnpackVisBuffer fails) are discarded — the GBuffer
// render targets must be cleared to zero before this pass runs.

#include "VisibilityBuffer.hlsli"

// --- Per-pass bindless indices (root constants b1, root param 12) ---
struct VisibilityGBufferParams
{
    uint VisBufSRVIdx;     // Descriptor-heap SRV index of the visibility buffer (R32_UINT)
    uint CandidatesSRVIdx; // Descriptor-heap SRV index of the VisibleMeshlets StructuredBuffer
};
ConstantBuffer<VisibilityGBufferParams> RP : register(b1);

// Per-frame constants
ConstantBuffer<FrameConstants> FrameCB : register(b0);

// Material buffer (root SRV param 1, t0 space1)
StructuredBuffer<MaterialConstants> MaterialBuffer : register(t0, space1);

// Bindless textures (space0)
Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0);

// Meshlet stream buffers (bound via root param 14 descriptor table t0-t8 space3) —
// same layout as MeshletRasterizeMS.hlsl / VisibilityDebugView.hlsl.
StructuredBuffer<float3>          GlobalPositions         : register(t0, space3);
StructuredBuffer<uint>            GlobalNormals           : register(t1, space3);
StructuredBuffer<uint>            GlobalUVs               : register(t2, space3);
StructuredBuffer<Meshlet>         GlobalMeshlets          : register(t3, space3);
StructuredBuffer<uint>            GlobalMeshletVertices   : register(t4, space3);
StructuredBuffer<MeshletTriangle> GlobalMeshletTriangles  : register(t5, space3);
StructuredBuffer<MeshletBounds>   GlobalMeshletBounds     : register(t6, space3); // unused here
StructuredBuffer<MeshData>        GlobalMeshData          : register(t7, space3);
StructuredBuffer<InstanceData>    GlobalInstanceData      : register(t8, space3);

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

// Fullscreen triangle — same trick as FullScreenDebug.hlsl VSMain.
PSInput VSMain(uint vertexID : SV_VertexID)
{
    PSInput output;
    output.texCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.position = float4(output.texCoord * 2.0f - 1.0f, 0.0f, 1.0f);
    output.texCoord.y = 1.0f - output.texCoord.y;
    return output;
}

struct GBufferOutput
{
    float4 albedo   : SV_Target0;
    float4 normal   : SV_Target1;
    float4 material : SV_Target2;
};

GBufferOutput PSMain(PSInput input)
{
    Texture2D<uint> visBufTex = ResourceDescriptorHeap[RP.VisBufSRVIdx];
    uint2 texel = (uint2)input.position.xy;
    uint visData = visBufTex[texel];

    uint candidateIndex, primitiveID;
    if (!UnpackVisBuffer(visData, candidateIndex, primitiveID))
        discard;

    StructuredBuffer<MeshletCandidate> visibleMeshlets = ResourceDescriptorHeap[RP.CandidatesSRVIdx];

    float2 viewportInv = float2(1.0f / (float)FrameCB.internalWidth, 1.0f / (float)FrameCB.internalHeight);
    VisBufferVertexAttribute vertex = GetVertexAttributes(
        input.texCoord, FrameCB.viewProj, viewportInv,
        visibleMeshlets,
        GlobalInstanceData, GlobalMeshData,
        GlobalMeshlets, GlobalMeshletVertices, GlobalMeshletTriangles,
        GlobalPositions, GlobalNormals, GlobalUVs,
        candidateIndex, primitiveID);

    MeshletCandidate candidate = visibleMeshlets[candidateIndex];
    InstanceData instance      = GlobalInstanceData[candidate.InstanceID];
    MeshData meshData          = GlobalMeshData[instance.MeshDataIndex];
    MaterialConstants matConstants = MaterialBuffer[meshData.MaterialIndex];

    // --- Albedo --- (alpha test already resolved during rasterization)
    float4 albedo = matConstants.baseColorFactor;
    if (matConstants.baseColorTextureIndex >= 0)
        albedo *= g_Textures[matConstants.baseColorTextureIndex].Sample(g_LinearSampler, vertex.UV);

    // --- Roughness / Metallic ---
    float roughness = matConstants.roughnessFactor;
    float metallic  = matConstants.metallicFactor;
    if (matConstants.metallicRoughnessTextureIndex >= 0)
    {
        float4 mr = g_Textures[matConstants.metallicRoughnessTextureIndex].Sample(g_LinearSampler, vertex.UV);
        roughness *= mr.g;
        metallic  *= mr.b;
    }

    GBufferOutput output;
    output.albedo   = albedo;
    output.normal   = float4(normalize(vertex.Normal) * 0.5f + 0.5f, 1.0f);
    output.material = float4(roughness, metallic, 0.0f, 1.0f);
    return output;
}
