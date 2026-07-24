#include "MeshletCommon.hlsli"
#include "VisibilityBuffer.hlsli"
#include "Random.hlsli"

// --- Debug overlay parameters (CPU→GPU via root constants b2) ---
struct DebugParams
{
    uint Mode;             // 0 = Off, 1 = Instance, 2 = Meshlet, 3 = Primitive
    uint VisBufSRVIdx;     // Descriptor-heap SRV index for visibility buffer (R32_UINT texture)
    uint CandidatesSRVIdx; // Descriptor-heap SRV index for VisibleMeshlets StructuredBuffer
    uint OutputUAVIdx;     // Descriptor-heap UAV index for output color (R16G16B16A16_FLOAT texture)
    uint Width;
    uint Height;
};
ConstantBuffer<DebugParams> DebugCB : register(b2);

// Global stream buffers for barycentric reconstruction (bound via root param 14 descriptor table, t0-t15 space3)
StructuredBuffer<float3>          GlobalPositions         : register(t0, space3);
StructuredBuffer<uint>            GlobalNormals           : register(t1, space3);
StructuredBuffer<uint>            GlobalUVs               : register(t2, space3);
StructuredBuffer<Meshlet>         GlobalMeshlets          : register(t3, space3);
StructuredBuffer<uint>            GlobalMeshletVertices   : register(t4, space3);
StructuredBuffer<MeshletTriangle> GlobalMeshletTriangles  : register(t5, space3);
StructuredBuffer<MeshData>        GlobalMeshData          : register(t6, space3);
StructuredBuffer<InstanceData>    GlobalInstanceData      : register(t7, space3);

ConstantBuffer<FrameConstants> FrameCB : register(b0);

[numthreads(8, 8, 1)]
void DebugRenderCS(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 texel = dispatchThreadId.xy;
    if (texel.x >= DebugCB.Width || texel.y >= DebugCB.Height)
        return;

    // Access dynamic resources via bindless ResourceDescriptorHeap
    Texture2D<uint>                    visBufTex       = ResourceDescriptorHeap[DebugCB.VisBufSRVIdx];
    StructuredBuffer<MeshletCandidate> visibleMeshlets = ResourceDescriptorHeap[DebugCB.CandidatesSRVIdx];
    RWTexture2D<float4>                outputTex       = ResourceDescriptorHeap[DebugCB.OutputUAVIdx];

    float3 color = float3(0, 0, 0);
    float2 screenUV = float2(
        (texel.x + 0.5f) / float(DebugCB.Width),
        (texel.y + 0.5f) / float(DebugCB.Height));

    uint visData = visBufTex[texel];
    uint candidateIndex, primitiveID;
    if (UnpackVisBuffer(visData, candidateIndex, primitiveID))
    {
        MeshletCandidate candidate = visibleMeshlets[candidateIndex];
        InstanceData instance = GlobalInstanceData[candidate.InstanceID];
        uint meshletIndex = candidate.MeshletIndex;

        // Reconstruct vertex attributes for wireframe
        float2 viewportInv = float2(1.0f / float(DebugCB.Width), 1.0f / float(DebugCB.Height));
        VisBufferVertexAttribute vertex = GetVertexAttributes(
            screenUV, FrameCB.viewProj, viewportInv,
            visibleMeshlets,
            GlobalInstanceData, GlobalMeshData,
            GlobalMeshlets, GlobalMeshletVertices, GlobalMeshletTriangles,
            GlobalPositions, GlobalNormals, GlobalUVs,
            candidateIndex, primitiveID);

        if (DebugCB.Mode == 1)
        {
            uint seed = SeedThread(candidate.InstanceID);
            color = RandomColor(seed);
        }
        else if (DebugCB.Mode == 2)
        {
            uint seed = SeedThread(meshletIndex);
            color = RandomColor(seed);
        }
        else if (DebugCB.Mode == 3)
        {
            uint seed = SeedThread(primitiveID);
            color = RandomColor(seed);
        }

        // Wireframe overlay on all debug modes
        color *= saturate(Wireframe(vertex.Barycentrics) + 0.8f);
    }

    outputTex[texel] = float4(color, 1.0f);
}
