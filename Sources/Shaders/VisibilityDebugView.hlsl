#include "MeshletCommon.hlsli"
#include "VisibilityBuffer.hlsli"
#include "Random.hlsli"

// --- Debug overlay parameters (CPU→GPU via root constants b2) ---
// MeshletDebugParams is shared with the CPU side (SharedTypes.h) and carries the
// meshlet stream bindless indices.
ConstantBuffer<MeshletDebugParams> DebugCB : register(b2);

// Heat ramp for the mip tint : fine mips (0) = blue → green → coarse mips = red
float3 MipTintColor(uint mip, uint mipCount)
{
    float t = (mipCount > 1) ? (float)mip / (float)(mipCount - 1) : 0.0f;
    return float3(saturate(t * 2.0f - 0.5f),
                  saturate(1.0f - abs(t * 2.0f - 1.0f)),
                  saturate(1.5f - t * 2.0f));
}

// Meshlet stream buffers are looked up individually via ResourceDescriptorHeap
// from DebugCB's flat bindless index fields.

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
        //InstanceData instance = globalInstanceData[candidate.InstanceID];
        uint meshletIndex = candidate.MeshletIndex;

        // Meshlet stream buffers — bindless heap lookups
        StructuredBuffer<InstanceData>    globalInstanceData     = ResourceDescriptorHeap[DebugCB.InstanceDataSRVIdx];
        StructuredBuffer<MeshData>        globalMeshData         = ResourceDescriptorHeap[DebugCB.MeshDataSRVIdx];
        StructuredBuffer<Meshlet>         globalMeshlets         = ResourceDescriptorHeap[DebugCB.GlobalMeshletsSRVIdx];
        StructuredBuffer<uint>            globalMeshletVertices  = ResourceDescriptorHeap[DebugCB.GlobalMeshletVerticesSRVIdx];
        StructuredBuffer<MeshletTriangle> globalMeshletTriangles = ResourceDescriptorHeap[DebugCB.GlobalMeshletTrianglesSRVIdx];
        StructuredBuffer<float3>          globalPositions        = ResourceDescriptorHeap[DebugCB.GlobalPositionsSRVIdx];
        StructuredBuffer<uint>            globalNormals          = ResourceDescriptorHeap[DebugCB.GlobalNormalsSRVIdx];
        StructuredBuffer<uint>            globalUVs              = ResourceDescriptorHeap[DebugCB.GlobalUVsSRVIdx];

        // Reconstruct vertex attributes for wireframe
         float2 viewportInv = float2(1.0f / float(DebugCB.Width), 1.0f / float(DebugCB.Height));
         VisBufferVertexAttribute vertex = GetVertexAttributes(
             screenUV, FrameCB.viewProj, viewportInv,
             visibleMeshlets,
             globalInstanceData, globalMeshData,
             globalMeshlets, globalMeshletVertices, globalMeshletTriangles,
             globalPositions, globalNormals, globalUVs,
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
        else if (DebugCB.Mode == MESHLET_DEBUG_MIP_TINT) // 4 — HZB mip-selection tint (task007 mode 3)
        {
            StructuredBuffer<uint> mips = ResourceDescriptorHeap[DebugCB.MipsSRVIdx];
            uint mip = mips[candidateIndex];
            // 0xFF sentinel: no occlusion test ran for this meshlet (frustum-only / near-plane fallback)
            color = (mip == 0xFFu) ? float3(0.15f, 0.15f, 0.15f) : MipTintColor(mip, DebugCB.HZBMipCount);
        }

        // Wireframe overlay on all debug modes
        color *= saturate(Wireframe(vertex.Barycentrics) + 0.8f);
    }

    outputTex[texel] = float4(color, 1.0f);
}
