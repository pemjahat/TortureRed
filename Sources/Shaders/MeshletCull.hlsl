#include "MeshletCommon.hlsli"

// Input: all instances × all meshlets → cull against frustum
// Output: visible MeshletCandidate list

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// 7 global stream buffers
StructuredBuffer<MeshletBounds>  GlobalMeshletBounds : register(t0, space3);

// Per-instance and per-mesh data
StructuredBuffer<MeshData>       GlobalMeshData      : register(t1, space3);
StructuredBuffer<InstanceData>   GlobalInstanceData   : register(t2, space3);

struct CullConstants {
    uint totalMeshlets;   // sum of all MeshletCount across all instances
    uint instanceCount;   // number of entries in GlobalMeshData[] / GlobalInstanceData[]
    uint _pad0;
    uint _pad1;
};
ConstantBuffer<CullConstants> CullCB : register(b1);

// Output
RWStructuredBuffer<MeshletCandidate>      VisibleMeshlets        : register(u0);
RWStructuredBuffer<uint>                  VisibleMeshletsCounter : register(u1);
// DEBUG: per-visible-meshlet vertex/triangle counts, written at the same slot as VisibleMeshlets
RWStructuredBuffer<MeshletCandidateDebug> VisibleMeshletsDebug   : register(u2);

// Also need GlobalMeshlets to read VertexCount/TriangleCount
StructuredBuffer<Meshlet>                 GlobalMeshlets         : register(t3, space3);

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint globalIdx = dispatchThreadID.x;
    if (globalIdx >= CullCB.totalMeshlets)
        return;

    // Map globalIdx → (instanceID, localMeshletIdx) by walking instances in order.
    // Loop is bounded by instanceCount (passed from CPU), never reads out-of-bounds.
    uint instanceID = 0;

    for (uint i = 0; i < CullCB.instanceCount; i++)
    {        
        MeshData md = GlobalMeshData[i];        
        if (globalIdx >= md.GlobalMeshletStart &&
            globalIdx < md.GlobalMeshletStart + md.MeshletCount)
        {
            instanceID = i;
            break;
        }
    }

    MeshData md = GlobalMeshData[instanceID];
    uint localMeshletIdx = globalIdx - md.GlobalMeshletStart;

    // --- Debug isolation: only process meshlets from one primitive.
    //     Define DEBUG_SINGLE_PRIMITIVE=0 to isolate the 1st primitive,
    //     =1 for the 2nd, etc. Remove the define to process all normally.
#ifdef DEBUG_SINGLE_PRIMITIVE
    if (instanceID != DEBUG_SINGLE_PRIMITIVE)
        return;
#endif

    // Load meshlet bounds and instance transform
    MeshletBounds bounds = GlobalMeshletBounds[md.MeshletBoundsOffset + localMeshletIdx];
    InstanceData inst    = GlobalInstanceData[instanceID];

    // Frustum cull — skip meshlets fully outside the view frustum
    if (!FrustumCullMeshlet(bounds, inst.LocalToWorld, FrameCB.viewProj))
        return;

    uint slot;
    InterlockedAdd(VisibleMeshletsCounter[0], 1, slot);

    MeshletCandidate cand;
    cand.InstanceID   = instanceID;
    cand.MeshletIndex = localMeshletIdx;
    VisibleMeshlets[slot] = cand;

    // DEBUG: write per-meshlet vertex/triangle counts at the same slot
    Meshlet m = GlobalMeshlets[md.MeshletOffset + localMeshletIdx];
    MeshletCandidateDebug dbg;
    dbg.VertexCount   = m.VertexCount;
    dbg.TriangleCount = m.TriangleCount;
    VisibleMeshletsDebug[slot] = dbg;
}