#include "MeshletCommon.hlsli"

// Input: all instances × all meshlets → cull against frustum
// Output: visible MeshletCandidate list

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// 7 global stream buffers
StructuredBuffer<MeshletBounds>  GlobalMeshletBounds : register(t0, space3);

// Per-instance and per-mesh data
StructuredBuffer<MeshData>       GlobalMeshData      : register(t1, space3);
StructuredBuffer<InstanceData>   GlobalInstanceData   : register(t2, space3);

// Cull dispatch constants (passed via constant buffer or root constant)
struct CullConstants {
    uint totalMeshlets;   // sum of all MeshletCount across all instances
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
ConstantBuffer<CullConstants> CullCB : register(b1);

// Output
RWStructuredBuffer<MeshletCandidate> VisibleMeshlets        : register(u0);
RWStructuredBuffer<uint>             VisibleMeshletsCounter : register(u1);

// Build a flat mapping: globalThreadID → (instance, meshlet)
// We walk through InstanceData sequentially to find which instance owns this global meshlet index.
// This is O(N) per thread but simple and acceptable for small instance counts.

void FindInstanceAndLocalMeshlet(uint globalMeshletIdx, out uint instanceID, out uint localMeshletIdx)
{
    // We don't know the instance count at compile time, but we can walk
    // through InstanceData. For efficiency, we use the flat array.
    // A better approach: store cumulative offsets, but for simplicity we do linear scan.
    // Actually, we don't have the instance count available. Let's use a simple approach:
    // The dispatch is over totalMeshlets threads, one per (instance, local meshlet).
    // We need a way to map from thread ID to (instanceID, localMeshlet).
    // We'll skip for now and use a pre-built mapping.
    // For now, just use global index directly:
    instanceID = globalMeshletIdx;  // placeholder
    localMeshletIdx = 0;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint globalIdx = dispatchThreadID.x;
    if (globalIdx >= CullCB.totalMeshlets)
        return;

    // Linear scan through instances to find the owner of this global meshlet
    // We don't have instance count, so we use a max bound.
    // This is a simplistic implementation - should be replaced with pre-built offset array.
    uint accumulatedMeshlets = 0;
    uint instanceID = 0;

    // Walk through instances (max 1024 for safety)
    for (uint i = 0; i < 1024; i++)
    {
        MeshData md = GlobalMeshData[i];
        if (md.MeshletCount == 0)
            continue;

        if (globalIdx < accumulatedMeshlets + md.MeshletCount)
        {
            instanceID = i;
            break;
        }
        accumulatedMeshlets += md.MeshletCount;
    }

    // Guard: if instanceID wasn't found, bail
    if (instanceID >= 1024)
        return;

    uint localMeshletIdx = globalIdx - accumulatedMeshlets;
    MeshData md = GlobalMeshData[instanceID];

    // Load meshlet bounds
    MeshletBounds bounds = GlobalMeshletBounds[md.MeshletBoundsOffset + localMeshletIdx];

    // Load instance transform
    InstanceData inst = GlobalInstanceData[instanceID];

    // Frustum cull
    if (FrustumCullMeshlet(bounds, inst.LocalToWorld, FrameCB.viewProj))
    {
        uint slot;
        InterlockedAdd(VisibleMeshletsCounter[0], 1, slot);

        MeshletCandidate cand;
        cand.InstanceID   = instanceID;
        cand.MeshletIndex = localMeshletIdx;
        VisibleMeshlets[slot] = cand;
    }
}