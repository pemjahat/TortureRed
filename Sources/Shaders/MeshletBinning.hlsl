// =============================================================================
// MeshletBinning.hlsl — BuildDispatchMeshArgsCS
//
// Single-thread compute shader: reads the VisibleMeshletsCounter and writes
// a single D3D12_DISPATCH_MESH_ARGUMENTS entry (3 uints) used by ExecuteIndirect.
//
// Alpha-blended instances are rejected in CullInstancesCS.
// =============================================================================

#include "Shared/SharedTypes.h"

// Root param 12 (b1): RasterParams — contains raw descriptor heap indices
ConstantBuffer<RasterParams> Params : register(b1, space0);

// VisibleMeshletsCounter (SRV) — single uint counter written by culling
StructuredBuffer<uint> VisibleMeshletsCounter : register(t0, space4);

// DispatchMeshArgs (UAV) — D3D12_DISPATCH_MESH_ARGUMENTS (3 uints)
RWStructuredBuffer<uint> DispatchMeshArgs : register(u0, space4);

[numthreads(1, 1, 1)]
void BuildDispatchMeshArgsCS(uint3 tid : SV_DispatchThreadID)
{
    uint srvIdx = Params.VisibleMeshletsCounterIdx;
    uint uavIdx = Params.DispatchMeshArgsIdx;

    // Read the atomic counter written by CullInstancesCS
    StructuredBuffer<uint> counter = ResourceDescriptorHeap[srvIdx];
    uint meshletCount = counter[0];

    // Build indirect DispatchMesh arguments: ThreadGroupCountX = meshletCount
    // Each group processes 1 meshlet; ThreadGroupCountY = ThreadGroupCountZ = 1.
    RWStructuredBuffer<uint> args = ResourceDescriptorHeap[uavIdx];
    args[0] = meshletCount; // ThreadGroupCountX
    args[1] = 1;            // ThreadGroupCountY
    args[2] = 1;            // ThreadGroupCountZ
}
