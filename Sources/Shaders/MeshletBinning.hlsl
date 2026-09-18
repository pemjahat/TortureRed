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

// VisibleMeshletsCounter (SRV) and DispatchMeshArgs (UAV) are fully bindless:
// fetched via ResourceDescriptorHeap[Params.*Idx] at the point of use below,
// so no static register declarations are needed.

[numthreads(1, 1, 1)]
void BuildDispatchMeshArgsCS(uint3 tid : SV_DispatchThreadID)
{
    uint srvIdx = Params.VisibleMeshletsCounterIdx;
    uint uavIdx = Params.DispatchMeshArgsIdx;

    // VisibleMeshletsCounter now has 2 slots (see TWO_PASS_PHASE_FIRST/SECOND in
    // SharedTypes.h) — readONLY this phase's own slot, not a running total, so this
    // phase's DispatchMesh only covers its own new meshlets (Phase 2 must not re-rasterize
    // Phase 1's already-drawn range). Params.Phase doubles as the slot index.
    StructuredBuffer<uint> counter = ResourceDescriptorHeap[srvIdx];
    uint meshletCount = counter[Params.Phase];

    // Build indirect DispatchMesh arguments: ThreadGroupCountX = meshletCount
    // Each group processes 1 meshlet; ThreadGroupCountY = ThreadGroupCountZ = 1.
    RWStructuredBuffer<uint> args = ResourceDescriptorHeap[uavIdx];
    args[0] = meshletCount; // ThreadGroupCountX
    args[1] = 1;            // ThreadGroupCountY
    args[2] = 1;            // ThreadGroupCountZ
}
