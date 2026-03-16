// SHaRC_Resolve.hlsl
// No special defines needed (not a SHARC_UPDATE shader).
//
// One thread per hash entry.  Calls SharcResolveEntry which:
//   1. Blends this frame's accumulated samples into the resolved buffer (EMA)
//   2. Evicts stale entries (no samples for staleFrameNumMax frames)
//   3. Clears the accumulation slot so SHaRC_Update starts fresh next frame
//
// Dispatch: (SHARC_HASH_ENTRIES_NUM + 255) / 256 groups of 256 threads.

#include "SharcCommon.h"
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>       g_Frame  : register(b0);
ConstantBuffer<SharcBindlessIndices> g_Sharc  : register(b2);

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= SHARC_HASH_ENTRIES_NUM) return;

    SharcParameters sharcParams;
    sharcParams.gridParameters.cameraPosition  = g_Frame.cameraPosition.xyz;
    sharcParams.gridParameters.logarithmBase   = SHARC_GRID_LOGARITHM_BASE;
    sharcParams.gridParameters.sceneScale      = g_Frame.sharcSceneScale;
    sharcParams.gridParameters.levelBias       = 0.0f;
    sharcParams.hashMapData.capacity           = SHARC_HASH_ENTRIES_NUM;
    sharcParams.hashMapData.hashEntriesBuffer  = ResourceDescriptorHeap[g_Sharc.HashEntriesBufIdx];
    sharcParams.accumulationBuffer             = ResourceDescriptorHeap[g_Sharc.AccumulationBufIdx];
    sharcParams.resolvedBuffer                 = ResourceDescriptorHeap[g_Sharc.ResolvedBufIdx];
    sharcParams.radianceScale                  = 1e3f;
    sharcParams.enableAntiFireflyFilter        = false;

    SharcResolveParameters resolveParams;
    resolveParams.cameraPositionPrev   = g_Frame.prevCameraPosition.xyz;
    resolveParams.accumulationFrameNum = g_Frame.sharcAccumulationFrameNum;
    resolveParams.staleFrameNumMax     = g_Frame.sharcStaleFrameNum;
    resolveParams.frameIndex           = g_Frame.frameIndex;

    SharcResolveEntry(DTid.x, sharcParams, resolveParams);
}
