// CullStats.hlsl
// GPU on-screen debugging: meshlet culling statistics overlay.
//
// Two entry points:
//   CopyCullStatsCS   — 1-thread CS. After each culling phase, copies the current
//                        per-phase functional counters (CandidateMeshletsCounter,
//                        VisibleMeshletsCounter, OccludedInstancesCounter) into
//                        the dedicated CullStatsBuffer at the phase-appropriate slot.
//                        Dispatched by GPUCulling::CullTwoPass internally.
//   CullStatsCS       — 1-thread CS. Reads CullStatsBuffer, computes derived stats,
//                        and renders a formatted table via the GPU debug-text system
//                        (DebugTextRender.hlsli). Dispatched by Renderer after both
//                        culling phases complete.

#include "Shared/SharedTypes.h"
#include "DebugTextRender.hlsli"

// =============================================================================
// CopyCullStatsCS
// Copies per-phase functional counter values to the CullStatsBuffer.
//   BaseSlot=0 → P1: write Candidate at [0], Visible at [1], Occluded at [2]
//   BaseSlot=4 → P2: write Candidate at [4], Visible at [5], Occluded at [6],
//                     and [7] = Occluded counter element [0] (P2 input instances)
// =============================================================================

ConstantBuffer<CullStatsCopyParams> CopyParams : register(b2);

[numthreads(1, 1, 1)]
void CopyCullStatsCS(uint3 tid : SV_DispatchThreadID)
{
    StructuredBuffer<uint> candidateCounter = ResourceDescriptorHeap[CopyParams.CandidateCounterSRVIdx];
    StructuredBuffer<uint> visibleCounter   = ResourceDescriptorHeap[CopyParams.VisibleCounterSRVIdx];
    StructuredBuffer<uint> occludedCounter  = ResourceDescriptorHeap[CopyParams.OccludedCounterSRVIdx];
    RWStructuredBuffer<uint> stats          = ResourceDescriptorHeap[CopyParams.StatsBufferUAVIdx];

    uint base = CopyParams.BaseSlot;

    stats[base + 0] = candidateCounter[0];          // P1_CANDIDATE_MESHLETS or P2_CANDIDATE_MESHLETS
    stats[base + 1] = visibleCounter[0];            // P1_VISIBLE_MESHLETS   or P2_VISIBLE_MESHLETS
    stats[base + 2] = occludedCounter[0];            // P1_OCCLUDED_INSTANCES or P2_OCCLUDED_INSTANCES

    if (base == 0) // (Phase 1 always runs first each frame)
    {
        // Pre-zero the P2 slots so CullStatsCS never displays stale numbers left
        // over from an earlier frame — e.g. when two-pass culling is disabled
        // this frame, Phase 2's copy never runs at all. If Phase 2 DOES run
        // later this frame, its own copy (base==4, above) overwrites these.
        stats[CULL_STATS_P2_CANDIDATE_MESHLETS] = 0;
        stats[CULL_STATS_P2_VISIBLE_MESHLETS]   = 0;
        stats[CULL_STATS_P2_OCCLUDED_INSTANCES] = 0;
    }
}

// =============================================================================
// CullStatsCS
// Reads CullStatsBuffer, computes derived stats, and renders an on-screen table
// via the GPU debug-text system (DebugTextRender.hlsli). The output is a set of
// formatted text lines anchored at (StartX, StartY).
//
// Table:
//   Total Instances:  NNN
//   Total Meshlets:   NNN
//   ────────────────────────
//          Instances  Meshlets
//   Phase1 Occluded: NNN / NNN
//   Phase2 Occluded: NNN / NNN
//   ────────────────────────
//   Visible:          NNN / NNN
// =============================================================================

ConstantBuffer<CullStatsParams> Params : register(b2);

[numthreads(1, 1, 1)]
void CullStatsCS(uint3 tid : SV_DispatchThreadID)
{
    StructuredBuffer<uint> stats = ResourceDescriptorHeap[Params.StatsBufferSRVIdx];

    uint p1Candidate   = stats[CULL_STATS_P1_CANDIDATE_MESHLETS];
    uint p1Visible     = stats[CULL_STATS_P1_VISIBLE_MESHLETS];
    uint p1OccludedInst = stats[CULL_STATS_P1_OCCLUDED_INSTANCES];
    uint p2Candidate   = stats[CULL_STATS_P2_CANDIDATE_MESHLETS];
    uint p2Visible     = stats[CULL_STATS_P2_VISIBLE_MESHLETS];
    uint p2OccludedInst = stats[CULL_STATS_P2_OCCLUDED_INSTANCES];  // include p1 occluded instances
    uint totalMeshlets = Params.TotalMeshlets;
    uint totalInst     = Params.TotalInstances;

    // Derived stats
    uint p1OccludedMeshlet = (p1Candidate - p1Visible);
    uint p2OccludedMeshlet = (p2Candidate - p2Visible);
    // Total Visible Meshlets = P1Visible + P2Visible
    uint visibleMesh = p1Visible + p2Visible;
    // todo: occluded instance for p1 and p2 (can't be derived from single p1OccludedInst)

    // Setup debug text writer
    DebugRenderContext ctx;
    ctx.DataUAVIdx  = Params.DataUAVIdx;
    ctx.GlyphSRVIdx = Params.GlyphSRVIdx;
    ctx.FontSize    = Params.FontSize;
    ctx._pad        = 0;

    float2 pos = float2(Params.StartX, Params.StartY);
    float lineH = Params.FontSize * 1.25f;
    float indent = 100.0f; // column indent for numbers

    float4 hdrColor   = float4(1.0f, 0.9f, 0.3f, 1.0f);  // Gold headers
    float4 labelColor = float4(0.7f, 0.7f, 0.7f, 1.0f);   // Grey labels
    float4 valColor   = float4(1.0f, 1.0f, 1.0f, 1.0f);   // White values
    float4 sepColor   = float4(0.3f, 0.3f, 0.3f, 1.0f);   // Dim separators

    // --- Total counts ---
    {
        pos.y += lineH;
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.SetColor(valColor);
        w.Char('I');
        w.Char(' ');
        w.Int((int)totalInst);  // Total Instance
        w.Char('/');
        w.Char('M');
        w.Char(' ');
        w.Int((int)totalMeshlets);  // Total Meshlets
        pos.y += lineH;
    }
    // --- Separator ---
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, sepColor, 1.0f);
        [unroll]
        for (int i = 0; i < 29; ++i)
            w.Char('-');
        pos.y += lineH;
    }

    // --- Occluded ---
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char('1');
        w.Char(' ');
        w.Int((int)p1OccludedMeshlet);
        w.Char('/');
        w.Char('2');
        w.Char(' ');
        w.Int((int)p2OccludedMeshlet);
        pos.y += lineH;
    }

    // --- Separator ---
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, sepColor, 1.0f);
        [unroll]
        for (int i = 0; i < 29; ++i)
            w.Char('-');
        pos.y += lineH;
    }

    // --- Visible (derived) ---
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char('M');
        w.Char(' ');
        w.Int((int)visibleMesh);
        pos.y += lineH;
    }
}
