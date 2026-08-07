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
//   BaseSlot=0 → P1: write Candidate@[0], Visible@[1], Occluded@[2], InstanceVisible@[3]
//                     also zero P2 slots [4]–[7] so stale data from prior frames is cleared
//   BaseSlot=4 → P2: write Candidate@[4], Visible@[5], Occluded@[6], InstanceVisible@[7]
// =============================================================================

ConstantBuffer<CullStatsCopyParams> CopyParams : register(b2);

[numthreads(1, 1, 1)]
void CopyCullStatsCS(uint3 tid : SV_DispatchThreadID)
{
    StructuredBuffer<uint> candidateCounter        = ResourceDescriptorHeap[CopyParams.CandidateCounterSRVIdx];
    StructuredBuffer<uint> visibleCounter          = ResourceDescriptorHeap[CopyParams.VisibleCounterSRVIdx];
    StructuredBuffer<uint> occludedCounter         = ResourceDescriptorHeap[CopyParams.OccludedCounterSRVIdx];
    StructuredBuffer<uint> instanceVisibleCounter  = ResourceDescriptorHeap[CopyParams.InstanceVisibleCounterSRVIdx];
    RWStructuredBuffer<uint> stats                 = ResourceDescriptorHeap[CopyParams.StatsBufferUAVIdx];

    uint base = CopyParams.BaseSlot;

    stats[base + 0] = candidateCounter[0];          // P1_CANDIDATE_MESHLETS / P2_CANDIDATE_MESHLETS
    stats[base + 1] = visibleCounter[0];            // P1_VISIBLE_MESHLETS   / P2_VISIBLE_MESHLETS
    stats[base + 2] = occludedCounter[0];           // P1_OCCLUDED_INSTANCES / P2_OCCLUDED_INSTANCES
    stats[base + 3] = instanceVisibleCounter[0];    // P1_VISIBLE_INSTANCES  / P2_VISIBLE_INSTANCES

    if (base == 0) // (Phase 1 always runs first each frame)
    {
        // Pre-zero the P2 slots so CullStatsCS never displays stale numbers left
        // over from an earlier frame — e.g. when two-pass culling is disabled
        // this frame, Phase 2's copy never runs at all. If Phase 2 DOES run
        // later this frame, its own copy (base==4, above) overwrites these.
        stats[CULL_STATS_P2_CANDIDATE_MESHLETS]   = 0;
        stats[CULL_STATS_P2_VISIBLE_MESHLETS]     = 0;
        stats[CULL_STATS_P2_OCCLUDED_INSTANCES]   = 0;
        stats[CULL_STATS_P2_VISIBLE_INSTANCES]    = 0;
    }
}

// =============================================================================
// CullStatsCS
// Reads CullStatsBuffer, computes derived stats, and renders an on-screen table
// via the GPU debug-text system (DebugTextRender.hlsli). The output is a set of
// formatted text lines anchored at (StartX, StartY).
//
// Table:
//   Total Meshlet: NNN
//
//   Phase1
//     Input instances:    NNN  (totalInst from Params)
//     Input meshlet:      NNN  (P1_CANDIDATE_MESHLETS)
//     Culled instance:    NNN  (totalInst - P1_VISIBLE_INSTANCES)
//     Culled meshlet:     NNN  (candidate - visible)
//     Visible instance:   NNN  (P1_VISIBLE_INSTANCES)
//     Visible meshlet:    NNN  (P1_VISIBLE_MESHLETS)
//
//   Phase2
//     Input instance:     NNN  (P1_OCCLUDED_INSTANCES — deferred from P1)
//     Input meshlet:      NNN  (P2_CANDIDATE_MESHLETS)
//     Visible instance:   NNN  (P2_VISIBLE_INSTANCES)
//     Visible meshlet:    NNN  (P2_VISIBLE_MESHLETS)
// =============================================================================

ConstantBuffer<CullStatsParams> Params : register(b2);

[numthreads(1, 1, 1)]
void CullStatsCS(uint3 tid : SV_DispatchThreadID)
{
    StructuredBuffer<uint> stats = ResourceDescriptorHeap[Params.StatsBufferSRVIdx];

    uint p1Candidate     = stats[CULL_STATS_P1_CANDIDATE_MESHLETS];
    uint p1Visible       = stats[CULL_STATS_P1_VISIBLE_MESHLETS];
    uint p1OccludedInst  = stats[CULL_STATS_P1_OCCLUDED_INSTANCES];
    uint p1VisibleInst   = stats[CULL_STATS_P1_VISIBLE_INSTANCES];   // instances that passed P1
    uint p2Candidate     = stats[CULL_STATS_P2_CANDIDATE_MESHLETS];
    uint p2Visible       = stats[CULL_STATS_P2_VISIBLE_MESHLETS];
    uint p2VisibleInst   = stats[CULL_STATS_P2_VISIBLE_INSTANCES];   // instances newly visible in P2
    uint totalMeshlets = Params.TotalMeshlets;
    uint totalInst     = Params.TotalInstances;

    // Derived stats
    uint p1CulledInstance = totalInst - p1VisibleInst;   // total - visible = culled (frustum + occluded) P1
    uint p1CulledMeshlet  = p1Candidate - p1Visible;     // candidate - visible = culled meshlets P1

    // Setup debug text writer
    DebugRenderContext ctx;
    ctx.DataUAVIdx  = Params.DataUAVIdx;
    ctx.GlyphSRVIdx = Params.GlyphSRVIdx;
    ctx.FontSize    = Params.FontSize;
    ctx._pad        = 0;

    float2 pos = float2(Params.StartX, Params.StartY);
    float lineH = Params.FontSize * 1.25f;

    float4 hdrColor   = float4(1.0f, 0.9f, 0.3f, 1.0f);  // Gold headers
    float4 labelColor = float4(0.7f, 0.7f, 0.7f, 1.0f);   // Grey labels
    float4 valColor   = float4(1.0f, 1.0f, 1.0f, 1.0f);   // White values

    // --- Total Meshlet count ---
    pos.y += lineH;
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char('T'); w.Char('o'); w.Char('t'); w.Char('a'); w.Char('l');
        w.Char(' '); w.Char('M'); w.Char('e'); w.Char('s'); w.Char('h'); w.Char('l'); w.Char('e'); w.Char('t');
        w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)totalMeshlets);
        pos.y += lineH;
    }

    // --- Phase 1 ---
    pos.y += lineH;
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, hdrColor, 1.0f);
        w.Char('P'); w.Char('h'); w.Char('a'); w.Char('s'); w.Char('e'); w.Char('1');
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('I'); w.Char('n'); w.Char('p'); w.Char('u'); w.Char('t');
        w.Char(' '); w.Char('i'); w.Char('n'); w.Char('s'); w.Char('t'); w.Char('a'); w.Char('n');
        w.Char('c'); w.Char('e'); w.Char('s'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)totalInst);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('I'); w.Char('n'); w.Char('p'); w.Char('u'); w.Char('t');
        w.Char(' '); w.Char('m'); w.Char('e'); w.Char('s'); w.Char('h'); w.Char('l'); w.Char('e');
        w.Char('t'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p1Candidate);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('C'); w.Char('u'); w.Char('l'); w.Char('l'); w.Char('e');
        w.Char('d'); w.Char(' '); w.Char('i'); w.Char('n'); w.Char('s'); w.Char('t'); w.Char('a');
        w.Char('n'); w.Char('c'); w.Char('e'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p1CulledInstance);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('C'); w.Char('u'); w.Char('l'); w.Char('l'); w.Char('e');
        w.Char('d'); w.Char(' '); w.Char('m'); w.Char('e'); w.Char('s'); w.Char('h'); w.Char('l');
        w.Char('e'); w.Char('t'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p1CulledMeshlet);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('V'); w.Char('i'); w.Char('s'); w.Char('i'); w.Char('b');
        w.Char('l'); w.Char('e'); w.Char(' '); w.Char('i'); w.Char('n'); w.Char('s'); w.Char('t');
        w.Char('a'); w.Char('n'); w.Char('c'); w.Char('e'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p1VisibleInst);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('V'); w.Char('i'); w.Char('s'); w.Char('i'); w.Char('b');
        w.Char('l'); w.Char('e'); w.Char(' '); w.Char('m'); w.Char('e'); w.Char('s'); w.Char('h');
        w.Char('l'); w.Char('e'); w.Char('t'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p1Visible);
        pos.y += lineH;
    }

    // --- Phase 2 ---
    pos.y += lineH;
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, hdrColor, 1.0f);
        w.Char('P'); w.Char('h'); w.Char('a'); w.Char('s'); w.Char('e'); w.Char('2');
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('I'); w.Char('n'); w.Char('p'); w.Char('u'); w.Char('t');
        w.Char(' '); w.Char('i'); w.Char('n'); w.Char('s'); w.Char('t'); w.Char('a'); w.Char('n');
        w.Char('c'); w.Char('e'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p1OccludedInst);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('I'); w.Char('n'); w.Char('p'); w.Char('u'); w.Char('t');
        w.Char(' '); w.Char('m'); w.Char('e'); w.Char('s'); w.Char('h'); w.Char('l'); w.Char('e');
        w.Char('t'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p2Candidate);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('V'); w.Char('i'); w.Char('s'); w.Char('i'); w.Char('b');
        w.Char('l'); w.Char('e'); w.Char(' '); w.Char('i'); w.Char('n'); w.Char('s'); w.Char('t');
        w.Char('a'); w.Char('n'); w.Char('c'); w.Char('e'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p2VisibleInst);
        pos.y += lineH;
    }
    {
        DebugTextWriter w = CreateDebugTextWriter(ctx, pos, labelColor, 1.0f);
        w.Char(' '); w.Char(' '); w.Char('V'); w.Char('i'); w.Char('s'); w.Char('i'); w.Char('b');
        w.Char('l'); w.Char('e'); w.Char(' '); w.Char('m'); w.Char('e'); w.Char('s'); w.Char('h');
        w.Char('l'); w.Char('e'); w.Char('t'); w.Char(':'); w.Char(' ');
        w.SetColor(valColor); w.Int((int)p2Visible);
        pos.y += lineH;
    }
}
