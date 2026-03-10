#ifndef IRCACHE_DEBUG_SPHERES_HLSL
#define IRCACHE_DEBUG_SPHERES_HLSL

#include "IrCache_Lookup.hlsl"

// ---------------------------------------------------------------------------
// IrCache debug sphere visualization
//   Draws one UV sphere per live probe entry using instanced draw.
//   SV_InstanceID indexes IndirectionBuf (compact live-entry list).
//   VS culls unoccupied instances by emitting a behind-near-plane position.
//
//   Sphere dimensions: 8 stacks × 8 slices → 8×8×6 = 384 verts per instance.
//   Instance count  : IRCACHE_MAX_ENTRIES (32768) — VS culls vacant entries.
//   Sphere radius   : cell_diameter × SPHERE_RADIUS_SCALE (cascade-proportional).
// ---------------------------------------------------------------------------

#define SPHERE_N_STACKS      8
#define SPHERE_N_SLICES      8
#define SPHERE_RADIUS_SCALE  0.25f

static const float SPHERE_PI = 3.14159265f;

// Clip-space sentinel: z < 0 with w = 1 → behind near plane → entire triangle clipped.
static const float4 DEGENERATE_POS = float4(0.0f, 0.0f, -1.0f, 1.0f);

ConstantBuffer<FrameConstants>         FrameCB   : register(b0);
ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

struct VSOutput
{
    float4 position : SV_POSITION;
    float3 color    : COLOR;
};

// ---------------------------------------------------------------------------
// VSMain
// ---------------------------------------------------------------------------
VSOutput VSMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;
    output.color = float3(1.0f, 1.0f, 1.0f);

    RWByteAddressBuffer meta = ResourceDescriptorHeap[g_IrCache.MetaBufIdx];
    uint liveCount = meta.Load(IRCACHE_META_TRACING_ALLOC_COUNT);
    if (instanceID >= liveCount)
    {
        output.position = DEGENERATE_POS;
        return output;
    }

    // ---- Resolve entry and cell from IndirectionBuf ----
    RWStructuredBuffer<uint> indirection = ResourceDescriptorHeap[g_IrCache.IndirectionBufIdx];
    uint entryIdx = indirection[instanceID];

    RWStructuredBuffer<uint> entryCell = ResourceDescriptorHeap[g_IrCache.EntryCellBufIdx];
    uint cellIdx = entryCell[entryIdx];

    RWByteAddressBuffer gridMeta = ResourceDescriptorHeap[g_IrCache.GridMetaBufIdx];
    uint cellPacked = gridMeta.Load(cellIdx * 4);

    // Cull unoccupied entries — emit a degenerate position so every triangle clips
    if (!(ircache_cell_flags(cellPacked) & IRCACHE_ENTRY_META_OCCUPIED))
    {
        output.position = DEGENERATE_POS;
        return output;
    }

    // ---- Probe world centre and radius ----
    IrcacheCoord coord    = ircache_cell_idx_to_coord(cellIdx);

    // Cull entries that don't belong to the selected cascade filter
    if (FrameCB.debugIrCacheCascadeFilter >= 0 && (int)coord.cascade != FrameCB.debugIrCacheCascadeFilter)
    {
        output.position = DEGENERATE_POS;
        return output;
    }

    float3       probePos = ircache_coord_to_world_center(coord, FrameCB.irCacheCameraPosition.xyz);
#if IRCACHE_USE_POSITION_VOTING
    // Prefer the voted/applied probe position; fall back to cell centre if not yet set (w == 0).
    RWStructuredBuffer<float4> posBuf = ResourceDescriptorHeap[g_IrCache.PosBufIdx];
    float4 storedPos = posBuf[entryIdx];
    if (storedPos.w != 0.0f)
        probePos = storedPos.xyz;
#endif
    float        radius   = ircache_cascade_cell_diameter(coord.cascade) * SPHERE_RADIUS_SCALE;

    // ---- Decode UV sphere vertex from SV_VertexID ----
    // Quad layout (6 verts): 0,1,2 then 1,3,2
    //   corner 0: (stack,   slice  )
    //   corner 1: (stack+1, slice  )
    //   corner 2: (stack,   slice+1)
    //   corner 3: (stack+1, slice+1)
    static const uint cornerTable[6] = { 0u, 1u, 2u, 1u, 3u, 2u };

    uint quadIdx  = vertexID / 6;
    uint localIdx = vertexID % 6;
    uint stackIdx = quadIdx / SPHERE_N_SLICES;
    uint sliceIdx = quadIdx % SPHERE_N_SLICES;

    uint corner = cornerTable[localIdx];
    uint si = stackIdx + (corner >> 1);   // stack offset: 0 or 1
    uint li = sliceIdx + (corner  & 1u);  // slice offset: 0 or 1

    float phi   = (float)si / (float)SPHERE_N_STACKS * SPHERE_PI;          // [0, PI]
    float theta = (float)li / (float)SPHERE_N_SLICES * 2.0f * SPHERE_PI;   // [0, 2PI]

    float3 localPos = float3(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );

    float3 worldPos = probePos + localPos * radius;
    output.position = mul(float4(worldPos, 1.0f), FrameCB.viewProj);

    // ---- Per-probe colour based on active debug mode ----
    static const float3 cascadeColors[8] =
    {
        float3(1.00f, 0.20f, 0.20f),   // 0 red
        float3(1.00f, 0.60f, 0.10f),   // 1 orange
        float3(1.00f, 1.00f, 0.10f),   // 2 yellow
        float3(0.20f, 1.00f, 0.20f),   // 3 green
        float3(0.10f, 1.00f, 1.00f),   // 4 cyan
        float3(0.20f, 0.40f, 1.00f),   // 5 blue
        float3(0.70f, 0.20f, 1.00f),   // 6 violet
        float3(1.00f, 0.20f, 0.80f),   // 7 magenta
    };

    [branch]
    if (FrameCB.debugIrCache == IRCACHE_DEBUG_IRRADIANCE)
    {
        RWStructuredBuffer<Reservoir> probeReservoirs = ResourceDescriptorHeap[g_IrCache.IrradianceBufIdx];
        Reservoir r = probeReservoirs[entryIdx];
        float validity = (r.M > 0.0f && r.W > 0.0f) ? 1.0f : 0.0f;
        float3 resolved = r.radiance * r.W;
        //output.color = lerp(float3(0.05f, 0.05f, 0.05f), resolved, validity);
        output.color = resolved;
    }
    else if (FrameCB.debugIrCache == IRCACHE_DEBUG_LIFE)
    {
        RWByteAddressBuffer lifeB = ResourceDescriptorHeap[g_IrCache.LifeBufIdx];
        uint lifeVal = lifeB.Load(entryIdx * 4);
        float t = saturate((float)lifeVal / (float)(IRCACHE_ENTRY_LIFE_MAX - 1));
        float3 fresh  = float3(0.0f, 1.0f, 0.0f);
        float3 mid    = float3(1.0f, 1.0f, 0.0f);
        float3 expiry = float3(1.0f, 0.0f, 0.0f);
        output.color = t < 0.5f ? lerp(fresh, mid, t * 2.0f) : lerp(mid, expiry, (t - 0.5f) * 2.0f);
    }
    else if (FrameCB.debugIrCache == IRCACHE_DEBUG_CASCADE)
    {
        output.color = cascadeColors[coord.cascade];
    }
    // IRCACHE_DEBUG_OFF → default white set above

    return output;
}

// ---------------------------------------------------------------------------
// PSMain
// ---------------------------------------------------------------------------
float4 PSMain(VSOutput input) : SV_TARGET
{
    return float4(input.color, 0.85f);
}

#endif // IRCACHE_DEBUG_SPHERES_HLSL
