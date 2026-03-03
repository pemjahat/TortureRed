#ifndef IRCACHE_COMMON_HLSL
#define IRCACHE_COMMON_HLSL

#include "Common.hlsl"

// ---------------------------------------------------------------------------
// Spatial cascaded irradiance cache
//   8 cascades of 32^3 cells, camera-centered.
//   Base cell diameter = 0.5 world units (cascade 0).
//   Each subsequent cascade doubles the cell size.
// ---------------------------------------------------------------------------

#define IRCACHE_CASCADE_SIZE        32
#define IRCACHE_CASCADE_COUNT       8
#define IRCACHE_GRID_CELL_DIAMETER  0.5f        // base cell size in world units at cascade 0
#define IRCACHE_RAYS_PER_PROBE      8

#define IRCACHE_MAX_ENTRIES         32768       // hard cap on simultaneous live probes
#define IRCACHE_ENTRY_LIFE_MAX      8           // frames without tracing before expiry

// Total flat cell array = CASCADE_SIZE^3 * CASCADE_COUNT = 32768 * 8 = 262144
#define IRCACHE_TOTAL_CELLS         262144

// ---------------------------------------------------------------------------
// Entry lifecycle flags  (stored as uint in the life RWByteAddressBuffer)
// ---------------------------------------------------------------------------
static const uint IRCACHE_ENTRY_META_OCCUPIED       = 1u;
static const uint IRCACHE_ENTRY_META_JUST_ALLOCATED = 2u;
static const uint IRCACHE_ENTRY_LIFE_RECYCLED       = 0xFFFFFFFFu;

bool is_ircache_entry_valid(uint life) { return life < (uint)IRCACHE_ENTRY_LIFE_MAX; }

// ---------------------------------------------------------------------------
// Meta buffer byte offsets  (4 × uint32 = 16 bytes total)
// ---------------------------------------------------------------------------
#define IRCACHE_META_TRACING_ALLOC_COUNT    0   // snapshot used by current frame's Update dispatch
#define IRCACHE_META_ENTRY_COUNT            4   // high-watermark; Age sweeps [0, entry_count)
#define IRCACHE_META_ALLOC_COUNT            8   // current live count (alloc/free kept in sync)
#define IRCACHE_META_COMPACT_WRITE_IDX     12   // atomic slot counter used by the Age pass

// ---------------------------------------------------------------------------
// IrcacheCoord — (coord, cascade) pair with flat cell index
// ---------------------------------------------------------------------------
struct IrcacheCoord
{
    uint3 coord;    // [0 .. CASCADE_SIZE-1] per axis
    uint  cascade;  // [0 .. CASCADE_COUNT-1]

    uint cell_idx()
    {
        return coord.x
             + coord.y * IRCACHE_CASCADE_SIZE
             + coord.z * (IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE)
             + cascade * (IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE);
    }

    static IrcacheCoord from_coord_cascade(uint3 c, uint casc)
    {
        IrcacheCoord r;
        r.coord   = c;
        r.cascade = casc;
        return r;
    }
};

// Cell diameter (world units) for a given cascade
float ircache_cascade_cell_diameter(uint cascade)
{
    return IRCACHE_GRID_CELL_DIAMETER * (float)(1u << cascade);
}

// World-space corner of cascade `cascade`, snapped so the grid is as centred
// on `cameraPos` as the discrete cell size allows.
float3 ircache_cascade_origin(uint cascade, float3 cameraPos)
{
    float cell = ircache_cascade_cell_diameter(cascade);
    int3  snap = (int3)floor(cameraPos / cell);
    return (float3)(snap - (int)(IRCACHE_CASCADE_SIZE / 2)) * cell;
}

// Map a world position to the finest cascade whose grid contains it.
// Falls back to clamped coarsest cascade for points outside all grids.
IrcacheCoord ws_pos_to_ircache_coord(float3 worldPos, float3 cameraPos)
{
    [unroll]
    for (uint c = 0; c < IRCACHE_CASCADE_COUNT; ++c)
    {
        float  cell   = ircache_cascade_cell_diameter(c);
        float3 origin = ircache_cascade_origin(c, cameraPos);
        float3 gridF  = (worldPos - origin) / cell;
        if (all(gridF >= 0.0f) && all(gridF < (float)IRCACHE_CASCADE_SIZE))
            return IrcacheCoord::from_coord_cascade((uint3)gridF, c);
    }

    // Clamp to coarsest cascade
    uint   c      = IRCACHE_CASCADE_COUNT - 1;
    float  cell   = ircache_cascade_cell_diameter(c);
    float3 origin = ircache_cascade_origin(c, cameraPos);
    uint3  gridI  = (uint3)clamp((worldPos - origin) / cell,
                                 0.0f, (float)(IRCACHE_CASCADE_SIZE - 1));
    return IrcacheCoord::from_coord_cascade(gridI, c);
}

// Reconstruct the world-space cell centre from a coord + live camera position
float3 ircache_coord_to_world_center(IrcacheCoord coord, float3 cameraPos)
{
    float  cell   = ircache_cascade_cell_diameter(coord.cascade);
    float3 origin = ircache_cascade_origin(coord.cascade, cameraPos);
    return origin + (float3(coord.coord) + 0.5f) * cell;
}

// Decode a flat cell_idx back to IrcacheCoord (cascade + per-axis coord)
IrcacheCoord ircache_cell_idx_to_coord(uint cellIdx)
{
    uint cellsPerCascade = IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE;
    uint cascade         = cellIdx / cellsPerCascade;
    uint local           = cellIdx % cellsPerCascade;
    uint3 coord;
    coord.x = local  % IRCACHE_CASCADE_SIZE;
    coord.y = (local / IRCACHE_CASCADE_SIZE) % IRCACHE_CASCADE_SIZE;
    coord.z =  local / (IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE);
    return IrcacheCoord::from_coord_cascade(coord, cascade);
}

#endif // IRCACHE_COMMON_HLSL
