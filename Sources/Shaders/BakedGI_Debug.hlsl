// =============================================================================
// BakedGI_Debug.hlsl — probe debug overlay (post-composite LDR).
//
// Four views (FrameConstants::bakedGIDebugView):
//   1 Placement      — cube per grid cell: green = valid (free space),
//                      red = invalid (inside geometry).
//   2 Lit Irradiance — billboard SH ball per valid probe: each sphere point
//                      evaluates the probe's lit SH9 coefficients at its own
//                      normal, showing the directional structure of e(p);
//                      scaled by bakedGIDebugScale (lit values are
//                      FP16Scale'd scene HDR).
//   3 Sky Visibility — cube per cell, grayscale = mean of the 16 baked
//                      directional visibilities V_i (sky-open fraction);
//                      red = invalid.
//   4 Visibility Rays— thin oriented box per (probe, fibonacci direction):
//                      green = open sky (V_i = 1), red = blocked. Camera-
//                      distance culled (spacing * 8 window), invalid probes
//                      skipped.
//
// Post-composite: drawn after the TAA resolve/tonemap directly onto the
// backbuffer at output resolution — display-referred colors, no taaEnabled
// branching, temporal history unpolluted. Occlusion is a manual reverse-Z
// compare against the GBuffer depth SRV. Geometry is generated entirely from
// SV_VertexID/SV_InstanceID (no vertex buffers); SV_VertexID restarts per
// instance, so the cell index comes from SV_InstanceID.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// BakedGI.hlsli references FrameCB — include after the declaration above.
#include "BakedGI.hlsli"

#define BAKED_GI_DEBUG_PLACEMENT  1u
#define BAKED_GI_DEBUG_LIT        2u
#define BAKED_GI_DEBUG_SKYVIS     3u
#define BAKED_GI_DEBUG_VISRAYS    4u

// UV-sphere tessellation (same layout as IrCache_DebugSpheres: 8x8 quads).
#define DBG_N_STACKS    8
#define DBG_N_SLICES    8
#define DBG_SPHERE_VERTS (DBG_N_STACKS * DBG_N_SLICES * 6)

struct VSOut
{
    float4 pos   : SV_POSITION;
    float4 color : COLOR;
};

// 12 triangles of a [-1,1]^3 cube.
static const float3 kCubeTris[36] =
{
    // +X
    float3( 1, -1, -1), float3( 1,  1, -1), float3( 1,  1,  1),
    float3( 1, -1, -1), float3( 1,  1,  1), float3( 1, -1,  1),
    // -X
    float3(-1, -1,  1), float3(-1,  1,  1), float3(-1,  1, -1),
    float3(-1, -1,  1), float3(-1,  1, -1), float3(-1, -1, -1),
    // +Y
    float3(-1,  1, -1), float3(-1,  1,  1), float3( 1,  1,  1),
    float3(-1,  1, -1), float3( 1,  1,  1), float3( 1,  1, -1),
    // -Y
    float3(-1, -1,  1), float3(-1, -1, -1), float3( 1, -1, -1),
    float3(-1, -1,  1), float3( 1, -1, -1), float3( 1, -1,  1),
    // +Z
    float3(-1, -1,  1), float3( 1, -1,  1), float3( 1,  1,  1),
    float3(-1, -1,  1), float3( 1,  1,  1), float3(-1,  1,  1),
    // -Z
    float3( 1, -1, -1), float3(-1, -1, -1), float3(-1,  1, -1),
    float3( 1, -1, -1), float3(-1,  1, -1), float3( 1,  1, -1),
};

// Clip-space sentinel: z < 0 with w = 1 fails the reverse-Z clip test, so the
// whole triangle is discarded (used for per-instance culls).
static const float4 kDegenerate = float4(0.0f, 0.0f, -1.0f, 1.0f);

// Sphere quad corner decode (IrCache_DebugSpheres layout):
// quad 0-63, 6 verts each: triangles (0,1,2) and (1,3,2).
static const uint kCornerTable[6] = { 0u, 1u, 2u, 1u, 3u, 2u };

float3 BakedGIProbeCellPos(uint probe, uint3 dims, float3 gridMin, float spacing)
{
    uint3 cell = uint3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    return gridMin + float3(cell) * spacing;
}

VSOut VSMain(uint vid : SV_VertexID, uint instance : SV_InstanceID)
{
    VSOut o;
    o.pos   = kDegenerate;
    o.color = float4(1.0f, 1.0f, 1.0f, 1.0f);

    const uint   mode    = FrameCB.bakedGIDebugView;
    const uint3  dims    = uint3(FrameCB.bakedGIDimX, FrameCB.bakedGIDimY, FrameCB.bakedGIDimZ);
    const float3 gridMin = float3(FrameCB.bakedGIGridMinX, FrameCB.bakedGIGridMinY, FrameCB.bakedGIGridMinZ);
    const float  spacing = FrameCB.bakedGISpacing;
    const uint   probeCount = dims.x * dims.y * dims.z;

    StructuredBuffer<float4> response = ResourceDescriptorHeap[FrameCB.bakedGIResponseSRVIndex];
    StructuredBuffer<uint>   meta     = ResourceDescriptorHeap[FrameCB.bakedGIProbeMetaSRVIndex];

    // --- Mode 4: per-(probe, direction) visibility rays ---
    if (mode == BAKED_GI_DEBUG_VISRAYS)
    {
        const uint probe = instance / BAKED_GI_DIRECTIONS;
        const uint dir   = instance % BAKED_GI_DIRECTIONS;
        if (probe >= probeCount)
            return o;

        const float3 probePos = BakedGIProbeCellPos(probe, dims, gridMin, spacing);
        if (!(meta[probe] & 1u))
            return o; // invalid probes have zeroed transport — nothing to show
        if (distance(probePos, FrameCB.cameraPosition.xyz) > spacing * 8.0f)
            return o; // near-camera window keeps the ray count readable

        const float3 w = BakedGIDeltaDirection(dir);
        const uint slot = (probe * BAKED_GI_DIRECTIONS + dir) * BAKED_GI_RESPONSE_F4;
        const float V   = response[slot + 9].x;
        o.color = (V > 0.5f) ? float4(0.1f, 0.9f, 0.2f, 1.0f)      // open sky
                             : float4(0.95f, 0.15f, 0.15f, 1.0f);   // blocked

        // Thin oriented box from the probe center along w (local z -> w).
        const float3 up        = (abs(w.y) < 0.99f) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
        const float3 tangent   = normalize(cross(up, w));
        const float3 bitangent = cross(w, tangent);
        const float3 c    = kCubeTris[vid % 36u];
        const float  len  = spacing * 0.35f;
        const float  half = spacing * 0.015f;
        const float  along = (c.z * 0.5f + 0.5f) * len; // map [-1,1] -> [0, len]
        const float3 world = probePos + tangent * (c.x * half) + bitangent * (c.y * half) + w * along;

        o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
        return o;
    }

    // --- Per-probe views (placement / lit irradiance / sky visibility) ---
    const uint probe = instance;
    if (probe >= probeCount)
        return o;

    const float3 probePos = BakedGIProbeCellPos(probe, dims, gridMin, spacing);
    const bool   valid    = (meta[probe] & 1u) != 0u;

    if (mode == BAKED_GI_DEBUG_LIT)
    {
        if (!valid)
            return o;

        // UV-sphere vertex: unit sphere position doubles as the eval normal.
        const uint v         = vid % DBG_SPHERE_VERTS;
        const uint quadIdx   = v / 6u;
        const uint localIdx  = v % 6u;
        const uint stackIdx  = quadIdx / DBG_N_SLICES;
        const uint sliceIdx  = quadIdx % DBG_N_SLICES;
        const uint corner    = kCornerTable[localIdx];
        const uint si = stackIdx + (corner >> 1u);
        const uint li = sliceIdx + (corner & 1u);
        const float sph   = (float)si / (float)DBG_N_STACKS * PI;         // [0, PI]
        const float theta = (float)li / (float)DBG_N_SLICES * 2.0f * PI;  // [0, 2PI]
        const float3 n = float3(sin(sph) * cos(theta), cos(sph), sin(sph) * sin(theta));

        // Lit SH9 evaluation at the sphere normal: the ball's shading IS e(p).
        StructuredBuffer<float4> lit = ResourceDescriptorHeap[FrameCB.bakedGIProbeSRVIndex];
        float basis[9];
        EvalSH9Basis(n, basis);
        float3 e = float3(0.0f, 0.0f, 0.0f);
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
            e += lit[probe * 9u + lm].rgb * basis[lm];

        o.color = float4(saturate(max(e, 0.0f.xxx) * FrameCB.bakedGIDebugScale), 1.0f);

        const float3 world = probePos + n * (spacing * 0.12f);
        o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
        return o;
    }

    // Cube modes: pick the color source.
    if (mode == BAKED_GI_DEBUG_SKYVIS)
    {
        float vMean = 0.0f;
        [unroll]
        for (uint i = 0; i < BAKED_GI_DIRECTIONS; ++i)
        {
            const uint slot = (probe * BAKED_GI_DIRECTIONS + i) * BAKED_GI_RESPONSE_F4;
            vMean += response[slot + 9].x;
        }
        vMean /= (float)BAKED_GI_DIRECTIONS;
        o.color = valid ? float4(vMean.xxx, 1.0f)                    // sky-open fraction
                        : float4(0.95f, 0.15f, 0.15f, 1.0f);         // invalid
    }
    else // BAKED_GI_DEBUG_PLACEMENT
    {
        o.color = valid ? float4(0.1f, 0.9f, 0.2f, 1.0f)             // valid (free space)
                        : float4(0.95f, 0.15f, 0.15f, 1.0f);         // invalid (inside geometry)
    }

    const float3 c = kCubeTris[vid % 36u];
    const float3 world = probePos + c * (spacing * 0.04f);
    o.pos = mul(float4(world, 1.0f), FrameCB.viewProjUnjittered);
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    // Manual reverse-Z occlusion against the GBuffer (same semantics as the
    // old DSV GREATER_EQUAL test): map this output-res pixel to its
    // internal-res texel the way NaiveTsr_Resolve does. NDC z is unaffected
    // by the xy jitter, so the comparison is exact.
    Texture2D<float> depthTex = ResourceDescriptorHeap[FrameCB.depthIndex];
    float2 scale = float2(FrameCB.internalWidth, FrameCB.internalHeight)
                 / float2(FrameCB.outputWidth, FrameCB.outputHeight);
    int2 ip = min(int2(i.pos.xy * scale),
                  int2(FrameCB.internalWidth, FrameCB.internalHeight) - 1);
    float sceneZ = depthTex[ip];
    clip(i.pos.z - sceneZ + 1e-5f); // visible when fragZ >= sceneZ (reverse-Z)

    return i.color;
}
