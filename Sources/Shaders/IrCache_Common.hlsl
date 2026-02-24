#ifndef IRCACHE_COMMON_HLSL
#define IRCACHE_COMMON_HLSL

#include "Common.hlsl"

// Irradiance Cache Grid Parameters
#define IRCACHE_GRID_SIZE_X 64
#define IRCACHE_GRID_SIZE_Y 32
#define IRCACHE_GRID_SIZE_Z 64

#define IRCACHE_RAYS_PER_PROBE 8

// Scene bounds for mapping world space to grid space
// Adjust these based on the Sponza scene bounds
static const float3 IRCACHE_SCENE_MIN = float3(-20.0f, -2.0f, -10.0f);
static const float3 IRCACHE_SCENE_MAX = float3(20.0f, 20.0f, 10.0f);

// Helper to convert world position to 3D grid UVW [0, 1]
float3 WorldToIrCacheUVW(float3 worldPos) {
    return saturate((worldPos - IRCACHE_SCENE_MIN) / (IRCACHE_SCENE_MAX - IRCACHE_SCENE_MIN));
}

// Helper to convert 3D grid UVW [0, 1] to world position
float3 IrCacheUVWToWorld(float3 uvw) {
    return lerp(IRCACHE_SCENE_MIN, IRCACHE_SCENE_MAX, uvw);
}

// Helper to convert grid index to world position
float3 GridIndexToWorld(uint3 gridIndex) {
    float3 uvw = (float3(gridIndex) + 0.5f) / float3(IRCACHE_GRID_SIZE_X, IRCACHE_GRID_SIZE_Y, IRCACHE_GRID_SIZE_Z);
    return IrCacheUVWToWorld(uvw);
}

#endif // IRCACHE_COMMON_HLSL
