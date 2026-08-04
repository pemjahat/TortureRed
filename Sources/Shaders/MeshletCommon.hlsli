#ifndef MESHLET_COMMON_HLSLI
#define MESHLET_COMMON_HLSLI

#include "Shared/SharedTypes.h"

// Unpack helpers for compressed vertex attributes

float3 UnpackPosition(StructuredBuffer<float3> positions, uint offset, uint index)
{
    return positions[offset + index];
}

float3 UnpackNormalRGB10A2(StructuredBuffer<uint> normals, uint offset, uint index)
{
    uint packed = normals[offset + index];
    // Decode RGB10A2_SNORM
    int x = ((packed >> 0)  & 0x3FF) - 512;
    int y = ((packed >> 10) & 0x3FF) - 512;
    int z = ((packed >> 20) & 0x3FF) - 512;
    return normalize(float3(float(x) / 511.0, float(y) / 511.0, float(z) / 511.0));
}

float2 UnpackUVRG16(StructuredBuffer<uint> uvs, uint offset, uint index)
{
    uint packed = uvs[offset + index];
    uint u16 = packed & 0xFFFF;
    uint v16 = (packed >> 16) & 0xFFFF;
    return float2(f16tof32(u16), f16tof32(v16));
}

// Frustum culling: world-space AABB vs view-projection matrix — conservative test.
// Adria-style shared projection (Adria workspace: Assets/Shaders/Meshlets/
// GpuDrivenRendering.hlsli, FrustumCull): the 8 corners are projected ONCE and the
// NDC rect + nearest depth are returned alongside visibility, so HZBCull can
// consume the rect directly instead of re-projecting the AABB.
struct FrustumCullData {
    bool   IsVisible;
    float3 RectMin; // NDC min xy (z = farthest corner; kept for symmetry, unused)
    float3 RectMax; // NDC max xy (z = nearest corner — reverse-Z: larger = closer)
};

FrustumCullData FrustumCullAABB(float3 worldCenter, float3 worldExtents, float4x4 viewProj)
{
    FrustumCullData data = (FrustumCullData)0;
    data.IsVisible = true;

    // 8 clip-space corners via 3 axis vectors (4 muls + adds instead of 8 muls)
    float3x4 axis;
    axis[0] = mul(float4(worldExtents.x * 2, 0, 0, 0), viewProj);
    axis[1] = mul(float4(0, worldExtents.y * 2, 0, 0), viewProj);
    axis[2] = mul(float4(0, 0, worldExtents.z * 2, 0), viewProj);

    float4 pos000 = mul(float4(worldCenter - worldExtents, 1.0), viewProj);
    float4 pos100 = pos000 + axis[0];
    float4 pos010 = pos000 + axis[1];
    float4 pos110 = pos010 + axis[0];
    float4 pos001 = pos000 + axis[2];
    float4 pos101 = pos100 + axis[2];
    float4 pos011 = pos010 + axis[2];
    float4 pos111 = pos110 + axis[2];

    float minW = min(min(min(pos000.w, pos100.w), min(pos010.w, pos110.w)),
                     min(min(pos001.w, pos101.w), min(pos011.w, pos111.w)));
    float maxW = max(max(max(pos000.w, pos100.w), max(pos010.w, pos110.w)),
                     max(max(pos001.w, pos101.w), max(pos011.w, pos111.w)));

    // Check if ALL corners are outside the same plane → culled
    float4 corners[8] = { pos000, pos100, pos010, pos110, pos001, pos101, pos011, pos111 };
    bool allOutsideLeft   = true;
    bool allOutsideRight  = true;
    bool allOutsideTop    = true;
    bool allOutsideBottom = true;
    bool allOutsideNear   = true;
    bool allOutsideFar    = true;

    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float w = abs(corners[i].w);
        float x = corners[i].x;
        float y = corners[i].y;
        float z = corners[i].z;

        allOutsideLeft   = allOutsideLeft   && (x < -w);
        allOutsideRight  = allOutsideRight  && (x >  w);
        allOutsideTop    = allOutsideTop    && (y >  w);
        allOutsideBottom = allOutsideBottom && (y < -w);
        allOutsideNear   = allOutsideNear   && (z <  0.0);
        allOutsideFar    = allOutsideFar    && (z >  w);
    }

    bool culled = allOutsideLeft || allOutsideRight || allOutsideTop ||
                  allOutsideBottom || allOutsideNear || allOutsideFar;
    data.IsVisible = !culled && (maxW > 0.0f);

    // NDC divide → screen rect + nearest depth (reverse-Z: nearest = max ndc.z).
    // Unguarded like Adria: the minW/maxW fixup below sanitizes the only case
    // where the divide is meaningless.
    float3 ssPos000 = pos000.xyz / pos000.w;
    float3 ssPos100 = pos100.xyz / pos100.w;
    float3 ssPos010 = pos010.xyz / pos010.w;
    float3 ssPos110 = pos110.xyz / pos110.w;
    float3 ssPos001 = pos001.xyz / pos001.w;
    float3 ssPos101 = pos101.xyz / pos101.w;
    float3 ssPos011 = pos011.xyz / pos011.w;
    float3 ssPos111 = pos111.xyz / pos111.w;

    data.RectMin = min(min(min(ssPos000, ssPos100), min(ssPos010, ssPos110)),
                       min(min(ssPos001, ssPos101), min(ssPos011, ssPos111)));
    data.RectMax = max(max(max(ssPos000, ssPos100), max(ssPos010, ssPos110)),
                       max(max(ssPos001, ssPos101), max(ssPos011, ssPos111)));

    // Near-plane straddle: some corner is behind the camera, so the projected
    // positions are meaningless — force the widest, closest possible footprint
    // (z=1 is the nearest depth in reverse-Z) so occlusion can never falsely cull.
    if (minW <= 0.0f && maxW > 0.0f)
    {
        data.RectMin = float3(-1.0f, -1.0f, 0.0f);
        data.RectMax = float3( 1.0f,  1.0f, 1.0f);
    }

    return data;
}

// Transform a local-space AABB by localToWorld → conservative world-space AABB.
// Center is fully transformed; extents are expanded by the absolute value of the
// linear part (row-vector convention: worldExtents_i = dot(extents, abs(column_i))).
// Exact for the AABB of a transformed box — required for rotated/scaled instances.
void TransformAABBToWorld(float3 localCenter, float3 localExtents, float4x4 localToWorld,
                          out float3 worldCenter, out float3 worldExtents)
{
    worldCenter  = mul(float4(localCenter, 1.0f), localToWorld).xyz;
    worldExtents = mul(localExtents, (float3x3)localToWorld);
}

// Frustum culling: meshlet-space AABB → world → clip, conservative test.
// Returns the shared FrustumCullData (visibility + NDC rect + nearest depth).
FrustumCullData FrustumCullMeshlet(MeshletBounds bounds, float4x4 localToWorld, float4x4 viewProj)
{
    float3 worldCenter, worldExtents;
    TransformAABBToWorld(bounds.LocalCenter, bounds.LocalExtents, localToWorld,
                         worldCenter, worldExtents);
    return FrustumCullAABB(worldCenter, worldExtents, viewProj);
}

// --- Wireframe edge detection from barycentric coordinates ---
// Returns 0 on edges, 1 in interior. Use: color *= saturate(Wireframe(bary) + 0.8)
float Wireframe(float3 barycentrics, float thickness = 0.2f, float smoothing = 1.0f)
{
    float3 deltas = fwidth(barycentrics);
    float3 bary   = smoothstep(deltas * thickness, deltas * (thickness + smoothing), barycentrics);
    float  minBary = min(bary.x, min(bary.y, bary.z));
    return minBary;
}

#endif // MESHLET_COMMON_HLSLI