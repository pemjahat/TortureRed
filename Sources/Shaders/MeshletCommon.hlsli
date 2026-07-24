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

// Frustum culling: conservative AABB vs view-projection matrix
// Returns true if the AABB is at least partially visible.
bool FrustumCullMeshlet(MeshletBounds bounds, float4x4 localToWorld, float4x4 viewProj)
{
    // Transform center to world space, then clip space
    float3 worldCenter = mul(float4(bounds.LocalCenter, 1.0), localToWorld).xyz;

    // Compute world-space AABB corners using extents
    // For conservative frustum test, we use 8 corners of the AABB
    float3 ext = bounds.LocalExtents;

    // Transform all 8 corners to clip space and test
    float4 corners[8];
    corners[0] = mul(float4(worldCenter + float3(-ext.x, -ext.y, -ext.z), 1.0), viewProj);
    corners[1] = mul(float4(worldCenter + float3( ext.x, -ext.y, -ext.z), 1.0), viewProj);
    corners[2] = mul(float4(worldCenter + float3(-ext.x,  ext.y, -ext.z), 1.0), viewProj);
    corners[3] = mul(float4(worldCenter + float3( ext.x,  ext.y, -ext.z), 1.0), viewProj);
    corners[4] = mul(float4(worldCenter + float3(-ext.x, -ext.y,  ext.z), 1.0), viewProj);
    corners[5] = mul(float4(worldCenter + float3( ext.x, -ext.y,  ext.z), 1.0), viewProj);
    corners[6] = mul(float4(worldCenter + float3(-ext.x,  ext.y,  ext.z), 1.0), viewProj);
    corners[7] = mul(float4(worldCenter + float3( ext.x,  ext.y,  ext.z), 1.0), viewProj);

    // Check if ALL corners are outside the same plane → culled
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
    return !culled;
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