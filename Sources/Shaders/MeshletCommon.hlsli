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

// Frustum + HZB culling: world-space bounding SPHERE — conservative test.
// Ported from Niagara (D:\niagara\src\shaders\drawcull.comp.glsl / clustercull.comp.glsl /
// math.h::projectSphere). A sphere is cheaper to test AND cheaper to keep up to date under an
// affine LocalToWorld than an AABB: the center just transforms, and the radius only needs
// scaling by the transform's max axis scale — no re-derivation from 8 corners on every
// transform like an AABB requires (see TransformSphereToWorld below).
//
// Niagara operates in true view space (mat4 view, plus P00/P11/znear from the projection).
// TortureRed only carries a combined viewProj (no separate view matrix) — so instead of
// transforming to view space, this recovers the exact same quantities Niagara's "c" (view-
// space center) needs directly from clip space:
//   clipPos = mul(worldCenter, viewProj);  clipPos.w == -viewSpace.z (Camera::GetProjMatrix's
//   reverse-Z infinite-far layout: clip.w = -view.z, clip.x = view.x*P00, clip.y = view.y*P11)
//   => c = float3(clipPos.x / P00, clipPos.y / P11, clipPos.w)  is EXACTLY Niagara's
//      positive-forward view-space center, with zero extra matrix math.
// P00/P11/zNear themselves come from FrameCB.projectionInverseUnjittered (see call sites):
//   P00 = 1/projInv[0][0], P11 = 1/projInv[1][1], zNear = 1/projInv[2][3] — exact inverse
//   of Camera::GetProjMatrix()'s layout, verified algebraically, jitter-immune by construction.
//
// viewProj vs hzbViewProj: the frustum VISIBILITY decision must always use the CURRENT
// camera (viewProj) — "is this in view this frame". But the HZB rect/depth is a lookup
// into a texture that may have been built from a DIFFERENT camera than the current one
// (Phase 1 of the two-pass cull samples last frame's HZB, built with last frame's camera).
// Take separate projection matrices: viewProj for the frustum test, hzbViewProj for
// everything the HZB lookup consumes (near-plane guard, tangent rect, nearest depth).
// Call sites pass hzbViewProj = FrameCB.viewProjPrevious for Phase 1 (prev HZB) and
// FrameCB.viewProj for Phase 2 (fresh HZB, already aligned with the current camera).
struct FrustumCullData {
    bool   IsVisible;
    float3 RectMin; // NDC min xy (z unused, kept for symmetry with RectMax)
    float3 RectMax; // NDC max xy (z = nearest depth — reverse-Z: larger = closer)
};

FrustumCullData FrustumCullSphere(float3 worldCenter, float worldRadius,
                                  float4x4 viewProj, float4x4 hzbViewProj,
                                  float P00, float P11, float zNear)
{
    FrustumCullData data = (FrustumCullData)0;

    // --- Frustum visibility test: ALWAYS against the CURRENT camera (viewProj) ---
    float4 clipPos = mul(float4(worldCenter, 1.0f), viewProj);
    // Pseudo view-space center: c.z > 0 in front of the camera (Niagara convention).
    float3 c = float3(clipPos.x / P00, clipPos.y / P11, clipPos.w);

    // Frustum test: sphere vs symmetric frustum planes (Niagara drawcull.comp.glsl).
    // frustumX/Y are the normalized (P00,1)/(P11,1) plane coefficients — testing left/right
    // (resp. top/bottom) simultaneously via abs(c.x)/abs(c.y) frustum symmetry.
    float2 frustumX = normalize(float2(P00, 1.0f));
    float2 frustumY = normalize(float2(P11, 1.0f));
    bool visible = true;
    visible = visible && (c.z * frustumX.y - abs(c.x) * frustumX.x > -worldRadius);
    visible = visible && (c.z * frustumY.y - abs(c.y) * frustumY.x > -worldRadius);
    visible = visible && (c.z + worldRadius > zNear); // near-plane only — proj is infinite-far

    data.IsVisible = visible;
    if (!visible)
        return data;

    // --- HZB rect: projected with hzbViewProj — the basis the HZB texture that will
    // actually be sampled was built in (see comment above the struct). Re-derive the
    // pseudo view-space center from scratch using hzbViewProj instead of reusing "c",
    // since the two projections can legitimately disagree when the camera moved. ---
    float4 hzbClipPos = mul(float4(worldCenter, 1.0f), hzbViewProj);
    float3 hc = float3(hzbClipPos.x / P00, hzbClipPos.y / P11, hzbClipPos.w);

    // --- Analytic sphere → NDC rect (Mara/McGuire, "2D Polyhedral Bounds of a Clipped,
    // Perspective-Projected 3D Sphere", 2013) — ported from Niagara's projectSphere(). ---
    if (hc.z < worldRadius + zNear)
    {
        // Sphere straddles/behind the near plane (in the HZB's camera basis) — tangent-line
        // geometry below is meaningless. Same near-plane-straddle policy as the old AABB
        // path: force the widest, closest possible footprint so occlusion can never
        // falsely cull. Also naturally covers "object didn't exist in last frame's view".
        data.RectMin = float3(-1.0f, -1.0f, 0.0f);
        data.RectMax = float3( 1.0f,  1.0f, 1.0f);
        return data;
    }

    float3 cr = hc * worldRadius;
    float czr2 = hc.z * hc.z - worldRadius * worldRadius; // >= 0 guaranteed by the guard above

    float vx = sqrt(hc.x * hc.x + czr2);
    float minx = (vx * hc.x - cr.z) / (vx * hc.z + cr.x);
    float maxx = (vx * hc.x + cr.z) / (vx * hc.z - cr.x);

    float vy = sqrt(hc.y * hc.y + czr2);
    float miny = (vy * hc.y - cr.z) / (vy * hc.z + cr.y);
    float maxy = (vy * hc.y + cr.z) / (vy * hc.z - cr.y);

    // minx/maxx/miny/maxy are dimensionless view-space slopes (x/z, y/z) at the tangent
    // points; multiplying by P00/P11 gives NDC x/y directly (no perspective divide needed —
    // that's the whole point of solving via tangent geometry instead of per-corner project).
    data.RectMin = float3(clamp(minx * P00, -1.0f, 1.0f), clamp(miny * P11, -1.0f, 1.0f), 0.0f);
    data.RectMax.x = clamp(maxx * P00, -1.0f, 1.0f);
    data.RectMax.y = clamp(maxy * P11, -1.0f, 1.0f);

    // Nearest depth (reverse-Z: larger = closer). The sphere's closest point along the
    // view axis is at view-depth (hc.z - radius); convert to reverse-Z NDC depth the same
    // way Camera::GetProjMatrix() does: ndc.z = zNear / viewDepth (matches Niagara's
    // depthSphere = znear/(center.z-radius) exactly). Guard above guarantees the
    // denominator is >= zNear > 0.
    data.RectMax.z = saturate(zNear / (hc.z - worldRadius));

    return data;
}

// Transform a local-space bounding sphere by localToWorld → conservative world-space sphere.
// Center is fully transformed; radius is scaled by the transform's largest per-axis scale
// (max row length of the 3x3 linear part) — exact for uniform scale, conservative for
// non-uniform scale/rotation (assumes no shear, true for standard TRS node transforms).
void TransformSphereToWorld(float3 localCenter, float localRadius, float4x4 localToWorld,
                            out float3 worldCenter, out float worldRadius)
{
    worldCenter = mul(float4(localCenter, 1.0f), localToWorld).xyz;
    float scaleX = length(localToWorld[0].xyz);
    float scaleY = length(localToWorld[1].xyz);
    float scaleZ = length(localToWorld[2].xyz);
    worldRadius = localRadius * max(scaleX, max(scaleY, scaleZ));
}

// Frustum culling: meshlet-space bounding sphere → world → clip, conservative test.
// Returns the shared FrustumCullData (visibility + NDC rect + nearest depth).
// See FrustumCullSphere above for why viewProj and hzbViewProj are separate.
FrustumCullData FrustumCullMeshletSphere(MeshletBounds bounds, float4x4 localToWorld,
                                         float4x4 viewProj, float4x4 hzbViewProj,
                                         float P00, float P11, float zNear)
{
    float3 worldCenter; float worldRadius;
    TransformSphereToWorld(bounds.LocalCenter, bounds.LocalRadius, localToWorld,
                           worldCenter, worldRadius);
    return FrustumCullSphere(worldCenter, worldRadius, viewProj, hzbViewProj, P00, P11, zNear);
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