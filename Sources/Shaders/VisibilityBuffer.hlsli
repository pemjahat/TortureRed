#pragma once

#include "MeshletCommon.hlsli"

// --- Visibility buffer pack/unpack ---
// candidateIndex: 1-based (0 = invalid/sky pixel)
// primitiveID: triangle index within the meshlet (0..123, 7 bits)
// Layout: [primitiveID:7] | [candidateIndex+1: rest]

bool UnpackVisBuffer(uint data, out uint candidateIndex, out uint primitiveID)
{
    primitiveID    = data & 0x7F;
    candidateIndex = data >> 7;
    candidateIndex -= 1; // Value of 0 means 'Invalid'
    return candidateIndex != 0xFFFFFFFF;
}

uint PackVisBuffer(uint candidateIndex, uint primitiveID)
{
    return primitiveID | ((candidateIndex + 1) << 7);
}

// --- Barycentric interpolation ---
template<typename T>
T BaryInterpolate(T a, T b, T c, float3 barycentrics)
{
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

// --- Barycentric derivatives (for wireframe edge detection) ---
struct BaryDerivs
{
    float3 Barycentrics;
    float3 DDX_Barycentrics;
    float3 DDY_Barycentrics;
};

// Convert UV [0,1] to clip-space [-1,1]
float2 UVToClip(float2 uv)
{
    return float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
}

// Compute barycentric coordinates from screen-space pixel position
// and the three triangle clip-space vertex positions.
// Based on D3D12_Research VisibilityBuffer.hlsli.
BaryDerivs ComputeBarycentrics(float2 pixelClip, float4 vertexCS0, float4 vertexCS1, float4 vertexCS2, float2 viewportInv)
{
    BaryDerivs result;

    float3 pos0 = vertexCS0.xyz / vertexCS0.w;
    float3 pos1 = vertexCS1.xyz / vertexCS1.w;
    float3 pos2 = vertexCS2.xyz / vertexCS2.w;

    float3 RcpW = rcp(float3(vertexCS0.w, vertexCS1.w, vertexCS2.w));

    float3 pos120X = float3(pos1.x, pos2.x, pos0.x);
    float3 pos120Y = float3(pos1.y, pos2.y, pos0.y);
    float3 pos201X = float3(pos2.x, pos0.x, pos1.x);
    float3 pos201Y = float3(pos2.y, pos0.y, pos1.y);

    float3 C_dx = pos201Y - pos120Y;
    float3 C_dy = pos120X - pos201X;

    float3 C = C_dx * (pixelClip.x - pos120X) + C_dy * (pixelClip.y - pos120Y);
    float3 G = C * RcpW;

    float H = dot(C, RcpW);
    float rcpH = rcp(H);

    result.Barycentrics = G * rcpH;

    float3 G_dx = C_dx * RcpW;
    float3 G_dy = C_dy * RcpW;

    float H_dx = dot(C_dx, RcpW);
    float H_dy = dot(C_dy, RcpW);

    result.DDX_Barycentrics = (G_dx * H - G * H_dx) * (rcpH * rcpH) * ( 2.0f * viewportInv.x);
    result.DDY_Barycentrics = (G_dy * H - G * H_dy) * (rcpH * rcpH) * (-2.0f * viewportInv.y);

    return result;
}

// --- Vertex attribute structures for debug view ---
struct VisBufferVertexAttribute
{
    float3 WorldPosition;
    float3 Normal;
    float2 UV;
    float3 Barycentrics;
    float LinearDepth;
};

// Reconstruct per-pixel vertex attributes from visibility buffer token.
// This loads the full meshlet indirection chain (candidate → instance → mesh → meshlet → triangle → vertices)
// and computes barycentrics analytically from screen-space positions.
VisBufferVertexAttribute GetVertexAttributes(
    float2 screenUV,
    float4x4 worldToClip,
    float2 viewportInv,
    StructuredBuffer<MeshletCandidate> visibleMeshlets,
    StructuredBuffer<InstanceData>   globalInstanceData,
    StructuredBuffer<MeshData>       globalMeshData,
    StructuredBuffer<Meshlet>        globalMeshlets,
    StructuredBuffer<uint>           globalMeshletVertices,
    StructuredBuffer<MeshletTriangle> globalMeshletTriangles,
    StructuredBuffer<float3>         globalPositions,
    StructuredBuffer<uint>           globalNormals,
    StructuredBuffer<uint>           globalUVs,
    uint candidateIndex,
    uint primitiveID)
{
    MeshletCandidate cand = visibleMeshlets[candidateIndex];
    InstanceData inst     = globalInstanceData[cand.InstanceID];
    MeshData md           = globalMeshData[inst.MeshDataIndex];
    Meshlet m             = globalMeshlets[md.MeshletOffset + cand.MeshletIndex];
    MeshletTriangle tri   = globalMeshletTriangles[md.MeshletTriangleOffset + m.TriangleOffset + primitiveID];

    uint3 indices = uint3(
        globalMeshletVertices[md.MeshletVertexOffset + m.VertexOffset + tri.V0],
        globalMeshletVertices[md.MeshletVertexOffset + m.VertexOffset + tri.V1],
        globalMeshletVertices[md.MeshletVertexOffset + m.VertexOffset + tri.V2]
    );

    float3 localPos[3];
    float3 worldPos[3];
    [unroll]
    for (uint i = 0; i < 3; ++i)
    {
        localPos[i]  = globalPositions[md.PositionOffset + indices[i]];
        worldPos[i]  = mul(float4(localPos[i], 1.0), inst.LocalToWorld).xyz;
    }

    float4 clipPos0 = mul(float4(worldPos[0], 1.0), worldToClip);
    float4 clipPos1 = mul(float4(worldPos[1], 1.0), worldToClip);
    float4 clipPos2 = mul(float4(worldPos[2], 1.0), worldToClip);
    float2 pixelClip = UVToClip(screenUV);
    BaryDerivs bary = ComputeBarycentrics(pixelClip, clipPos0, clipPos1, clipPos2, viewportInv);

    VisBufferVertexAttribute result;
    result.WorldPosition = BaryInterpolate(worldPos[0], worldPos[1], worldPos[2], bary.Barycentrics);
    result.Barycentrics  = bary.Barycentrics;
    result.LinearDepth   = BaryInterpolate(clipPos0.w, clipPos1.w, clipPos2.w, bary.Barycentrics);

    // For UV and normal we need the packed formats — read back from the buffers
    // (same unpacking logic as MeshletCommon.hlsli)
    float3 localNormals[3];
    float2 uvs[3];
    [unroll]
    for (uint j = 0; j < 3; ++j)
    {
        localNormals[j] = UnpackNormalRGB10A2(globalNormals, md.NormalOffset, indices[j]);
        uvs[j]          = UnpackUVRG16(globalUVs, md.UVOffset, indices[j]);
    }
    result.Normal = normalize(mul(BaryInterpolate(localNormals[0], localNormals[1], localNormals[2], bary.Barycentrics), (float3x3)inst.LocalToWorld));
    result.UV     = BaryInterpolate(uvs[0], uvs[1], uvs[2], bary.Barycentrics);

    return result;
}
