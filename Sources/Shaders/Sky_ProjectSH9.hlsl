// Sky_ProjectSH9.hlsl
//
// Single-dispatch compute shader that projects the baked Hosek-Wilkie cubemap
// onto order-3 Spherical Harmonics (9 RGB coefficients).
//
// Follows "Stupid SH Tricks" (Sloan 2008): accumulate radiance * SH basis *
// per-texel weight w = 4/(1+u²+v²)^(3/2), then normalize by 4π / Σw so the
// total integral equals the sphere area — self-correcting regardless of
// cubemap resolution.
//
// Inputs (via ResourceDescriptorHeap bindless indices from FrameConstants):
//   skyCubemapIndex   — SRV: TextureCube<float4> (baked radiance)
//   skySH9BufferIndex — UAV: RWStructuredBuffer<float4>, 9 elements

#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  FrameCB   : register(b0);

#define SKY_CUBEMAP_SIZE  128
#define TOTAL_TEXELS      (SKY_CUBEMAP_SIZE * SKY_CUBEMAP_SIZE * 6)

// SH9 irradiance basis: A_l * Y_lm(dir) where A_l are Lambertian cosine
// convolution coefficients (A0=pi, A1=2pi/3, A2=pi/4).
void EvalSH9IrradianceBasis(float3 N, out float sh[9])
{
    sh[0] = 0.8862269254527579f; // Y00 * pi = sqrt(pi)/2

    // Band 1: Y1m * 2*pi/3
    float a1 = 2.0f * 3.14159265f / 3.0f * sqrt(3.0f / (4.0f * 3.14159265f));
    sh[1] = a1 * N.y;
    sh[2] = a1 * N.z;
    sh[3] = a1 * N.x;

    // Band 2: Y2m * pi/4
    float a2 = 3.14159265f / 4.0f * 0.5f * sqrt(15.0f / 3.14159265f);
    sh[4] = a2 * N.x * N.y;
    sh[5] = a2 * N.y * N.z;
    sh[6] = 3.14159265f/4.0f * 0.25f * sqrt(5.0f/3.14159265f) * (3.0f*N.z*N.z - 1.0f);
    sh[7] = a2 * N.x * N.z;
    sh[8] = a2 * 0.5f * (N.x*N.x - N.y*N.y);
}

// Per-texel weight from "Stupid SH Tricks": w = 4 / (1+u²+v²)^(3/2)
float CubemapTexelWeight(float u, float v)
{
    float denom = 1.0f + u*u + v*v;
    return 4.0f / (denom * sqrt(denom));
}

// Group-shared memory for reduction: 64 threads × (9 coeffs × 3 channels + 1 weight)
groupshared float gs_AccumR[64][9];
groupshared float gs_AccumG[64][9];
groupshared float gs_AccumB[64][9];
groupshared float gs_Weight[64];

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    TextureCube<float4> skyCubemap = ResourceDescriptorHeap[FrameCB.skyCubemapIndex];

    // Build face basis vectors for direction computation.
    float3 faceDir[6] = {
        float3(1,0,0), float3(-1,0,0),
        float3(0,1,0), float3(0,-1,0),
        float3(0,0,1), float3(0,0,-1)
    };
    float3 faceU[6] = {
        float3(0,0,-1), float3(0,0,1),
        float3(1,0,0),  float3(1,0,0),
        float3(1,0,0),  float3(-1,0,0)
    };
    float3 faceV[6] = {
        float3(0,-1,0), float3(0,-1,0),
        float3(0,0,1),  float3(0,0,-1),
        float3(0,-1,0), float3(0,-1,0)
    };

    // Per-thread local accumulation.
    float localR[9] = {0,0,0,0,0,0,0,0,0};
    float localG[9] = {0,0,0,0,0,0,0,0,0};
    float localB[9] = {0,0,0,0,0,0,0,0,0};
    float localWeightSum = 0.0f;

    uint texelsPerThread = (TOTAL_TEXELS + 63) / 64;
    uint startTexel = groupIndex * texelsPerThread;
    uint endTexel   = min(startTexel + texelsPerThread, TOTAL_TEXELS);

    for (uint idx = startTexel; idx < endTexel; ++idx)
    {
        uint face     = idx / (SKY_CUBEMAP_SIZE * SKY_CUBEMAP_SIZE);
        uint faceIdx  = idx % (SKY_CUBEMAP_SIZE * SKY_CUBEMAP_SIZE);
        uint py       = faceIdx / SKY_CUBEMAP_SIZE;
        uint px       = faceIdx % SKY_CUBEMAP_SIZE;

        float u = ((float)px + 0.5f) / (float)SKY_CUBEMAP_SIZE;
        float v = ((float)py + 0.5f) / (float)SKY_CUBEMAP_SIZE;
        float uc = 2.0f * u - 1.0f;
        float vc = 2.0f * v - 1.0f;

        float3 dir = normalize(faceDir[face] + faceU[face] * uc + faceV[face] * vc);
        float weight = CubemapTexelWeight(uc, vc);
        float3 radiance = skyCubemap.SampleLevel(g_LinearSampler, dir, 0.0f).rgb;

        float shBasis[9];
        EvalSH9IrradianceBasis(dir, shBasis);

        for (int i = 0; i < 9; ++i)
        {
            float contrib = shBasis[i] * weight;
            localR[i] += radiance.r * contrib;
            localG[i] += radiance.g * contrib;
            localB[i] += radiance.b * contrib;
        }
        localWeightSum += weight;
    }

    // Write to group-shared memory.
    for (int i = 0; i < 9; ++i)
    {
        gs_AccumR[groupIndex][i] = localR[i];
        gs_AccumG[groupIndex][i] = localG[i];
        gs_AccumB[groupIndex][i] = localB[i];
    }
    gs_Weight[groupIndex] = localWeightSum;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction (binary tree, collapse to thread 0).
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (groupIndex < stride)
        {
            for (int i = 0; i < 9; ++i)
            {
                gs_AccumR[groupIndex][i] += gs_AccumR[groupIndex + stride][i];
                gs_AccumG[groupIndex][i] += gs_AccumG[groupIndex + stride][i];
                gs_AccumB[groupIndex][i] += gs_AccumB[groupIndex + stride][i];
            }
            gs_Weight[groupIndex] += gs_Weight[groupIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Thread 0 normalizes and writes the final result.
    if (groupIndex == 0)
    {
        // Normalize so total integral = 4π  (Stupid SH Tricks equation).
        float totalWeight = gs_Weight[0];
        float norm = (totalWeight > 0.0f) ? (4.0f * 3.14159265f / totalWeight) : 0.0f;

        RWStructuredBuffer<float4> skySH9Buf = ResourceDescriptorHeap[FrameCB.skySH9BufferIndex];
        for (int i = 0; i < 9; ++i)
        {
            skySH9Buf[i] = float4(gs_AccumR[0][i] * norm,
                                  gs_AccumG[0][i] * norm,
                                  gs_AccumB[0][i] * norm, 0.0f);
        }
    }
}
