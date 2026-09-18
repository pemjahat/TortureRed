#ifndef BAKED_GI_HLSLI
#define BAKED_GI_HLSLI

// =============================================================================
// BakedGI.hlsli — shared helpers for the baked GI probe system.
//
// Representation :
//   Per probe and per delta direction W_i (16 fibonacci-hemisphere dirs), the
//   bake stores the INDIRECT-ONLY directional-irradiance SH9 RGB response to a
//   unit delta light at W_i, plus the direct-delta visibility V_i. Both the sun
//   and the sky recombine from the same table at runtime:
//     e_sky = sum_i wq * L_sky(W_i) * [ R_i  +  A_l Y_lm(W_i) V_i ]
//             (sky bounces)              (occluded first-bounce sky)
//     e_sun = E_sun * interp_j(R_j)      (sun bounces only — receiver direct
//                                          sun stays with the runtime-exact path)
// =============================================================================

#define BAKED_GI_DIRECTIONS    16
#define BAKED_GI_RESPONSE_F4   10  // float4s per (probe, dir): [0..8] = SH9 RGB lm coeffs, [9].x = V_i

// Fibonacci-hemisphere direction set (deterministic — bake and update MUST agree).
float3 BakedGIDeltaDirection(uint i)
{
    float y = ((float)i + 0.5f) / (float)BAKED_GI_DIRECTIONS;
    float phi = (float)i * 2.39996323f; // golden angle
    float r = sqrt(max(0.0f, 1.0f - y * y));
    return normalize(float3(r * cos(phi), y, r * sin(phi)));
}

// Lambertian cosine-convolution band factors (Ramamoorthi 2001): A0 = pi,
// A1 = 2pi/3, A2 = pi/4 — matches Sky_ProjectSH9.hlsl's irradiance basis.
// Takes the SH9 COEFFICIENT index lm (0..8): band 0 = {0}, band 1 = {1,2,3},
// band 2 = {4..8} (EvalSH9Basis ordering).
float BakedGIA_l(uint lm)
{
    const uint band = (lm <= 0u) ? 0u : ((lm <= 3u) ? 1u : 2u);
    return (band == 0u) ? PI : ((band == 1u) ? 2.09439510f : 0.78539816f);
}

// Flat probe-cell index from a 3D cell (row-major XYZ).
uint BakedGIFlatten(uint3 c, uint3 dims)
{
    return (c.z * dims.y + c.y) * dims.x + c.x;
}

// ---------------------------------------------------------------------------
// Deferred fetch (Lighting.hlsl): manual trilinear over the lit probe grid
// with VALIDITY RENORMALIZATION — invalid (in-geometry) probes are dropped
// from the blend and the weights renormalized, so they do not darken their
// neighbors.
//
// REQUIRES in scope at include time:
//   - ConstantBuffer<FrameConstants> FrameCB : register(b0)
//   - EvalSH9Basis (PBR.hlsl, included via Common.hlsl / CommonTracing.hlsl)
// ---------------------------------------------------------------------------
float3 SampleBakedGIProbe(float3 worldPos, float3 N)
{
    uint3  dims    = uint3(FrameCB.bakedGIDimX, FrameCB.bakedGIDimY, FrameCB.bakedGIDimZ);
    float3 gridMin = float3(FrameCB.bakedGIGridMinX, FrameCB.bakedGIGridMinY, FrameCB.bakedGIGridMinZ);

    // Continuous cell coordinates; probes live on integer cells in [0, dim-1].
    float3 t = clamp((worldPos - gridMin) / FrameCB.bakedGISpacing,
                     float3(0.0f, 0.0f, 0.0f), dims - 1.0f);
    int3   base = int3(min(floor(t), max(dims - 2.0f, float3(0.0f, 0.0f, 0.0f))));
    float3 f    = t - base;

    StructuredBuffer<float4> lit  = ResourceDescriptorHeap[FrameCB.bakedGIProbeSRVIndex];
    StructuredBuffer<uint>   meta = ResourceDescriptorHeap[FrameCB.bakedGIProbeMetaSRVIndex];

    float3 coef[9] = (float3[9])0;
    float  wsum = 0.0f;

    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        int3  off = int3(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        float3 w3 = lerp(float3(1.0f, 1.0f, 1.0f) - f, f, off);
        float w = w3.x * w3.y * w3.z;

        uint flat = BakedGIFlatten(base + off, dims);
        if (meta[flat] & 1u)
        {
            [unroll]
            for (uint lm = 0; lm < 9; ++lm)
                coef[lm] += lit[flat * 9u + lm].rgb * w;
            wsum += w;
        }
    }

    if (wsum < 1e-6f)
        return float3(0.0f, 0.0f, 0.0f);

    // Evaluate directional irradiance at N (lit coefficients use the same
    // plain-SH convention as the sky SH9 buffer).
    float basis[9];
    EvalSH9Basis(N, basis);
    float3 E = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
    {
        coef[lm2] /= wsum;
        E += coef[lm2] * basis[lm2];
    }
    return max(E, 0.0f.xxx);
}

#endif // BAKED_GI_HLSLI
