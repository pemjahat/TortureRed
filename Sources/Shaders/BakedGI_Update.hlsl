// =============================================================================
// BakedGI_Update.hlsl — per-frame "light the probes" pass.
//
// Folds the LIVE sun/sky state into the baked transport, one thread per probe:
//
//   e_lm = sum_i wq * L_sky(W_i) * [ R_i_lm  +  A_l Y_lm(W_i) V_i ]   (sky)
//        + E_sun * sum_j b_j(W_sun) * R_j_lm                            (sun)
//
//   sky: wq = 2pi/16 hemisphere quadrature; L_sky sampled from the baked
//        Hosek-Wilkie cubemap (Tier 1); R = bounces, A_l Y V = occluded
//        first-bounce sky (remedy for the disabled Tier-2 ambient).
//   sun: cos^8-kernel interpolation over the 16 directions (the response is
//        INDIRECT-ONLY, so there is no double count with the runtime-exact
//        direct sun; below-horizon sun fades out with the kernel weights).
//
// Writes THREE buffers: the sky-sourced and sun-sourced splits, and the
// TOTAL as their exact sum (the runtime fetch reads only the total; the
// splits feed the Lit Sky / Lit Sun debug ball views).
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>       FrameCB : register(b0);
ConstantBuffer<BakedGIUpdateParams>  Upd     : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

// Sun kernel exponent. The weight falls to half at acos(0.5^(1/64)), about 8.4
// degrees, against 23.5 at exponent 8. Sharper localises a narrow light path and
// cuts the weight that leaks onto unlit bins; too sharp approaches nearest-bin
// selection and pops at a bin boundary. Measured 2026-10-27: argmax pops by the
// full bin-to-bin contrast 2.28 degrees from the captured sun, exponent 64 steps
// 0.17 percent and recovers 85 percent of the gap to the exact single-direction
// bake.
//
// MUST match kBakedGISunKernelExponent in BakedGI.cpp. The CPU mirror of this
// kernel publishes bakedGISunWeight, which the Ray Fan and the bin panel read, so
// a mismatch would make the picture and the numbers describe different folds.
#define BAKED_GI_SUN_KERNEL_EXPONENT 64.0f

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint3 dims = uint3(Upd.dimX, Upd.dimY, Upd.dimZ);
    uint probeCount = dims.x * dims.y * dims.z;
    uint probe = dtid.x;
    if (probe >= probeCount)
        return;

    StructuredBuffer<float4> response = ResourceDescriptorHeap[Upd.responseSRVIdx];
    RWStructuredBuffer<float4> lit     = ResourceDescriptorHeap[Upd.litUAVIdx];
    RWStructuredBuffer<float4> litSky  = ResourceDescriptorHeap[Upd.litSkyUAVIdx];
    RWStructuredBuffer<float4> litSun  = ResourceDescriptorHeap[Upd.litSunUAVIdx];

    float3 sunDir = normalize(float3(Upd.sunDirX, Upd.sunDirY, Upd.sunDirZ));
    float3 sunE   = float3(Upd.sunIrradianceX, Upd.sunIrradianceY, Upd.sunIrradianceZ);

    float3 coefSky[9] = (float3[9])0;
    float3 coefSun[9] = (float3[9])0;

    // --- Sky: quadrature over the hemisphere direction set ---
    float skyWeight = 2.0f * PI / (float)BAKED_GI_DIRECTIONS;
    [unroll]
    for (uint i = 0; i < BAKED_GI_DIRECTIONS; ++i)
    {
        float3 wi   = BakedGIDeltaDirection(i);
        float3 Lsky = SampleSky(wi, FrameCB.skyCubemapIndex);

        uint slot = (probe * BAKED_GI_DIRECTIONS + i) * BAKED_GI_RESPONSE_F4;
        float basis[9];
        EvalSH9Basis(wi, basis);
        float V = response[slot + 9].x;

        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
        {
            float3 Rbounce = response[slot + lm].rgb;                    // sky bounces
            float3 Rdirect = BakedGIA_l(lm) * basis[lm] * V;             // occluded first-bounce sky
            coefSky[lm] += skyWeight * Lsky * (Rbounce + Rdirect);
        }
    }

    // --- Sun: kernel-interpolated indirect response x sun irradiance ---
    //
    // TWO SOURCES, chosen by the method baked into this table:
    //
    //   sunMode 0 (Probe)      : the 16-bin cos^8 fold. Each bin is the response
    //                            to a delta light from W_j, and the kernel mixes
    //                            them to approximate the response at the sun. The
    //                            mix is sound for smooth radiance and wrong for
    //                            visibility: the weighted answer sits 23 degrees
    //                            from the sun on average, so a light path narrower
    //                            than about 15 degrees reads as fully lit or fully
    //                            shadowed with no partial credit.
    //
    //   sunMode 1 (SingleDir)  : ONE term, no kernel. The bake traced visibility
    //                            toward the real sun, so the response already
    //                            describes the sun and there is nothing to
    //                            interpolate. This is the whole point of the method.
    if (Upd.sunMode != 0u)
    {
        // One record per probe: no bin index, so the stride is kResponseFloat4s
        // rather than BAKED_GI_DIRECTIONS * kResponseFloat4s.
        StructuredBuffer<float4> sunResp = ResourceDescriptorHeap[Upd.sunResponseSRVIdx];
        const uint sunSlot = probe * BAKED_GI_RESPONSE_F4;
        [unroll]
        for (uint lms = 0; lms < 9; ++lms)
            coefSun[lms] += sunE * sunResp[sunSlot + lms].rgb;
    }
    else
    {
        float bw[BAKED_GI_DIRECTIONS];
        float wsum = 0.0f;
        [unroll]
        for (uint j = 0; j < BAKED_GI_DIRECTIONS; ++j)
        {
            bw[j] = pow(max(0.0f, dot(BakedGIDeltaDirection(j), sunDir)),
                        BAKED_GI_SUN_KERNEL_EXPONENT);
            wsum += bw[j];
        }
        if (wsum > 1e-6f)
        {
            // Kernel-interpolated response. The visibility stays INSIDE the stored
            // table: the bake multiplied each bin by the visibility of its own
            // delta direction, so this blend sees the occluded value, as shipped.
            // Do not separate it here. A per-direction scalar visibility cannot be
            // applied after the SH projection without destroying the directional
            // structure, and a band-limited SH fit of that field overshoots by
            // about 28 percent at a 45-60 degree light path.
            //
            // SAMPLING FLOOR, which no estimator fixes: a 256-ray bin integrates
            // 4pi/256 = 0.049 sr per ray, so a light path of half-angle H holds
            // 2pi(1-cos H)/0.049 rays. Measured: 5 degrees -> 0 rays, 10 degrees ->
            // about 2, 15 degrees -> about 4. Under 10 degrees a bin may hold no
            // sun-visible hit at all, so a window seen from across a hall sits in
            // the band this CANNOT fix.
            [unroll]
            for (uint j2 = 0; j2 < BAKED_GI_DIRECTIONS; ++j2)
            {
                if (bw[j2] <= 0.0f)
                    continue;
                const float b = bw[j2] / wsum;
                const uint slot = (probe * BAKED_GI_DIRECTIONS + j2) * BAKED_GI_RESPONSE_F4;
                [unroll]
                for (uint lm = 0; lm < 9; ++lm)
                    coefSun[lm] += sunE * b * response[slot + lm].rgb;
            }
        }
    }

    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
    {
        // Total = sky + sun, computed as the exact same sum the debug views
        // display — the three balls always add up.
        litSky[probe * 9u + lm2] = float4(coefSky[lm2], 0.0f);
        litSun[probe * 9u + lm2] = float4(coefSun[lm2], 0.0f);
        lit[probe * 9u + lm2]    = float4(coefSky[lm2] + coefSun[lm2], 0.0f);
    }
}
