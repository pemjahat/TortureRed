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
    float bw[BAKED_GI_DIRECTIONS];
    float wsum = 0.0f;
    [unroll]
    for (uint j = 0; j < BAKED_GI_DIRECTIONS; ++j)
    {
        bw[j] = pow(max(0.0f, dot(BakedGIDeltaDirection(j), sunDir)), 8.0f);
        wsum += bw[j];
    }
    if (wsum > 1e-6f)
    {
        [unroll]
        for (uint j2 = 0; j2 < BAKED_GI_DIRECTIONS; ++j2)
        {
            if (bw[j2] <= 0.0f)
                continue;
            float b = bw[j2] / wsum;
            uint slot = (probe * BAKED_GI_DIRECTIONS + j2) * BAKED_GI_RESPONSE_F4;
            [unroll]
            for (uint lm = 0; lm < 9; ++lm)
                coefSun[lm] += sunE * b * response[slot + lm].rgb;
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
