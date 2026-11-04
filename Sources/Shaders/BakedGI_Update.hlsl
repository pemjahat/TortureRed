// =============================================================================
// BakedGI_Update.hlsl — per-frame "light the probes" pass.
//
// Folds the LIVE sun/sky state into the baked transport, one thread per probe:
//
//   e_lm = sum_i wq * L_sky(W_i) * [ R_i_lm  +  A_l Y_lm(W_i) V(W_i) ]   (sky)
//        + E_sun * lerp(R[i0], R[i0+1], t)                            (sun, 2 records)
//
//   sky: wq = 2pi/16 hemisphere quadrature; live SkySH9 stores C_lm = A_l L_lm.
//        L_sky = sum_lm (C_lm / A_l) Y_lm reconstructs incident radiance.
//        R = bounces, A_l Y V = occluded first-bounce sky (Tier-2 ambient remedy).
//   sun: the 17 baked hour states bracket the current hour, and the fold reads
//        exactly the two that do. The response is INDIRECT-ONLY, so there is no
//        double count with the runtime-exact direct sun.
//
// Writes THREE buffers: the sky-sourced and sun-sourced splits, and the
// TOTAL as their exact sum (the runtime fetch reads only the total; the
// splits feed the Lit Sky / Lit Sun debug ball views).
//
// SKY VISIBILITY MODE — the TWO things this pass lets a user change. The V in
// `A_l Y_lm(W_i) V(W_i)` comes from one of two sources:
//   skyVisMode 0  the raw SH9 sky-visibility buffer, reconstructed at W_i and
//                 clamped to [0, 1]. The default.
//   skyVisMode 1  V = 0. A diagnostic: it removes the direct first-bounce sky
//                 term only. Bounces R_i and the sun term stay.
// The CPU sends 1 whenever the SH buffer does not exist (no bake yet, or a
// Probes/ cache load), so mode 0 never reaches a read of an absent buffer and
// there is no third case here.
// Everything else — R_i, the sun term, the receiver weighting, the lighting —
// is the same code on both paths.
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

    // The sky quadrature reads the 16-record table below; the sun term reads the
    // separate 17-state sun table. Two tables, one thread.
    StructuredBuffer<float4> response = ResourceDescriptorHeap[Upd.responseSRVIdx];
    StructuredBuffer<float4> skySH9   = ResourceDescriptorHeap[FrameCB.skySH9BufferIndex];
    RWStructuredBuffer<float4> lit     = ResourceDescriptorHeap[Upd.litUAVIdx];
    RWStructuredBuffer<float4> litSky  = ResourceDescriptorHeap[Upd.litSkyUAVIdx];
    RWStructuredBuffer<float4> litSun  = ResourceDescriptorHeap[Upd.litSunUAVIdx];

    // Sky-visibility mode. Load the probe's 9 coefficients ONCE, not for each
    // of the 16 directions. Mode 0 is the only mode that reads the SH buffer,
    // and the CPU sends 1 when the buffer does not exist, so a mode 0 dispatch
    // always carries a real index. Mode 1 is a scalar switch: it reads no
    // descriptor at all.
    const uint skyVisMode   = Upd.skyVisMode;
    const bool useSkyVisSH  = (skyVisMode == 0u);
    float skyVisSH[BAKED_GI_SKYVIS_SH_COEFFS] = (float[BAKED_GI_SKYVIS_SH_COEFFS])0;
    if (useSkyVisSH)
    {
        StructuredBuffer<float> shBuf = ResourceDescriptorHeap[Upd.skyVisSHSRVIdx];
        [unroll]
        for (uint c = 0; c < BAKED_GI_SKYVIS_SH_COEFFS; ++c)
            skyVisSH[c] = shBuf[probe * BAKED_GI_SKYVIS_SH_COEFFS + c];
    }

    float3 sunE   = float3(Upd.sunIrradianceX, Upd.sunIrradianceY, Upd.sunIrradianceZ);

    float3 coefSky[9] = (float3[9])0;
    float3 coefSun[9] = (float3[9])0;

    // --- Sky: quadrature over the hemisphere direction set ---
    float skyWeight = 2.0f * PI / (float)BAKED_GI_DIRECTIONS;
    [unroll]
    for (uint i = 0; i < BAKED_GI_DIRECTIONS; ++i)
    {
        float3 wi = BakedGIDeltaDirection(i);
        float basis[9];
        EvalSH9Basis(wi, basis);

        // The buffer stores A_l * L_lm. Divide by A_l to recover radiance;
        // the transport below owns the only cosine convolution.
        float3 Lsky = float3(0.0f, 0.0f, 0.0f);
        [unroll]
        for (uint skyLm = 0; skyLm < 9; ++skyLm)
            Lsky += skySH9[skyLm].rgb * basis[skyLm] / BakedGIA_l(skyLm);

        uint slot = (probe * BAKED_GI_DIRECTIONS + i) * BAKED_GI_RESPONSE_F4;
        // V — the direct sky visibility at W_i. It is reconstructed from the
        // probe's raw SH9 field with the same basis array that reconstructs Lsky
        // above, then clamped: a band-limited fit overshoots [0, 1] and a
        // visibility fraction must stay inside it. Mode 1 leaves V at zero and
        // reads no descriptor. skyVisMode is uniform for the whole dispatch, so
        // this branch does not diverge inside a wave.
        float V = 0.0f;
        if (useSkyVisSH)
        {
            float vSH = 0.0f;
            [unroll]
            for (uint vlm = 0; vlm < BAKED_GI_SKYVIS_SH_COEFFS; ++vlm)
                vSH += skyVisSH[vlm] * basis[vlm];
            V = saturate(vSH);
        }

        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
        {
            float3 Rbounce = response[slot + lm].rgb;                    // sky bounces
            float3 Rdirect = BakedGIA_l(lm) * basis[lm] * V;             // occluded first-bounce sky
            coefSky[lm] += skyWeight * Lsky * (Rbounce + Rdirect);
        }
    }

    // --- Sun: TWO records, the pair that brackets the current hour ---
    //
    // The 17 states sit on the equinox arc at 0.75 h intervals: state k is at
    // hour 6 + 0.75 k. A given hour can only be between two of them, so the fold
    // reads two records and never the whole set. The sky loop above is a
    // different integral and still reads all 16 of its own records.
    //
    // The weight is LINEAR, and that is a decision rather than a shortcut. The
    // equinox moves the sun at exactly 15.00 degrees each hour, so a linear weight
    // in hours is also a linear weight in angle: time and angle are the same
    // parameter here. A power kernel would place almost all the weight on one
    // state and pop at every midpoint. Do not reintroduce one.
    {
        StructuredBuffer<float4> sunResp = ResourceDescriptorHeap[Upd.sunResponseSRVIdx];
        const float u  = (Upd.hour - BAKED_GI_SUN_STATE_HOUR0) * (1.0f / BAKED_GI_SUN_STATE_STEP);
        // The clamp is what makes the night case safe with no branch. Below hour 6
        // and above hour 18 the two end states are used and sunE is already near
        // zero, so the term fades to nothing without a test and without a step.
        const int   i0 = (int)clamp(floor(u), 0.0f, (float)(BAKED_GI_SUN_STATES - 2));
        const float t  = saturate(u - (float)i0);
        const uint  base = (probe * BAKED_GI_SUN_STATES + (uint)i0) * BAKED_GI_RESPONSE_F4;
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
        {
            const float3 a = sunResp[base + lm].rgb;
            const float3 b = sunResp[base + BAKED_GI_RESPONSE_F4 + lm].rgb;
            coefSun[lm] += sunE * lerp(a, b, t);
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
