// =============================================================================
// BakedGI_IterFold.hlsl — per-bake-term response ball (debug view 8).
//
// ONE thread for each probe. Reads ONE bake term's transport table (10 float4
// for each (probe, direction), BakedGI.hlsli:24) and writes the per-probe SH9
// "response ball" of that term, the 16-direction fold
//
//   ball_lm = sum_i wq * R_i,lm ,      wq = 2 pi / 16
//
// wq and the SH basis are the update pass's sky quadrature
// (BakedGI_Update.hlsl:49-68) with the sky radiance set to 1 and WITHOUT its
// V_i first-bounce term: this ball describes the series response alone. R
// already carries the cosine band factors A_l (the bake stores
// `acc * norm * BakedGIA_l`, BakedGI_Bake.hlsl:254), so no second A_l enters
// and the ball is an irradiance-response field, in the table's own units.
//
// WHY THE FOLD IS ITS OWN PASS. The ball is a function of the BAKED table
// alone — no sun, no sky, no per-frame state — so it is computed once, on the
// bake command list, and it costs the frame path nothing. Folding it in the
// debug view's vertex shader instead would need 16 x 9 reads for each of the
// 384 sphere vertices of every probe.
//
// WHAT THE DEBUG VIEW DOES WITH IT. Per term k the bake contribution is
//
//   R^k - R^0 = G(Efb_k)
//
// because the gather is linear in Efb and Efb never changes a ray direction,
// the shadow-ray gate or an alpha-mask draw — the ray/hit stream is identical
// in every term. The fold is linear too, so the same identity holds on the
// balls, and the view differences ball[k] - ball[1] and needs nothing else.
//
// It rides the bake's root bindings unchanged: b0 FrameCB (BakedGI.hlsli
// references FrameCB in its unused fetch helpers, so the declaration must
// precede the include) and the shared b2 root-constants block.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>        FrameCB : register(b0);
ConstantBuffer<BakedGIIterFoldParams> Fold    : register(b2);

// BakedGI.hlsli references FrameCB — include after the declarations above.
#include "BakedGI.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint probe = dtid.x;
    if (probe >= Fold.probeCount)
        return;

    // No validity gate: an invalid probe's stored row is already all zeroes
    // (BakedGI_Bake.hlsl:147-155), so its ball is zero and the view's subtract
    // yields zero. Reading meta here would add a buffer dependency for nothing.
    StructuredBuffer<float4>   table = ResourceDescriptorHeap[Fold.tableSRVIdx];
    RWStructuredBuffer<float4> ball  = ResourceDescriptorHeap[Fold.ballUAVIdx];

    const float wq = 2.0f * PI / (float)BAKED_GI_DIRECTIONS;

    float3 coef[9] = (float3[9])0;
    [unroll]
    for (uint i = 0; i < BAKED_GI_DIRECTIONS; ++i)
    {
        const uint slot = (probe * BAKED_GI_DIRECTIONS + i) * BAKED_GI_RESPONSE_F4;
        [unroll]
        for (uint lm = 0; lm < 9; ++lm)
            coef[lm] += wq * table[slot + lm].rgb;
    }

    // 9 float4 for each probe — the lit buffer's layout (BakedGI::kLitFloat4s),
    // so the debug view evaluates this ball with the code that evaluates the
    // lit balls. The w channel is unused there and stays 0.
    [unroll]
    for (uint lm2 = 0; lm2 < 9; ++lm2)
        ball[probe * 9u + lm2] = float4(coef[lm2], 0.0f);
}
