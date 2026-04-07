// KajiyaTaa_Resolve.hlsl
// Kajiya-style TAA Resolve (single pass, output resolution)
//
// Key differences from NaiveTsr_Resolve:
//   - Operates in YCbCr perceptual space (luminance-weighted clamping)
//   - Blurred history (5x5 bilateral) separates low/high frequency detail
//   - Detail preservation: high-freq history detail selectively re-added after clamping
//   - Confidence-based box expansion: wider box when input is stable
//   - Smooth variance history (ping-pong) for temporal variance tracking
//
// Inputs (via bindless):
//   InputIdx0  = current_frame_tex       (SRV, internal-res, HDR)
//   InputIdx1  = reprojected_history_tex (SRV, output-res, R16G16B16A16_FLOAT, rgb+coverage)
//   InputIdx2  = closest_velocity_tex    (SRV, output-res, R16G16_FLOAT)
//   InputIdx3  = smooth_var_history_tex  (SRV, output-res, R16_FLOAT, luminance variance)
// Outputs (via bindless):
//   OutputIdx0 = history_tex             (UAV, output-res, R16G16B16A16_FLOAT, rgb+coverage)
//   OutputIdx1 = output_tex              (UAV, output-res, R8G8B8A8_UNORM, display)
//   OutputIdx2 = smooth_var_output_tex   (UAV, output-res, R16_FLOAT)

#include "Common.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

SamplerState g_LinearClamp : register(s0);

#define TARGET_SAMPLE_COUNT 8

// ---- Color space helpers ----
// Perceptual nonlinearity: sqrt(max_comp) weighting (matches kajiya TAA_NONLINEARITY_TYPE 1 + TAA_COLOR_MAPPING_MODE 1)
float linear_to_perceptual(float a) { return sqrt(max(0.0, a)); }
float perceptual_to_linear(float a) { return a * a; }

float3 decode_rgb(float3 v)
{
    float mc = max(v.r, max(v.g, v.b));
    return v * linear_to_perceptual(mc) / max(1e-20, mc);
}
float3 encode_rgb(float3 v)
{
    float mc = max(v.r, max(v.g, v.b));
    return v * perceptual_to_linear(mc) / max(1e-20, mc);
}

float sRGB_to_luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float3 sRGB_to_YCbCr(float3 c)
{
    float Y  =  0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    float Cb = -0.1146 * c.r - 0.3854 * c.g + 0.5000 * c.b;
    float Cr =  0.5000 * c.r - 0.4542 * c.g - 0.0458 * c.b;
    return float3(Y, Cb, Cr);
}
float3 YCbCr_to_sRGB(float3 ycc)
{
    float Y = ycc.x, Cb = ycc.y, Cr = ycc.z;
    return float3(
        Y + 1.5748 * Cr,
        Y - 0.1873 * Cb - 0.4681 * Cr,
        Y + 1.8556 * Cb
    );
}

// Remap HDR input -> perceptual YCbCr for TAA processing
float3 input_remap(float3 hdr)  { return sRGB_to_YCbCr(decode_rgb(hdr)); }
// History is stored as linear HDR; remap the same way
float3 history_remap(float3 v)  { return sRGB_to_YCbCr(decode_rgb(v)); }

// ---- Blurred history fetch (5x5 Gaussian bilateral on luminance) ----
float3 fetch_blurred_history(Texture2D<float4> history_tex, int2 px)
{
    const float3 center = history_tex[px].rgb;
    float4 csum = 0;
    float  wsum = 0;
    int k = 2;
    float sigma = 1.0;
    for (int y = -k; y <= k; ++y)
    {
        for (int x = -k; x <= k; ++x)
        {
            float4 c = history_tex[px + int2(x, y)];
            float2 offset = float2(x, y) * sigma;
            float w = exp(-dot(offset, offset));
            csum += c * w;
            wsum += w;
        }
    }
    return (csum / wsum).rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadID.xy;
    if (px.x >= g_Frame.outputWidth || px.y >= g_Frame.outputHeight) return;

    Texture2D<float4> current_frame_tex       = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4> reprojected_history_tex = ResourceDescriptorHeap[g_Indices.InputIdx1];
    Texture2D<float2> closest_velocity_tex    = ResourceDescriptorHeap[g_Indices.InputIdx2];
    Texture2D<float>  smooth_var_history_tex  = ResourceDescriptorHeap[g_Indices.InputIdx3];
    RWTexture2D<float4> history_tex_out       = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> output_tex            = ResourceDescriptorHeap[g_Indices.OutputIdx1];
    RWTexture2D<float>  smooth_var_output_tex = ResourceDescriptorHeap[g_Indices.OutputIdx2];

    float2 output_size   = float2(g_Frame.outputWidth,   g_Frame.outputHeight);
    float2 internal_size = float2(g_Frame.internalWidth, g_Frame.internalHeight);
    float2 input_resolution_scale = internal_size / output_size;

    float2 uv            = (px + 0.5) / output_size;
    float2 motion_vector = closest_velocity_tex[px];
    float2 history_uv    = uv + motion_vector;
    bool   history_valid = all(history_uv == saturate(history_uv));

    // ---- Read reprojected history ----
    float4 history_packed   = reprojected_history_tex[px];
    float  history_coverage = max(0.0, history_packed.a);

    // Sharp history in YCbCr perceptual space
    float3 history  = history_remap(history_packed.rgb);

    // Blurred history (low-frequency) — 5x5 Gaussian on reprojected history
    float3 bhistory_raw = fetch_blurred_history(reprojected_history_tex, (int2)px);
    float3 bhistory     = history_remap(bhistory_raw);

    // ---- Sample current frame with unjitter (3x3 Gaussian, internal-res) ----
    float2 jitter                  = g_Frame.taaJitter;
    float2 dst_unjittered_internal = (px + 0.5) * input_resolution_scale - jitter;
    int2   base_src_px             = int2(dst_unjittered_internal);

    float4 color_sum  = 0;
    float  wt_sum     = 0;
    float3 ex         = 0;
    float3 ex2        = 0;
    float  dev_wt_sum = 0;

    // Wider-kernel accumulator for bcenter (low-frequency current frame)
    float4 bcenter_sum  = 0;
    float  bcenter_wsum = 0;

    int k = 1;
    for (int y = -k; y <= k; ++y)
    {
        for (int x = -k; x <= k; ++x)
        {
            int2   src_px = base_src_px + int2(x, y);
            float3 mapped = input_remap(current_frame_tex[src_px].rgb);

            float2 src_center_internal = src_px + 0.5;
            float2 sample_offset       = src_center_internal - dst_unjittered_internal;
            float  dist2               = dot(sample_offset, sample_offset);

            // Tight kernel (matches kajiya: exp2(-10 * dist2 * scale))
            float wt     = exp2(-10.0 * dist2);
            // Wider kernel for variance estimation
            float dev_wt = exp2(-dist2);
            // Wider kernel (scale=0.333) for bcenter (low-frequency current frame)
            float bwt    = exp2(-10.0 * dist2 * (0.333 * 0.333));

            color_sum   += float4(mapped, 1) * wt;
            wt_sum      += wt;

            ex          += mapped * dev_wt;
            ex2         += mapped * mapped * dev_wt;
            dev_wt_sum  += dev_wt;

            bcenter_sum  += float4(mapped, 1) * bwt;
            bcenter_wsum += bwt;
        }
    }

    float  coverage = wt_sum;
    float3 center   = color_sum.rgb / max(wt_sum, 1e-6);
    float3 bcenter  = bcenter_sum.rgb / max(bcenter_wsum, 1e-6);

    ex  /= max(dev_wt_sum, 1e-6);
    ex2 /= max(dev_wt_sum, 1e-6);
    float3 var       = max(0.0, ex2 - ex * ex);
    float3 input_dev = sqrt(var);

    // ---- Smooth variance: blend current with history ----
    float prev_var_lum   = smooth_var_history_tex.SampleLevel(g_LinearClamp, history_uv, 0);
    float smooth_var_lum = lerp(prev_var_lum, var.x, 0.1);

    // ---- Blend history toward bcenter when coverage is low (first frames) ----
    history  = lerp(history,  bcenter, saturate(1.0 - history_coverage));
    bhistory = lerp(bhistory, bcenter, saturate(1.0 - history_coverage));

    // ---- Neighborhood clamping with detail preservation ----
    float3 clamped_history;

    // Confidence: low variance -> stable input -> wider clamping box
    float input_confidence = saturate(1.0 - sqrt(smooth_var_lum) / max(1e-3, abs(ex.x) + 0.1));
    float box_n_deviations = lerp(0.8, 3.0, input_confidence);

    float3 nmin = ex - input_dev * box_n_deviations;
    float3 nmax = ex + input_dev * box_n_deviations;

    float3 clamped_bhistory = clamp(bhistory, nmin, nmax);

    // Clamping event magnitude (used for coverage dampening on upsampling)
    float clamping_event = length(max(0.0, max(bhistory - nmax, nmin - bhistory)) / max(0.01, abs(ex) + 1e-5));

    // Outlier detection: sharp vs blurry history
    float3 outlier3  = max(0.0, max(nmin - history,  history  - nmax) / (0.1 + max(max(abs(history),  abs(ex)), 1e-5)));
    float3 boutlier3 = max(0.0, max(nmin - bhistory, bhistory - nmax) / (0.1 + max(max(abs(bhistory), abs(ex)), 1e-5)));
    float  outlier   = max(outlier3.x,  max(outlier3.y,  outlier3.z));
    float  boutlier  = max(boutlier3.x, max(boutlier3.y, boutlier3.z));

    if (history_valid)
    {
        // Non-disoccluding outliers: sharp history has detail that blurry doesn't
        float non_disoccluding_outliers = max(0.0, outlier - boutlier) * 10.0;

        float3 unclamped_history_detail = history - clamped_bhistory;

        // Temporal clamping detail: peaks when disocclusion happens
        float temporal_clamping_detail = length(unclamped_history_detail.x / max(1e-3, input_dev.x)) * 0.05;
        float temporal_stability       = saturate(1.0 - temporal_clamping_detail);

        float allow_unclamped_detail = saturate(non_disoccluding_outliers) * temporal_stability;

        // High-frequency detail from sharp history
        float3 history_detail = history - bhistory;
        history_detail = lerp(history_detail, unclamped_history_detail, allow_unclamped_detail);

        // How much clamping happened in blurry history (0..1)
        float3 bclamp_delta = clamped_bhistory - bhistory;
        float3 bcenter_delta = bcenter - bhistory;
        float  bclamp_len = length(bclamp_delta);
        float  bcenter_len = length(bcenter_delta);
        float  initial_bclamp_amount = saturate(dot(bclamp_delta, bcenter_delta) / max(1e-5, bclamp_len * bcenter_len));

        float effective_clamp_amount = saturate(initial_bclamp_amount) * (1.0 - allow_unclamped_detail);
        float keep_detail = 1.0 - effective_clamp_amount;
        history_detail *= keep_detail;

        clamped_history = clamped_bhistory + history_detail;

        // Dampen coverage on clamping events (especially important for upsampling)
        if (input_resolution_scale.x < 1.0)
        {
            history_coverage *= lerp(
                lerp(0.0, 0.9, keep_detail),
                1.0,
                saturate(10.0 * clamping_event)
            );
        }

        // Confidence-based: blend in unclamped history when input is stable
        clamped_history = lerp(clamped_history, history, smoothstep(0.5, 1.0, input_confidence));
    }
    else
    {
        // Disocclusion: reset to blurry center
        clamped_history  = clamped_bhistory;
        coverage         = 1.0;
        center           = bcenter;
        history_coverage = 0.0;
    }

    // ---- Temporal blend ----
    float total_coverage   = max(1e-5, history_coverage + coverage);
    float3 temporal_result = (clamped_history * history_coverage + center * coverage) / total_coverage;

    float max_coverage = max(2.0, TARGET_SAMPLE_COUNT / (input_resolution_scale.x * input_resolution_scale.y));
    total_coverage = min(max_coverage, total_coverage);

    // ---- Output ----
    // Convert back: YCbCr -> sRGB -> encode_rgb inverse -> HDR linear
    float3 result_srgb = YCbCr_to_sRGB(temporal_result);
    float3 final_color = encode_rgb(result_srgb);
    final_color = max(0.0, final_color);

    // Write history (linear HDR rgb + coverage)
    history_tex_out[px] = float4(final_color, total_coverage);

    // Write smooth variance (luminance channel only)
    smooth_var_output_tex[px] = smooth_var_lum;

    // Display output: exposure + Reinhard
    float3 exposed = final_color * g_Frame.exposure;
    float3 display = exposed / (exposed + 1.0);
    output_tex[px] = float4(saturate(display), 1.0);
}
