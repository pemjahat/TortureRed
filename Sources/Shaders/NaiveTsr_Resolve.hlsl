// NaiveTsr_Resolve.hlsl
// Pass 2 of Naive TSR: TAA resolve at output resolution
// - Bilinear sample internal-res current frame with unjitter
// - 3x3 neighborhood color bounding-box clamp on reprojected history
// - Fixed blend weight with disocclusion detection
// - Coverage accumulation up to TARGET_SAMPLE_COUNT
//
// Inputs (via bindless):
//   InputIdx0  = current_frame_tex (SRV, internal-res, path tracer HDR or lighting output)
//   InputIdx1  = reprojected_history_tex (SRV, output-res, R16G16B16A16_FLOAT)
//   InputIdx2  = closest_velocity_tex (SRV, output-res, R16G16_FLOAT)
// Outputs (via bindless):
//   OutputIdx0 = history_tex (UAV, output-res, R16G16B16A16_FLOAT, rgb + coverage)
//   OutputIdx1 = output_tex (UAV, output-res, R16G16B16A16_FLOAT, final output)

#include "Common.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

SamplerState g_LinearClamp : register(s0);

#define TARGET_SAMPLE_COUNT 8

// Perceptual color mapping for better clamping behavior
float3 tonemap(float3 c)
{
    float maxComp = max(c.r, max(c.g, c.b));
    float scale = (maxComp > 1e-20) ? sqrt(maxComp) / maxComp : 1.0;
    return c * scale;
}

float3 tonemap_inv(float3 c)
{
    float maxComp = max(c.r, max(c.g, c.b));
    float scale = (maxComp > 1e-20) ? (maxComp * maxComp) / maxComp : 1.0;
    return c * scale;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadID.xy;
    if (px.x >= g_Frame.outputWidth || px.y >= g_Frame.outputHeight) return;

    Texture2D<float4> current_frame_tex = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4> reprojected_history_tex = ResourceDescriptorHeap[g_Indices.InputIdx1];
    Texture2D<float2> closest_velocity_tex = ResourceDescriptorHeap[g_Indices.InputIdx2];
    RWTexture2D<float4> history_tex = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> output_tex = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    float2 output_size = float2(g_Frame.outputWidth, g_Frame.outputHeight);
    float2 internal_size = float2(g_Frame.internalWidth, g_Frame.internalHeight);
    float2 input_resolution_scale = internal_size / output_size;

    // Stable unjittered center
    // ---- Sample current frame with unjitter ----
    // Map output pixel to internal-resolution space, accounting for jitter
    float2 jitter = g_Frame.taaJitter; // In pixel units of internal resolution
    // The projection jitter shifts rendered content by (+jitter.x, +jitter.y) in internal-res pixels.
    // The unjittered internal-res location for this output pixel is:
    //   dst_unjittered_internal = (px + 0.5) * scale - jitter
    // We center the 3x3 kernel on this unjittered location so the neighborhood
    // doesn't shift each frame (which would cause jittered output).
    float2 dst_unjittered_internal = (px + 0.5) * input_resolution_scale - jitter;
    int2 base_src_px = int2(dst_unjittered_internal);

    // Weighted sampling with Gaussian kernel for unjittering
    float4 color_sum = 0;
    float wt_sum = 0;
    float3 ex = 0;
    float3 ex2 = 0;
    float dev_wt_sum = 0;

    // Compute offset in internal-res pixel units for the Gaussian weight.
    // dst_unjittered_internal is the unjittered position this output pixel maps to.
    // base_src_px is already centered on that location, so distances are jitter-stable.
    int k = 1; // kernel half-width
    for (int y = -k; y <= k; ++y)
    {
        for (int x = -k; x <= k; ++x)
        {
            int2 src_px = base_src_px + int2(x, y);

            float4 col = current_frame_tex[src_px];
            float3 mapped = tonemap(col.rgb);

            // Distance in internal-res pixel units between this src pixel center
            // and the unjittered destination location. Both are in unjittered space.
            float2 src_center_internal = src_px + 0.5;
            float2 sample_center_offset = src_center_internal - dst_unjittered_internal;
            float dist2 = dot(sample_center_offset, sample_center_offset);

            float wt = exp2(-10.0 * dist2); // thigher kernel for center
            float dev_wt = exp2(-dist2);    // wider kernel for neighbour

            color_sum += float4(mapped, 1) * wt;
            wt_sum += wt;

            ex += mapped * dev_wt;
            ex2 += mapped * mapped * dev_wt;
            dev_wt_sum += dev_wt;
        }
    }

    
    float coverage = wt_sum;

    ex /= max(dev_wt_sum, 1e-6);
    ex2 /= max(dev_wt_sum, 1e-6);
    float3 var = max(0.0, ex2 - ex * ex);
    float3 input_dev = sqrt(var);

    // -- Entire blend operation is in tonemapped space - both center and history --
    // To avoid bright pixel dominated the blend
    float3 center = color_sum.rgb / max(wt_sum, 1e-6);

    // ---- Read reprojected history ----
    float4 history_packed = reprojected_history_tex[px];
    float3 history = tonemap(history_packed.rgb);
    float history_coverage = max(0.0, history_packed.a);

    // ---- Check reprojection validity ----
    float2 uv = (px + 0.5) / output_size;
    float2 motion_vector = closest_velocity_tex[px];
    float2 history_uv = uv + motion_vector;
    bool history_valid = all(history_uv == saturate(history_uv));

    // ---- Neighborhood color bounding-box clamp ----
    float box_n_deviations = 1.25;
    float3 nmin = ex - input_dev * box_n_deviations;
    float3 nmax = ex + input_dev * box_n_deviations;

    float3 clamped_history = clamp(history, nmin, nmax);

    // Measure how much clamping happened (disocclusion indicator)
    float3 clamp_diff = abs(history - clamped_history);
    float clamp_amount = dot(clamp_diff, 1.0) / max(dot(abs(ex) + 1e-3, 1.0), 1e-6);

    if (!history_valid)
    {
        clamped_history = center;
        history_coverage = 0;
    }

    // ---- Blend current frame with clamped history ----
    // Reduce history coverage when clamping happens (disocclusion)
    history_coverage *= saturate(1.0 - clamp_amount * 2.0);

    float total_coverage = max(1e-5, history_coverage + coverage);

    // Cap coverage to TARGET_SAMPLE_COUNT adjusted for upsampling ratio (~effective sample contribution)
    float max_coverage = max(2.0, TARGET_SAMPLE_COUNT / (input_resolution_scale.x * input_resolution_scale.y));
    total_coverage = min(max_coverage, total_coverage);

    float3 temporal_result = (clamped_history * history_coverage + center * coverage) / total_coverage;

    // ---- Output ----
    float3 final_color = tonemap_inv(temporal_result);
    final_color = max(0.0, final_color);

    // Write history for next frame (rgb + coverage)
    history_tex[px] = float4(min(final_color, FP16Max), total_coverage);

    // Write final output (apply exposure + simple tonemap for display)
    float3 exposed = (final_color / FP16Scale) * exp2(g_Frame.exposure);
    float3 display = exposed / (exposed + 1.0); // Reinhard tonemap
    output_tex[px] = float4(saturate(display), 1.0);
}
