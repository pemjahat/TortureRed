// NaiveTsr_Reproject.hlsl
// Pass 1 of Naive TSR: Reproject history using motion vectors + 3x3 closest-depth velocity dilation
//
// Inputs (via bindless):
//   InputIdx0  = history_tex (SRV, output-res, R16G16B16A16_FLOAT, rgb + coverage)
//   InputIdx1  = reprojection_tex (SRV, internal-res, R16G16_FLOAT, motion vectors in UV space)
//   InputIdx2  = depth_tex (SRV, internal-res, R32_FLOAT via SRV as R32_FLOAT)
// Outputs (via bindless):
//   OutputIdx0 = reprojected_history_tex (UAV, output-res, R16G16B16A16_FLOAT)
//   OutputIdx1 = closest_velocity_tex (UAV, output-res, R16G16_FLOAT)

#include "Common.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

SamplerState g_LinearClamp : register(s0);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadID.xy;
    if (px.x >= g_Frame.outputWidth || px.y >= g_Frame.outputHeight) return;

    Texture2D<float4> history_tex = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float2> reprojection_tex = ResourceDescriptorHeap[g_Indices.InputIdx1];
    Texture2D<float>  depth_tex = ResourceDescriptorHeap[g_Indices.InputIdx2];
    RWTexture2D<float4> reprojected_history_tex = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float2> closest_velocity_tex = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    float2 output_size = float2(g_Frame.outputWidth, g_Frame.outputHeight);
    float2 internal_size = float2(g_Frame.internalWidth, g_Frame.internalHeight);
    float2 input_resolution_scale = internal_size / output_size;

    // Map output pixel to internal-resolution pixel
    uint2 reproj_px = uint2((px + 0.5) * input_resolution_scale);

    // 3x3 closest-depth velocity dilation
    uint2 closest_px = reproj_px;
    float closest_depth = depth_tex[reproj_px];

    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            uint2 sample_px = reproj_px + int2(x, y);
            float d = depth_tex[sample_px];
            // Standard Z: smaller depth value = closer to camera
            if (d < closest_depth)
            {
                closest_depth = d;
                closest_px = sample_px;
            }
        }
    }

    float2 motion_vector = reprojection_tex[closest_px]; // (prevUV - currentUV) in UV space
    closest_velocity_tex[px] = motion_vector;

    // Compute history UV
    float2 uv = (px + 0.5) / output_size;
    float2 history_uv = uv + motion_vector;

    // Sample history with bilinear filtering (Catmull-Rom 5-tap for better quality)
    float4 history_packed;
    if (all(history_uv == saturate(history_uv)))
    {
        // 5-tap Catmull-Rom approximation for sharper history sampling
        float2 samplePos = history_uv * output_size;
        float2 texPos1 = floor(samplePos - 0.5) + 0.5;
        float2 f = samplePos - texPos1;

        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        float2 w3 = f * f * (-0.5 + 0.5 * f);

        float2 w12 = w1 + w2;
        float2 offset12 = w2 / (w1 + w2);

        float2 texPos0  = (texPos1 - 1.0) / output_size;
        float2 texPos3  = (texPos1 + 2.0) / output_size;
        float2 texPos12 = (texPos1 + offset12) / output_size;

        history_packed  = history_tex.SampleLevel(g_LinearClamp, float2(texPos12.x, texPos0.y),  0) * w12.x * w0.y;
        history_packed += history_tex.SampleLevel(g_LinearClamp, float2(texPos0.x,  texPos12.y), 0) * w0.x  * w12.y;
        history_packed += history_tex.SampleLevel(g_LinearClamp, float2(texPos12.x, texPos12.y), 0) * w12.x * w12.y;
        history_packed += history_tex.SampleLevel(g_LinearClamp, float2(texPos3.x,  texPos12.y), 0) * w3.x  * w12.y;
        history_packed += history_tex.SampleLevel(g_LinearClamp, float2(texPos12.x, texPos3.y),  0) * w12.x * w3.y;

        float wt_sum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
        history_packed /= wt_sum;
    }
    else
    {
        // Out of bounds — no valid history
        history_packed = float4(0, 0, 0, 0);
    }

    reprojected_history_tex[px] = history_packed;
}
