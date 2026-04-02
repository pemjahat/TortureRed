// MotionVectors.hlsl
// Generate screen-space motion vectors from depth + current/previous viewProj matrices
// Output: (prevUV - currentUV) in normalized UV space [0,1]

#include "Common.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadID.xy;
    if (px.x >= g_Frame.screenWidth || px.y >= g_Frame.screenHeight) return;

    Texture2D<float> depth_tex = ResourceDescriptorHeap[g_Frame.depthIndex];
    RWTexture2D<float2> motion_vector_tex = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    float depth = depth_tex[px];

    // Sky/background: output zero motion vector
    if (depth <= 0.0 || depth >= 1.0)
    {
        motion_vector_tex[px] = float2(0, 0);
        return;
    }

    // Current UV
    float2 uv = (px + 0.5) / float2(g_Frame.screenWidth, g_Frame.screenHeight);

    // Reconstruct world position from depth using the UNJITTERED projection inverse.
    // The depth value is the same whether jittered or not (jitter only affects X/Y clip offsets).
    // Using the unjittered inverse ensures the reconstructed world position corresponds to
    // the pixel center's ray direction, so motion vectors encode pure camera motion
    // without jitter differences between frames.
    float3 worldPos = ReconstructWorldPos(uv, depth, g_Frame.projectionInverseUnjittered, g_Frame.viewInverse);

    // Project to previous frame
    float4 prevClip = mul(float4(worldPos, 1.0), g_Frame.viewProjPrevious);
    float2 prevNDC = prevClip.xy / prevClip.w;
    float2 prevUV = float2(prevNDC.x * 0.5 + 0.5, 0.5 - prevNDC.y * 0.5);

    // Motion vector = prevUV - currentUV
    float2 motionVector = prevUV - uv;

    motion_vector_tex[px] = motionVector;
}
