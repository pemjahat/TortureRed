#include "Common.hlsl"
#include "NRD.hlsli"

Texture2D g_Textures[] : register(t0, space0);
ConstantBuffer<FrameConstants> FrameCB : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(FrameCB.screenWidth, FrameCB.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y)
        return;

    RWTexture2D<float2> motionVectors = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> normalRoughness = ResourceDescriptorHeap[g_Indices.OutputIdx1];
    RWTexture2D<float> viewZ = ResourceDescriptorHeap[g_Indices.OutputIdx2];

    float depth = g_Textures[FrameCB.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        motionVectors[screenPos] = 0.0f.xx;
        normalRoughness[screenPos] = NRD_FrontEnd_PackNormalAndRoughness(float3(0.0f, 0.0f, 1.0f), 1.0f, 0.0f);
        viewZ[screenPos] = 1e6f;
        return;
    }

    float4 packedNormal = g_Textures[FrameCB.normalIndex].Load(int3(screenPos, 0));
    float4 packedMaterial = g_Textures[FrameCB.materialIndex].Load(int3(screenPos, 0));

    float3 surfaceNormal = normalize(packedNormal.xyz * 2.0f - 1.0f);
    float roughness = max(0.01f, packedMaterial.r);

    float2 uv = (float2(screenPos) + 0.5f) / float2(launchDims);
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, FrameCB.projectionInverse);
    viewPos /= max(viewPos.w, 1e-6f);
    float4 worldPos = mul(viewPos, FrameCB.viewInverse);

    float4 prevClip = mul(float4(worldPos.xyz, 1.0f), FrameCB.viewProjPrevious);
    prevClip /= max(prevClip.w, 1e-6f);
    float2 prevUv = prevClip.xy * float2(0.5f, -0.5f) + 0.5f;
    float2 motion = prevUv - uv;

    motionVectors[screenPos] = motion;
    // NRD_NORMAL_ENCODING = 2 expects R10G10B10A2_UNORM packed normal+roughness data.
    normalRoughness[screenPos] = NRD_FrontEnd_PackNormalAndRoughness(surfaceNormal, roughness, 0.0f);
    viewZ[screenPos] = viewPos.z;
}
