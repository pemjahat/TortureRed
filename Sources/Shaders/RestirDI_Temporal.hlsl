// RestirDI_Temporal.hlsl — ReSTIR DI Pass 2: Temporal Resampling
// Reprojects previous-frame reservoirs and merges them into the current frame.
// No Jacobian needed for light-domain reservoirs.

#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  g_Frame   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

static const float RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH = 20.0f;
static const uint  RESTIR_DI_TEMPORAL_MAX_AGE            = 30u;
static const float RESTIR_DI_DEPTH_THRESHOLD             = 0.1f;
static const float RESTIR_DI_NORMAL_THRESHOLD            = 0.85f;

// Reconstruct surface from G-Buffer (current or previous frame).
bool ReconstructGBufferSurface(uint2 screenPos, uint2 dims, bool usePrevFrame, out Surface surf)
{
    float2 uv = (float2(screenPos) + 0.5f) / float2(dims);

    Texture2D<float4> albedoTex   = ResourceDescriptorHeap[g_Frame.albedoIndex];
    Texture2D<float4> normalTex   = ResourceDescriptorHeap[g_Frame.normalIndex];
    Texture2D<float4> materialTex = ResourceDescriptorHeap[g_Frame.materialIndex];
    Texture2D<float>  depthTex    = ResourceDescriptorHeap[g_Frame.depthIndex];

    float depth = depthTex.SampleLevel(g_LinearSampler, uv, 0).r;
    if (depth <= 0.0f || depth >= 1.0f) { surf = (Surface)0; return false; }

    float4 albedo   = albedoTex.SampleLevel(g_LinearSampler, uv, 0);
    float3 normalWS = normalTex.SampleLevel(g_LinearSampler, uv, 0).rgb * 2.0f - 1.0f;
    float4 material = materialTex.SampleLevel(g_LinearSampler, uv, 0);

    float4x4 projInv = g_Frame.projectionInverse;
    float4x4 viewInv = usePrevFrame ? g_Frame.viewInversePrevious : g_Frame.viewInverse;

    float4 ndc     = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, projInv);
    viewPos /= viewPos.w;
    float3 worldPos = mul(viewPos, viewInv).xyz;

    surf.worldPos  = worldPos;
    surf.normal    = normalize(normalWS);
    surf.viewDir   = normalize(g_Frame.cameraPosition.xyz - worldPos);
    surf.albedo    = albedo.rgb;
    surf.roughness = max(0.01f, material.r);
    surf.metallic  = material.g;
    return true;
}

// Merge a DI reservoir (no Jacobian — light-domain reuse).
bool MergeDIReservoirs(inout DIRreservoir dst, DIRreservoir src, float risWeight, float rnd)
{
    dst.w_sum += risWeight;
    dst.M     += src.M;
    if (rnd * dst.w_sum <= risWeight)
    {
        dst.selectedLightIndex = src.selectedLightIndex;
        dst.targetPdf          = src.targetPdf;
        return true;
    }
    return false;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 dims      = uint2(g_Frame.screenWidth, g_Frame.screenHeight);
    if (screenPos.x >= dims.x || screenPos.y >= dims.y) return;

    uint pixelIdx = screenPos.y * dims.x + screenPos.x;

    StructuredBuffer<DIRreservoir>   prevReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<DIRreservoir> currReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    DIRreservoir curr = currReservoirs[pixelIdx]; // Written by Pass 1

    Surface surf;
    bool hasHit = ReconstructGBufferSurface(screenPos, dims, false, surf);

    if (!hasHit || curr.M <= 0.0f)
    {
        currReservoirs[pixelIdx] = (DIRreservoir)0;
        return;
    }

    float selectedTargetPdf = curr.targetPdf;

    // Temporal reprojection
    if (g_Frame.frameIndex > 1u)
    {
        float4 prevClip = mul(float4(surf.worldPos, 1.0f), g_Frame.viewProjPrevious);
        prevClip /= prevClip.w;
        float2 prevUV = prevClip.xy * float2(0.5f, -0.5f) + 0.5f;

        if (prevUV.x >= 0.0f && prevUV.x < 1.0f && prevUV.y >= 0.0f && prevUV.y < 1.0f)
        {
            uint2 prevPos = min((uint2)(prevUV * float2(dims)), dims - 1u);
            DIRreservoir prevRes = prevReservoirs[prevPos.y * dims.x + prevPos.x];

            if (prevRes.M > 0.0f && prevRes.selectedLightIndex < g_Frame.numLights)
            {
                // Validate reprojection with depth + normal
                RNG prevRng;
                seed_rng(prevRng, prevPos, (g_Frame.frameIndex - 1u) + 911u);
                Surface prevSurf;
                float prevRayT;
                bool hasPrevHit = TracePrimarySurface(prevPos, dims, g_Frame, prevRng, prevSurf, prevRayT, true);

                float expectedT = length(surf.worldPos - g_Frame.prevCameraPosition.xyz);
                float prevT     = length(prevSurf.worldPos - g_Frame.prevCameraPosition.xyz);
                bool depthOk  = hasPrevHit && abs(prevT - expectedT) <= RESTIR_DI_DEPTH_THRESHOLD * max(1.0f, expectedT);
                bool normalOk = hasPrevHit && dot(surf.normal, prevSurf.normal) > RESTIR_DI_NORMAL_THRESHOLD;

                if (depthOk && normalOk)
                {
                    // Recompute target PDF of previous light at current surface
                    LightConstants prevLight = g_Lights[prevRes.selectedLightIndex];
                    float prevTargetPdf = RIS_TargetPDF(surf.worldPos, surf.normal, surf.viewDir,
                                                        surf.albedo, surf.metallic, surf.roughness,
                                                        prevLight);

                    if (prevTargetPdf > 0.0f)
                    {
                        // Cap history before merge
                        DIRreservoir cappedPrev = prevRes;
                        if (cappedPrev.M > RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH)
                        {
                            cappedPrev.w_sum *= RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH / cappedPrev.M;
                            cappedPrev.M      = RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH;
                        }

                        float risWeight = prevTargetPdf * cappedPrev.W * cappedPrev.M;
                        RNG rng;
                        seed_rng(rng, screenPos, g_Frame.frameIndex + 13u);
                        if (MergeDIReservoirs(curr, cappedPrev, risWeight, next_float(rng)))
                        {
                            selectedTargetPdf = prevTargetPdf;
                        }
                    }
                }
            }
        }
    }

    // Cap total history
    if (curr.M > RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH)
    {
        curr.w_sum *= RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH / curr.M;
        curr.M      = RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH;
    }

    // Normalize
    if (curr.M > 0.0f && selectedTargetPdf > 0.0f)
        curr.W = curr.w_sum / (curr.M * selectedTargetPdf);
    else
        curr.W = 0.0f;

    curr.targetPdf = selectedTargetPdf;

    // Debug heatmap
    if (g_Frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF)
    {
        RWTexture2D<float4> dbg = ResourceDescriptorHeap[g_Indices.OutputIdx1];
        float v = 0.0f;
        if      (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_M_COUNT) v = curr.M / RESTIR_DI_TEMPORAL_MAX_HISTORY_LENGTH;
        else if (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_WEIGHT)  v = saturate(curr.W);
        dbg[screenPos] = float4(v, 0.0f, 0.0f, 1.0f);
    }

    currReservoirs[pixelIdx] = curr;
}
