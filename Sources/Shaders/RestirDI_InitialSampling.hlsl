// RestirDI_InitialSampling.hlsl — ReSTIR DI Pass 1: Initial Sampling
// Generates NUM_CANDIDATES light candidates via RIS (no shadow rays),
// selects one winner per pixel, writes to DIRreservoirBuffer[curr].

#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  g_Frame   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

static const uint  NUM_CANDIDATES            = 4u;
static const float RESTIR_DI_INITIAL_MAX_M   = 4.0f;

// Reconstruct surface from G-Buffer at a given screen position.
// Returns false if the pixel is sky (depth == 1.0 or 0.0 for reversed-Z).
bool ReconstructGBufferSurface(uint2 screenPos, uint2 dims, out Surface surf)
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

    float4 ndc     = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, g_Frame.projectionInverse);
    viewPos /= viewPos.w;
    float3 worldPos = mul(viewPos, g_Frame.viewInverse).xyz;

    surf.worldPos  = worldPos;
    surf.normal    = normalize(normalWS);
    surf.viewDir   = normalize(g_Frame.cameraPosition.xyz - worldPos);
    surf.albedo    = albedo.rgb;
    surf.roughness = max(0.01f, material.r);
    surf.metallic  = material.g;
    return true;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 dims      = uint2(g_Frame.screenWidth, g_Frame.screenHeight);
    if (screenPos.x >= dims.x || screenPos.y >= dims.y) return;

    uint pixelIdx = screenPos.y * dims.x + screenPos.x;

    RWStructuredBuffer<DIRreservoir> currReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Surface surf;
    bool hasHit = ReconstructGBufferSurface(screenPos, dims, surf);

    DIRreservoir res = (DIRreservoir)0;

    if (hasHit && GetLocalLightCount(g_Frame.numLights) > 0)
    {
        RNG rng;
        seed_rng(rng, screenPos, g_Frame.frameIndex + 7u);

        uint  selectedIdx = 1u;
        float weightSum   = 0.0f;
        float selectedTargetPdf = 0.0f;

        for (uint i = 0u; i < NUM_CANDIDATES; ++i)
        {
            LightSampleResult lsr = SampleSingleLight(next_float(rng), g_Lights, g_Frame.numLights, g_Frame);
            float tpdf = RIS_TargetPDF(surf.worldPos, surf.normal, surf.viewDir,
                                       surf.albedo, surf.metallic, surf.roughness,
                                       lsr.light);
            float risWeight = (lsr.pdf > 1e-6f) ? (tpdf / lsr.pdf) : 0.0f;
            weightSum += risWeight;
            if (next_float(rng) * weightSum <= risWeight)
            {
                selectedIdx      = lsr.lightIndex;
                selectedTargetPdf = tpdf;
            }
        }

        if (weightSum > 0.0f && selectedTargetPdf > 0.0f)
        {
            res.w_sum             = weightSum;
            res.M                 = float(NUM_CANDIDATES);
            res.targetPdf         = selectedTargetPdf;
            res.selectedLightIndex = selectedIdx;
            res.W                 = res.w_sum / (res.M * res.targetPdf);
        }
    }

    // Debug heatmap
    if (g_Frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF)
    {
        RWTexture2D<float4> dbg = ResourceDescriptorHeap[g_Indices.OutputIdx1];
        float v = 0.0f;
        if      (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_LIGHT_INDEX) v = float(res.selectedLightIndex) / float(max(1u, g_Frame.numLights));
        else if (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_M_COUNT)     v = res.M / RESTIR_DI_INITIAL_MAX_M;
        else if (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_WEIGHT)      v = saturate(res.W);
        dbg[screenPos] = float4(v, 0.0f, 0.0f, 1.0f);
    }

    currReservoirs[pixelIdx] = res;
}
