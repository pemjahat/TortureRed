// RestirDI_Spatial.hlsl — ReSTIR DI Pass 3: Spatial Resampling
// Merges NUM_NEIGHBORS random neighbors into the center pixel's reservoir.
// Recomputes target PDF at center surface for each neighbor's selected light.

#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  g_Frame   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

static const uint  NUM_NEIGHBORS                       = 5u;
static const float NEIGHBOR_RADIUS                     = 16.0f;
static const float RESTIR_DI_SPATIAL_MAX_HISTORY_LENGTH = 40.0f;
static const float RESTIR_DI_SPATIAL_REUSE_WEIGHT_CLAMP = 64.0f;
static const float RESTIR_DI_DEPTH_THRESHOLD            = 0.1f;
static const float RESTIR_DI_NORMAL_THRESHOLD           = 0.85f;

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

    StructuredBuffer<DIRreservoir>   currReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<DIRreservoir> outReservoirs  = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Surface centerSurf;
    bool hasCenterHit = ReconstructGBufferSurface(screenPos, dims, centerSurf);

    DIRreservoir center = currReservoirs[pixelIdx];

    if (!hasCenterHit || center.M <= 0.0f)
    {
        outReservoirs[pixelIdx] = (DIRreservoir)0;
        return;
    }

    // Start spatial accumulator with center reservoir
    DIRreservoir spatial = (DIRreservoir)0;
    float selectedTargetPdf = 0.0f;

    // Seed center into spatial accumulator
    {
        float tpdf = RIS_TargetPDF(centerSurf.worldPos, centerSurf.normal, centerSurf.viewDir,
                                   centerSurf.albedo, centerSurf.metallic, centerSurf.roughness,
                                   g_Lights[center.selectedLightIndex]);
        float risWeight = tpdf * center.W * center.M;
        spatial.w_sum += risWeight;
        spatial.M     += center.M;
        if (risWeight > 0.0f)
        {
            spatial.selectedLightIndex = center.selectedLightIndex;
            spatial.targetPdf          = tpdf;
            selectedTargetPdf          = tpdf;
        }
    }

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex + 3u);

    for (uint i = 0u; i < NUM_NEIGHBORS; ++i)
    {
        // Random neighbor within radius
        float angle  = next_float(rng) * 6.28318530718f;
        float radius = sqrt(next_float(rng)) * NEIGHBOR_RADIUS;
        int2  offset = int2(cos(angle) * radius, sin(angle) * radius);
        int2  nbPos  = int2(screenPos) + offset;

        if (nbPos.x < 0 || nbPos.x >= (int)dims.x || nbPos.y < 0 || nbPos.y >= (int)dims.y)
            continue;

        uint2 nbPosU = uint2(nbPos);
        DIRreservoir nbRes = currReservoirs[nbPosU.y * dims.x + nbPosU.x];
        if (nbRes.M <= 0.0f || nbRes.selectedLightIndex >= g_Frame.numLights)
            continue;

        // Validate geometry compatibility
        Surface nbSurf;
        bool hasNbHit = ReconstructGBufferSurface(nbPosU, dims, nbSurf);
        if (!hasNbHit) continue;

        float dotN = dot(centerSurf.normal, nbSurf.normal);
        float dist = distance(centerSurf.worldPos, nbSurf.worldPos);
        if (dotN < RESTIR_DI_NORMAL_THRESHOLD) continue;

        // Recompute target PDF of neighbor's light at center surface
        LightConstants nbLight = g_Lights[nbRes.selectedLightIndex];
        float shiftedTpdf = RIS_TargetPDF(centerSurf.worldPos, centerSurf.normal, centerSurf.viewDir,
                                          centerSurf.albedo, centerSurf.metallic, centerSurf.roughness,
                                          nbLight);
        if (shiftedTpdf <= 0.0f) continue;

        float risWeight = min(shiftedTpdf * nbRes.W * nbRes.M, RESTIR_DI_SPATIAL_REUSE_WEIGHT_CLAMP);
        if (MergeDIReservoirs(spatial, nbRes, risWeight, next_float(rng)))
        {
            selectedTargetPdf = shiftedTpdf;
        }
    }

    // Cap history
    if (spatial.M > RESTIR_DI_SPATIAL_MAX_HISTORY_LENGTH)
    {
        spatial.w_sum *= RESTIR_DI_SPATIAL_MAX_HISTORY_LENGTH / spatial.M;
        spatial.M      = RESTIR_DI_SPATIAL_MAX_HISTORY_LENGTH;
    }

    // Normalize
    if (spatial.M > 0.0f && selectedTargetPdf > 0.0f)
        spatial.W = spatial.w_sum / (spatial.M * selectedTargetPdf);
    else
        spatial.W = 0.0f;

    spatial.targetPdf = selectedTargetPdf;

    // Debug heatmap
    if (g_Frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF)
    {
        RWTexture2D<float4> dbg = ResourceDescriptorHeap[g_Indices.OutputIdx1];
        float v = 0.0f;
        if      (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_M_COUNT) v = spatial.M / RESTIR_DI_SPATIAL_MAX_HISTORY_LENGTH;
        else if (g_Frame.restirDIDebugMode == RESTIR_DI_DEBUG_WEIGHT)  v = saturate(spatial.W);
        dbg[screenPos] = float4(v, 0.0f, 0.0f, 1.0f);
    }

    outReservoirs[pixelIdx] = spatial;
}
