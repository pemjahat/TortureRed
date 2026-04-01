// LocalLight_SpatialReuse.hlsl — Local light specular spatial reuse pass
// Borrows local light samples from 8 neighbouring pixels using specular target PDF.
// Mirrors the approach in RestirGI_Specular_Spatial.hlsl but operates on local light reservoirs.

#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>  g_Frame   : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);

static const float RESTIR_SPATIAL_DEPTH_THRESHOLD = 0.1f;
static const float RESTIR_SPATIAL_NORMAL_THRESHOLD = 0.95f;
static const float RESTIR_SPATIAL_ALBEDO_THRESHOLD = 0.15f;
static const float RESTIR_SPATIAL_ROUGHNESS_THRESHOLD = 0.15f;
static const float RESTIR_SPATIAL_METALLIC_THRESHOLD = 0.15f;
static const float RESTIR_SPATIAL_MIN_JACOBIAN = 0.1f;
static const float RESTIR_SPATIAL_MAX_JACOBIAN = 1.25f;
static const float RESTIR_SPATIAL_REUSE_WEIGHT_CLAMP = 6.0f;

// Specular-only target PDF (same as RTR specular stream)
float GetSpecularTargetPDF(Surface s, float3 samplePos, float3 sampleRadiance)
{
    float3 L = normalize(samplePos - s.worldPos);
    float dotNL = max(0.0f, dot(s.normal, L));
    if (dotNL <= 0) return 0;
    float3 d, spec;
    EvaluateBSDF(s.normal, s.viewDir, L, s.albedo, s.metallic, s.roughness, d, spec);
    float3 reflected = spec * sampleRadiance * dotNL;
    return max(0.0f, Luminance(reflected));
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);
    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;

    // InputIdx0 = current local light reservoirs (SRV)
    // OutputIdx0 = local light reservoir intermediate (UAV)
    StructuredBuffer<Reservoir>     inputReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<Reservoir>   outputReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    // Early exit: no local lights
    if (g_Frame.numLights <= 1)
    {
        outputReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex + 37u); // Unique seed offset

    // Reconstruct center surface from GBuffer
    float depth = g_Textures[g_Frame.depthIndex].Load(int3(screenPos, 0)).r;
    if (depth == 0.0f)
    {
        outputReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    float3 albedo = g_Textures[g_Frame.albedoIndex].Load(int3(screenPos, 0)).rgb;
    float4 packedNormal = g_Textures[g_Frame.normalIndex].Load(int3(screenPos, 0));
    float4 packedMaterial = g_Textures[g_Frame.materialIndex].Load(int3(screenPos, 0));

    float2 uv = (float2(screenPos) + 0.5f) / float2(launchDims);
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 viewPos = mul(ndc, g_Frame.projectionInverse);
    viewPos /= max(viewPos.w, 1e-6f);
    float4 worldPos = mul(viewPos, g_Frame.viewInverse);

    float linearDepth = length(worldPos.xyz - g_Frame.cameraPosition.xyz);

    Surface centerSurface;
    centerSurface.worldPos = worldPos.xyz;
    centerSurface.normal = normalize(packedNormal.xyz * 2.0f - 1.0f);
    centerSurface.viewDir = normalize(g_Frame.cameraPosition.xyz - worldPos.xyz);
    centerSurface.albedo = albedo;
    centerSurface.metallic = packedMaterial.g;
    centerSurface.roughness = max(0.01f, packedMaterial.r);

    // Start with center pixel's reservoir
    Reservoir r = inputReservoirs[pixelIndex];
    float selectedPDF = 0.0f;
    if (r.M > 0.0f)
    {
        selectedPDF = GetSpecularTargetPDF(centerSurface, r.hitPos, r.radiance);
    }

    // Spatial reuse: 8 neighbours within a screen-space radius
    int numNeighbors = 8;
    float radius = 20.0f;

    for (int i = 0; i < numNeighbors; ++i)
    {
        float2 offset = float2(next_float(rng) * 2.0f - 1.0f, next_float(rng) * 2.0f - 1.0f) * radius;
        int2 neighborPos = int2(screenPos) + int2(offset);

        if (neighborPos.x < 0 || neighborPos.x >= (int)launchDims.x ||
            neighborPos.y < 0 || neighborPos.y >= (int)launchDims.y)
            continue;

        uint neighborIndex = neighborPos.y * launchDims.x + neighborPos.x;
        Reservoir neighborR = inputReservoirs[neighborIndex];

        if (neighborR.M <= 0.0f) continue;

        // Reconstruct neighbor surface from GBuffer for similarity tests
        float neighborDepth = g_Textures[g_Frame.depthIndex].Load(int3(neighborPos, 0)).r;
        if (neighborDepth == 0.0f) continue;

        float3 neighborAlbedo = g_Textures[g_Frame.albedoIndex].Load(int3(neighborPos, 0)).rgb;
        float4 neighborPackedNormal = g_Textures[g_Frame.normalIndex].Load(int3(neighborPos, 0));
        float4 neighborPackedMaterial = g_Textures[g_Frame.materialIndex].Load(int3(neighborPos, 0));

        float2 neighborUV = (float2(neighborPos) + 0.5f) / float2(launchDims);
        float4 neighborNDC = float4(neighborUV.x * 2.0f - 1.0f, (1.0f - neighborUV.y) * 2.0f - 1.0f, neighborDepth, 1.0f);
        float4 neighborViewPos = mul(neighborNDC, g_Frame.projectionInverse);
        neighborViewPos /= max(neighborViewPos.w, 1e-6f);
        float4 neighborWorldPos = mul(neighborViewPos, g_Frame.viewInverse);

        float neighborLinearDepth = length(neighborWorldPos.xyz - g_Frame.cameraPosition.xyz);

        Surface neighborSurface;
        neighborSurface.worldPos = neighborWorldPos.xyz;
        neighborSurface.normal = normalize(neighborPackedNormal.xyz * 2.0f - 1.0f);
        neighborSurface.viewDir = normalize(g_Frame.cameraPosition.xyz - neighborWorldPos.xyz);
        neighborSurface.albedo = neighborAlbedo;
        neighborSurface.metallic = neighborPackedMaterial.g;
        neighborSurface.roughness = max(0.01f, neighborPackedMaterial.r);

        // Similarity tests
        bool normalsMatch = dot(centerSurface.normal, neighborSurface.normal) > RESTIR_SPATIAL_NORMAL_THRESHOLD;
        bool depthMatch = abs(neighborLinearDepth - linearDepth) <= (RESTIR_SPATIAL_DEPTH_THRESHOLD * max(1.0f, linearDepth));
        bool materialMatch = AreMaterialsSimilar(centerSurface, neighborSurface,
            RESTIR_SPATIAL_ALBEDO_THRESHOLD,
            RESTIR_SPATIAL_ROUGHNESS_THRESHOLD,
            RESTIR_SPATIAL_METALLIC_THRESHOLD);

        if (!normalsMatch || !depthMatch || !materialMatch) continue;

        // Evaluate neighbor's reservoir at center surface
        float neighborTargetPDF = GetSpecularTargetPDF(centerSurface, neighborR.hitPos, neighborR.radiance);

        // Jacobian correction for the shift in primary surface position
        float jacobian = ComputeJacobian(centerSurface.worldPos, neighborSurface.worldPos, neighborR.hitPos, neighborR.hitNormal);
        bool jacobianValid = jacobian >= RESTIR_SPATIAL_MIN_JACOBIAN && jacobian <= RESTIR_SPATIAL_MAX_JACOBIAN;

        if (neighborTargetPDF > 0.0f && jacobianValid)
        {
            float shiftedTargetPDF = neighborTargetPDF * jacobian;

            Reservoir adjustedNeighbor = neighborR;
            float spatialReuseWeight = shiftedTargetPDF * adjustedNeighbor.W * adjustedNeighbor.M;
            spatialReuseWeight = min(spatialReuseWeight, RESTIR_SPATIAL_REUSE_WEIGHT_CLAMP);

            if (mergeReservoirsWithWeight(r, adjustedNeighbor, spatialReuseWeight, next_float(rng)))
            {
                selectedPDF = neighborTargetPDF;
            }
        }
    }

    // Normalize reservoir weight
    if (r.M > 0.0f && selectedPDF > 0.0f)
    {
        r.W = r.w_sum / (r.M * selectedPDF);
    }
    else
    {
        r.W = 0.0f;
    }

    outputReservoirs[pixelIndex] = r;
}
