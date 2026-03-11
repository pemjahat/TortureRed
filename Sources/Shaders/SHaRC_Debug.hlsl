// SHaRC_Debug.hlsl
// Shared debug visualization pass — two modes selected by g_Frame.sharcDebug:
//
//   1 — SHaRC Output : queries SharcGetCachedRadiance at the primary hit,
//       producing the same blocky voxel-boundary visualization as RTXGI's
//       "Enable Debug" checkbox.
//
//   2 — Bounce Heatmap : traces secondary bounces from the primary hit and
//       maps the geometric bounce count to a cyan→red color ramp so you can
//       see path depth distribution across the screen.
//
// Dispatched at full resolution ONLY when sharcDebug != 0, replacing the
// ReSTIR-resolved indirect irradiance with the debug visualization.
// OutputIdx0 must point to m_RasterIndirectLightingTex UAV.

#define SHARC_ENABLE_DEBUG 1
#include "sharc/SharcCommon.h"
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants>       g_Frame   : register(b0);
ConstantBuffer<BindlessIndices>      g_Indices : register(b1);
ConstantBuffer<SharcBindlessIndices> g_Sharc   : register(b2);

// Maps a bounce count to RTXGI's BounceHeatmap palette (PathtracerUtils.hlsli):
//   0  → blue  (first secondary ray missed sky — no geometry bounces)
//   1  → green (one secondary geometry hit)
//   2+ → red   (two or more secondary geometry hits)
// Counting semantics are identical to RTXGI: bounce++ only fires after a geometry
// hit, break on sky miss does not increment — so value N means N geometry hits.
float3 BounceCountColor(int n)
{
    switch (n)
    {
    case 0:  return float3(0.0f, 0.0f, 1.0f); // blue
    case 1:  return float3(0.0f, 1.0f, 0.0f); // green
    default: return float3(1.0f, 0.0f, 0.0f); // red (2+)
    }
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos  = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    RWTexture2D<float4> outDebug = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex);

    // Init sharc parameter
    SharcParameters sharcParams;
    sharcParams.gridParameters.cameraPosition  = g_Frame.irCacheCameraPosition.xyz;
    sharcParams.gridParameters.logarithmBase   = SHARC_GRID_LOGARITHM_BASE;
    sharcParams.gridParameters.sceneScale      = g_Frame.sharcSceneScale;
    sharcParams.gridParameters.levelBias       = 0.0f;
    sharcParams.hashMapData.capacity           = SHARC_HASH_ENTRIES_NUM;
    sharcParams.hashMapData.hashEntriesBuffer  = ResourceDescriptorHeap[g_Sharc.HashEntriesBufIdx];
    sharcParams.accumulationBuffer             = ResourceDescriptorHeap[g_Sharc.AccumulationBufIdx];
    sharcParams.resolvedBuffer                 = ResourceDescriptorHeap[g_Sharc.ResolvedBufIdx];
    sharcParams.radianceScale                  = 1e3f;
    sharcParams.enableAntiFireflyFilter        = false;

    float3 throughput = 1;
    int bounceCount = 0;
    float3 debugColor = float3(0, 0, 0);

    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, g_Frame, rng, surface, primaryRayT);

    if (hasPrimaryHit) {
        bool isPathDiffuse = false;
        float cumulativeRoughness = surface.roughness;

        if (g_Frame.sharcDebug == 1)
        {
            SharcHitData sharcQuery;
            sharcQuery.positionWorld = surface.worldPos;
            sharcQuery.normalWorld   = surface.normal;

            SharcGetCachedRadiance(sharcParams, sharcQuery, debugColor, true);
        }

        for (bounceCount = 1; bounceCount < 4; bounceCount++) {
            if (all(throughput <= 0.0f)) break;

            // Sample bounce
            float3 nextDir, nextThroughput;
            float next_pdf;
            SampleIndirectRay(surface.normal, surface.viewDir, 
                surface.albedo, surface.metallic, surface.roughness, 
                rng, nextDir, nextThroughput, next_pdf, isPathDiffuse, 
                g_Frame.enableIndirectSpecular != 0);                          

            throughput *= nextThroughput;

            // Russian Roulette
            if (bounceCount > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (next_float(rng) > p) break;
                throughput /= p;
            }

            RayDesc ray;
            ray.Origin    = surface.worldPos + surface.normal * 0.001f;
            ray.Direction = nextDir;
            ray.TMin      = 0.01f;
            ray.TMax      = 1000.0f;

            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
            while (q.Proceed()) {
                PROCESS_ALPHA_MASK(q, rng);
            }

            if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            {
                // Hit sky
                break;
            }

            // --- Decode triangle geometry ---
            uint instanceIdx = q.CommittedInstanceID();
            uint triIdx = q.CommittedPrimitiveIndex();
            float2 barys = q.CommittedTriangleBarycentrics();
            
            DrawNodeData nodeData = g_DrawNodeBuffer[instanceIdx];
            MaterialConstants mat = g_Materials[nodeData.materialID];

            uint i0 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 0];
            uint i1 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 1];
            uint i2 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 2];

            GLTFVertex v0 = g_GlobalVertices[nodeData.vertexOffset + i0];
            GLTFVertex v1 = g_GlobalVertices[nodeData.vertexOffset + i1];
            GLTFVertex v2 = g_GlobalVertices[nodeData.vertexOffset + i2];

            float  bary0     = 1.0f - barys.x - barys.y;
            float2 hitUv     = v0.texCoord * bary0 + v1.texCoord * barys.x + v2.texCoord * barys.y;
            float3 hitNormal = normalize(mul(v0.normal * bary0 + v1.normal * barys.x + v2.normal * barys.y, 
                                (float3x3)nodeData.world));

            float4 hitAlbedo = mat.baseColorFactor;
            if (mat.baseColorTextureIndex >= 0)
                hitAlbedo *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);

            float hitMetallic  = mat.metallicFactor;
            float hitRoughness = mat.roughnessFactor;
            if (mat.metallicRoughnessTextureIndex >= 0)
            {
                float4 mr = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
                hitRoughness *= mr.g;
                hitMetallic  *= mr.b;
            }

            float3 hitPos     = ray.Origin + ray.Direction * q.CommittedRayT();
            float3 hitViewDir = -ray.Direction;

            SharcHitData sharcQuery;
            sharcQuery.positionWorld = hitPos;
            sharcQuery.normalWorld   = hitNormal;

            // Sharc query
            uint gridLevel = HashGridGetLevel(hitPos, sharcParams.gridParameters);
            float voxelSize = HashGridGetVoxelSize(gridLevel, sharcParams.gridParameters);
            bool isValidHit = q.CommittedRayT() > voxelSize * sqrt(3.0f);   // If Hit point and ray origin inside same voxel (rejected)

            cumulativeRoughness = min(cumulativeRoughness, 0.99f);
            float alpha = cumulativeRoughness * cumulativeRoughness;
            float footprint = q.CommittedRayT() * sqrt(0.5f * alpha * alpha / (1.0f - alpha * alpha));
            isValidHit &= footprint > voxelSize;

            float3 sampleRadiance;
            if (isValidHit && SharcGetCachedRadiance(sharcParams, sharcQuery, sampleRadiance, false))
                break;
            // HashGridParameters gridParameters;
            // gridParameters.cameraPosition = g_Frame.irCacheCameraPosition.xyz;
            // gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
            // gridParameters.sceneScale = g_Frame.sharcSceneScale;
            // gridParameters.levelBias = 0.0f;
            // debugColor = HashGridDebugColoredHash(hitPos, worldNormal, gridParameters);
            // break;

            // Advance primary surface to this hit for the next bounce
            surface.worldPos  = hitPos;
            surface.normal    = hitNormal;
            surface.viewDir   = hitViewDir;
            surface.albedo    = hitAlbedo.rgb;
            surface.metallic  = hitMetallic;
            surface.roughness = max(0.01f, hitRoughness);

            cumulativeRoughness += surface.roughness;
        }
    }

    if (g_Frame.sharcDebug == 1)
    {
        outDebug[screenPos] = float4(debugColor, 1.0f);
    }
    else
    {
        outDebug[screenPos] = float4(BounceCountColor(bounceCount), 1.0f);
    }
}
