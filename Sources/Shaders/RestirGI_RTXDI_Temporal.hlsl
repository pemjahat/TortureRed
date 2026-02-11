#include "Common.hlsl"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);
Texture2D g_Textures[] : register(t0, space0);

RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirBuffer : register(u2);
RWStructuredBuffer<RTXDI_PackedGIReservoir> g_ReservoirHistory : register(u3);
Buffer<float2> g_NeighborOffsets : register(t5, space1);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<LightConstants> g_Light : register(b1);
SamplerState g_LinearSampler : register(s0);

#define RTXDI_GI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_BASIC
#define RTXDI_GI_RESERVOIR_BUFFER g_ReservoirHistory
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER g_NeighborOffsets
#define RTXDI_ENABLE_STORE_RESERVOIR 1

#include "RtxdiBridge.hlsli"
#include "Rtxdi/GI/Reservoir.hlsli"
#include "Rtxdi/GI/TemporalResampling.hlsli"

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_AccumulationBuffer.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(launchIndex, g_Frame.frameIndex);
    RAB_Surface surface = RAB_GetGBufferSurface(launchIndex, false);

    if (!RAB_IsSurfaceValid(surface)) {
        RTXDI_ReservoirBufferParameters reservoirParams;
        reservoirParams.reservoirBlockRowPitch = (launchDims.x + 15) / 16 * 256;
        reservoirParams.reservoirArrayPitch = 0;
        uint ptr = RTXDI_ReservoirPositionToPointer(reservoirParams, launchIndex, 0);
        g_ReservoirBuffer[ptr] = (RTXDI_PackedGIReservoir)0;
        return;
    }

    // --- Trace Initial Sample (copied from RestirGI_Temporal.hlsl) ---
    float3 throughput = 1;
    float firstBouncePDF = 1.0f;
    float3 firstBounceThroughput = 1.0f;
    bool isPathDiffuse = false;

    // Generate first indirect bounce
    float3 rayDir;
    SampleIndirectRay(surface.normal, surface.viewDir, surface.albedo, surface.metallic, surface.roughness, rng, rayDir, firstBounceThroughput, firstBouncePDF, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

    throughput = 1.0f; // Throughput relative to the secondary surface
    float3 rayPos = surface.worldPos + surface.normal * 0.001f;
    
    float3 L_secondary = 0;
    float3 pos_secondary = 0;
    float3 norm_secondary = 0;
    bool hit_secondary = false;

    for (int bounce = 1; bounce < 4; bounce++) {
        RayDesc ray;
        ray.Origin = rayPos; ray.Direction = rayDir;
        ray.TMin = 0.001f; ray.TMax = 10000.0f;

        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
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

            float3 worldNormal = normalize(mul(v0.normal * (1.0f-barys.x-barys.y) + v1.normal * barys.x + v2.normal * barys.y, (float3x3)nodeData.world));
            float2 uv_hit = v0.texCoord * (1.0f-barys.x-barys.y) + v1.texCoord * barys.x + v2.texCoord * barys.y;
            float3 hitPos = ray.Origin + ray.Direction * q.CommittedRayT();

            float4 albedo_hit = mat.baseColorFactor;
            if (mat.baseColorTextureIndex >= 0) albedo_hit *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, uv_hit, 0);

            float metallic_hit = mat.metallicFactor;
            float roughness_hit = mat.roughnessFactor;
            if (mat.metallicRoughnessTextureIndex >= 0) {
                float4 mrSample = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, uv_hit, 0);
                roughness_hit *= mrSample.g; metallic_hit *= mrSample.b;
            }
            roughness_hit = max(0.15f, roughness_hit);

            float3 V_hit = -ray.Direction;

            // Direct lighting at hit point
            float3 L_direct = GetDirectLighting(hitPos, worldNormal, V_hit, albedo_hit.rgb, metallic_hit, roughness_hit, g_Scene, g_Light, g_Frame, isPathDiffuse);
            
            if (bounce == 1) {
                pos_secondary = hitPos;
                norm_secondary = worldNormal;
                L_secondary = L_direct;
                hit_secondary = true;
            } else {
                L_secondary += L_direct * throughput;
            }

            float3 nextDir; float3 nextThroughput; float next_pdf;
            SampleIndirectRay(worldNormal, V_hit, albedo_hit.rgb, metallic_hit, roughness_hit, rng, nextDir, nextThroughput, next_pdf, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

            throughput *= nextThroughput;
            rayPos = hitPos + worldNormal * 0.001f;
            rayDir = nextDir;
            
            if (bounce > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (next_float(rng) > p) break;
                throughput /= p;
            }
        } else {
            float3 skyRadiance = float3(0.5f, 0.7f, 1.0f) * 0.2f;
            if (bounce == 1) {
                pos_secondary = ray.Origin + ray.Direction * 1000.0f;
                norm_secondary = -ray.Direction;
                L_secondary = skyRadiance;
                hit_secondary = true;
            } else {
                L_secondary += skyRadiance * throughput;
            }
            break;
        }
    }

    RTXDI_GIReservoir initialReservoir = RTXDI_EmptyGIReservoir();
    if (hit_secondary) {
        // RTXDI_MakeGIReservoir takes raw radiance and the PDF of the first bounce
        initialReservoir = RTXDI_MakeGIReservoir(pos_secondary, norm_secondary, L_secondary, firstBouncePDF);
    }

    // --- Temporal Resampling ---
    RTXDI_RuntimeParameters params;
    params.activeCheckerboardField = 0;
    params.neighborOffsetMask = 0;

    RTXDI_ReservoirBufferParameters reservoirParams;
    reservoirParams.reservoirBlockRowPitch = (launchDims.x + 15) / 16 * 256;
    reservoirParams.reservoirArrayPitch = 0;

    RTXDI_GITemporalResamplingParameters tparams;
    float4 clipPos = mul(float4(surface.worldPos, 1.0f), g_Frame.viewProj);
    float3 ndc = clipPos.xyz / clipPos.w;
    float4 prevClipPos = mul(float4(surface.worldPos, 1.0f), g_Frame.viewProjPrevious);
    float3 prevNDC = prevClipPos.xyz / prevClipPos.w;
    float2 pixelPos = ((ndc.xy * float2(0.5f, -0.5f)) + 0.5f) * launchDims;
    float2 prevPixelPos = ((prevNDC.xy * float2(0.5f, -0.5f)) + 0.5f) * launchDims;
    
    tparams.screenSpaceMotion.xy = prevPixelPos - pixelPos;
    tparams.screenSpaceMotion.z = 0; // Simple for now
    tparams.sourceBufferIndex = 1; // Used if reading from multiple-arrayed buffer, but we use history buffer directly
    tparams.maxHistoryLength = 32;
    //tparams.biasCorrectionMode = RTXDI_GI_ALLOWED_BIAS_CORRECTION;
    tparams.biasCorrectionMode = 1;
    tparams.depthThreshold = 0.1f;
    tparams.normalThreshold = 0.5f;
    tparams.maxReservoirAge = 30;
    tparams.enablePermutationSampling = true;
    tparams.enableFallbackSampling = true;
    tparams.uniformRandomNumber = g_Frame.frameIndex;

    // RTXDI_GITemporalResampling uses RAB_GetGBufferSurface(idx, true) for history
    // We must ensure 'g_ReservoirHistory' is used for that.
    // RTXDI SDK uses its own functions to read history reservoir.
    // In GI/TemporalResampling.hlsli:
    // RTXDI_GIReservoir temporalReservoir = RTXDI_LoadGIReservoir(idx, reservoirParams, tparams.sourceBufferIndex);
    // So we need to define RTXDI_LoadGIReservoir to use g_ReservoirHistory.

    RTXDI_GIReservoir result = RTXDI_GITemporalResampling(
        launchIndex, surface, initialReservoir, rng, params, reservoirParams, tparams);

    uint ptr = RTXDI_ReservoirPositionToPointer(reservoirParams, launchIndex, 0);
    g_ReservoirBuffer[ptr] = RTXDI_PackGIReservoir(result, 0);
}
