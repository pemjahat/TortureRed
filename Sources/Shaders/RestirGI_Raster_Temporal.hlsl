#include "CommonTracing.hlsl"
#include "IrCache_Common.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);

// Inputs
Texture3D<float4> g_CurrIrCache : register(t0, space3);
StructuredBuffer<Reservoir> g_PrevReservoirs : register(t1, space3);
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

// Outputs
RWStructuredBuffer<Reservoir> g_CurrReservoirs : register(u0);
RWTexture2D<float4> g_IndirectLightingTex : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims;
    g_IndirectLightingTex.GetDimensions(launchDims.x, launchDims.y);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;
    
    // Initialize RNG
    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex);

    // Read G-Buffer
    float depth = g_Textures[g_Frame.depthIndex].Sample(g_LinearSampler, screenPos).r;
    if (depth == 1.0f) {
        g_CurrReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    float2 uv = ((float2)screenPos + 0.5f) / (float2)launchDims;
    float4 ndc = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 worldPos4 = mul(ndc, g_Frame.projectionInverse);
    worldPos4 /= worldPos4.w;
    float3 worldPos = mul(worldPos4, g_Frame.viewInverse).xyz;

    float3 normal = g_Textures[g_Frame.normalIndex].Sample(g_LinearSampler, screenPos).rgb * 2.0f - 1.0f;
    float3 albedo = g_Textures[g_Frame.albedoIndex].Sample(g_LinearSampler, screenPos).rgb;
    float4 material = g_Textures[g_Frame.materialIndex].Sample(g_LinearSampler, screenPos);
    float metallic = material.r;
    float roughness = max(0.01f, material.g);
    float3 viewDir = normalize(g_Frame.cameraPosition.xyz - worldPos);

    // 1. Trace Single Indirect Bounce
    float3 rayDir;
    float3 throughput;
    float pdf;
    bool isDiffuse;
    SampleIndirectRay(normal, viewDir, albedo, metallic, roughness, rng, rayDir, throughput, pdf, isDiffuse, g_Frame.enableIndirectSpecular);

    RayDesc ray;
    ray.Origin = worldPos + normal * 0.001f;
    ray.Direction = rayDir;
    ray.TMin = 0.01f;
    ray.TMax = 1000.0f;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
        PROCESS_ALPHA_MASK(q, rng);
    }

    float3 sampleRadiance = 0.0f;
    float3 hitPos = 0.0f;
    float3 hitNormal = 0.0f;

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
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

        float bary0 = 1.0f - barys.x - barys.y;
        float2 hitUv = v0.texCoord * bary0 + v1.texCoord * barys.x + v2.texCoord * barys.y;
        hitNormal = normalize(mul(v0.normal * bary0 + v1.normal * barys.x + v2.normal * barys.y, (float3x3)nodeData.world));

        float4 hitAlbedo = mat.baseColorFactor;
        if (mat.baseColorTextureIndex >= 0) {
            hitAlbedo *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
        }

        float hitMetallic = mat.metallicFactor;
        float hitRoughness = mat.roughnessFactor;
        if (mat.metallicRoughnessTextureIndex >= 0) {
            float4 mrSample = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
            hitRoughness *= mrSample.g;
            hitMetallic *= mrSample.b;
        }

        hitPos = ray.Origin + ray.Direction * q.CommittedRayT();
        float3 hitViewDir = -ray.Direction;

        // Direct Lighting at hit point
        float3 directLighting = GetDirectLightingHybrid(hitPos, hitNormal, hitViewDir, hitAlbedo.rgb, hitMetallic, hitRoughness, g_Scene, g_Lights, g_Frame.numLights, g_Frame, true, rng);

        // Multi-bounce: Sample Current IrCache
        float3 hitUVW = WorldToIrCacheUVW(hitPos);
        float3 indirectLighting = g_CurrIrCache.SampleLevel(g_LinearSampler, hitUVW, 0).rgb;
        
        sampleRadiance = (directLighting + indirectLighting) * hitAlbedo.rgb;
    }

    // 2. Create Initial Reservoir
    Reservoir r = (Reservoir)0;
    Surface s;
    s.worldPos = worldPos;
    s.normal = normal;
    s.viewDir = viewDir;
    s.albedo = albedo;
    s.metallic = metallic;
    s.roughness = roughness;

    float targetPDF = GetTargetPDF(s, hitPos, sampleRadiance);
    
    if (targetPDF > 0.0f) {
        r.hitPos = hitPos;
        r.hitNormal = hitNormal;
        r.radiance = sampleRadiance;
        r.targetPDF = targetPDF;
        r.w_sum = (1.0f / pdf) * targetPDF;
        r.M = 1.0f;
    }

    // 3. Temporal Reuse
    float4 prevClipPos = mul(float4(worldPos, 1.0f), g_Frame.viewProjPrevious);
    prevClipPos /= prevClipPos.w;
    float2 prevUV = prevClipPos.xy * float2(0.5f, -0.5f) + 0.5f;
    
    if (prevUV.x >= 0 && prevUV.x <= 1 && prevUV.y >= 0 && prevUV.y <= 1) {
        uint2 prevScreenPos = (uint2)(prevUV * (float2)launchDims);
        Reservoir prevR = g_PrevReservoirs[prevScreenPos.y * launchDims.x + prevScreenPos.x];
        
        if (prevR.M > 0.0f) {
            // Re-evaluate target PDF for history sample at current surface
            float historyTargetPDF = GetTargetPDF(s, prevR.hitPos, prevR.radiance);
            
            if (historyTargetPDF > 0.0f) {
                prevR.targetPDF = historyTargetPDF;
                prevR.M = min(prevR.M, 20.0f); // Cap history length
                
                // Merge reservoirs
                r.M += prevR.M;
                r.w_sum += prevR.w_sum * prevR.targetPDF;
                
                float weight = (prevR.w_sum * prevR.targetPDF) / max(r.w_sum, 1e-6f);
                if (next_float(rng) < weight) {
                    r.hitPos = prevR.hitPos;
                    r.hitNormal = prevR.hitNormal;
                    r.radiance = prevR.radiance;
                    r.targetPDF = prevR.targetPDF;
                }
            }
        }
    }

    // Normalize reservoir weight
    if (r.M > 0.0f && r.targetPDF > 0.0f) {
        r.w_sum = r.w_sum / (r.M * r.targetPDF);
    } else {
        r.w_sum = 0.0f;
    }

    g_CurrReservoirs[pixelIndex] = r;
}
