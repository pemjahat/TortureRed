#include "IrCache_Common.hlsl"
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);

// Inputs
Texture3D<float4> g_PrevIrCache : register(t0, space3);
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

// Outputs
RWTexture3D<float4> g_CurrIrCache : register(u0);

[numthreads(8, 8, 8)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= IRCACHE_GRID_SIZE_X || DTid.y >= IRCACHE_GRID_SIZE_Y || DTid.z >= IRCACHE_GRID_SIZE_Z)
        return;

    float3 probePos = GridIndexToWorld(DTid);
    
    // Initialize RNG for this probe
    RNG rng;
    seed_rng(rng, DTid.xy + DTid.z * 100, g_Frame.frameIndex);

    float3 totalIrradiance = 0.0f;
    float validRays = 0.0f;

    // Trace rays spherically around the probe
    for (int i = 0; i < IRCACHE_RAYS_PER_PROBE; ++i)
    {
        // Generate uniform spherical direction
        float2 u = float2(next_float(rng), next_float(rng));
        float z = 1.0f - 2.0f * u.x;
        float r = sqrt(max(0.0f, 1.0f - z * z));
        float phi = 2.0f * 3.14159265f * u.y;
        float3 rayDir = float3(r * cos(phi), r * sin(phi), z);

        RayDesc ray;
        ray.Origin = probePos;
        ray.Direction = rayDir;
        ray.TMin = 0.01f;
        ray.TMax = 1000.0f;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
        while (q.Proceed()) {
            PROCESS_ALPHA_MASK(q, rng);
        }

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
            float3 worldNormal = normalize(mul(v0.normal * bary0 + v1.normal * barys.x + v2.normal * barys.y, (float3x3)nodeData.world));

            float4 albedo = mat.baseColorFactor;
            if (mat.baseColorTextureIndex >= 0) {
                albedo *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
            }

            float metallic = mat.metallicFactor;
            float roughness = mat.roughnessFactor;
            if (mat.metallicRoughnessTextureIndex >= 0) {
                float4 mrSample = g_Textures[mat.metallicRoughnessTextureIndex].SampleLevel(g_LinearSampler, hitUv, 0);
                roughness *= mrSample.g;
                metallic *= mrSample.b;
            }

            float3 hitPos = ray.Origin + ray.Direction * q.CommittedRayT();
            float3 viewDir = -ray.Direction;

            // 1. Direct Lighting at hit point
            float3 directLighting = GetDirectLightingHybrid(hitPos, worldNormal, viewDir, albedo.rgb, metallic, roughness, g_Scene, g_Lights, g_Frame.numLights, g_Frame, true, rng);

            // 2. Multi-bounce: Sample previous IrCache
            float3 hitUVW = WorldToIrCacheUVW(hitPos);
            float3 indirectLighting = g_PrevIrCache.SampleLevel(g_LinearSampler, hitUVW, 0).rgb;
            
            // Simple diffuse bounce approximation
            float3 bounceRadiance = (directLighting + indirectLighting) * albedo.rgb;
            
            totalIrradiance += bounceRadiance;
            validRays += 1.0f;
        }
    }

    float3 newIrradiance = validRays > 0.0f ? (totalIrradiance / validRays) : 0.0f;

    // Exponential moving average with previous frame
    float3 prevIrradiance = g_PrevIrCache[DTid].rgb;
    float blendFactor = 0.05f; // 5% new, 95% old
    
    // If this is the first frame or history is invalid, snap to new value
    if (g_Frame.frameIndex == 0) {
        blendFactor = 1.0f;
    }

    float3 finalIrradiance = lerp(prevIrradiance, newIrradiance, blendFactor);
    
    g_CurrIrCache[DTid] = float4(finalIrradiance, 1.0f);
}
