#include "CommonTracing.hlsl"
#include "IrCache_Lookup.hlsl"

ConstantBuffer<FrameConstants>         g_Frame   : register(b0);
ConstantBuffer<BindlessIndices>        g_Indices : register(b1);
ConstantBuffer<IrCacheBindlessIndices> g_IrCache : register(b2);

StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 screenPos = DTid.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (screenPos.x >= launchDims.x || screenPos.y >= launchDims.y) return;

    uint pixelIndex = screenPos.y * launchDims.x + screenPos.x;
    
    // Initialize RNG
    RNG rng;
    seed_rng(rng, screenPos, g_Frame.frameIndex);

    // Accessing reservoirs bindless
    StructuredBuffer<Reservoir>    prevReservoirs = ResourceDescriptorHeap[g_Indices.InputIdx0];
    RWStructuredBuffer<Reservoir>  currReservoirs = ResourceDescriptorHeap[g_Indices.OutputIdx0];

    Surface surface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(screenPos, launchDims, g_Frame, rng, surface, primaryRayT);
    if (!hasPrimaryHit) {
        currReservoirs[pixelIndex] = (Reservoir)0;
        return;
    }

    // 1. Trace Single Indirect Bounce
    float3 rayDir;
    float3 throughput;
    float pdf;
    bool isDiffuse;
    SampleIndirectRay(surface.normal, surface.viewDir, surface.albedo, surface.metallic, surface.roughness, rng, rayDir, throughput, pdf, isDiffuse, g_Frame.enableIndirectSpecular != 0);

    RayDesc ray;
    ray.Origin = surface.worldPos + surface.normal * 0.001f;
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

        // Multi-bounce: sample spatial irradiance cache at hit point
        float3 indirectLighting = SampleIrCache(hitPos, g_IrCache, g_Frame.cameraPosition.xyz);
        // Demand-allocate a probe here so the cache populates on subsequent frames
        IrCacheMaybeAllocate(hitPos, g_IrCache, g_Frame.cameraPosition.xyz);
        
        sampleRadiance = (directLighting + indirectLighting) * hitAlbedo.rgb;
    }

    // 2. Create Initial Reservoir
    //Reservoir r = (Reservoir)0;
    Reservoir r;
    r.hitPos = 0; r.hitNormal = 0; r.radiance = 0;
    r.w_sum = 0; r.W = 0; r.M = 0;

    Surface s;
    s.worldPos = surface.worldPos;
    s.normal = surface.normal;
    s.viewDir = surface.viewDir;
    s.albedo = surface.albedo;
    s.metallic = surface.metallic;
    s.roughness = surface.roughness;

    float selectedPDF = 0.f;
    float targetPDF = GetTargetPDF(s, hitPos, sampleRadiance);
    float risWeight = (pdf > 0.0f) ? (targetPDF / pdf) : 0.0f;
    if (updateReservoir(r, hitPos, hitNormal, sampleRadiance, risWeight, next_float(rng))) {
        selectedPDF = targetPDF;
    }

    // 3. Temporal Reuse
    float4 prevClipPos = mul(float4(surface.worldPos, 1.0f), g_Frame.viewProjPrevious);
    prevClipPos /= prevClipPos.w;
    float2 prevUV = prevClipPos.xy * float2(0.5f, -0.5f) + 0.5f;
    
    if (prevUV.x >= 0 && prevUV.x <= 1 && prevUV.y >= 0 && prevUV.y <= 1) {
        uint2 prevScreenPos = (uint2)(prevUV * (float2)launchDims);
        Reservoir prevR = prevReservoirs[prevScreenPos.y * launchDims.x + prevScreenPos.x];
        
        if (prevR.M > 0.0f) {
            // Re-evaluate target PDF for history sample at current surface
            float historyTargetPDF = GetTargetPDF(s, prevR.hitPos, prevR.radiance);
            
            if (historyTargetPDF > 0.0f) {

                if (mergeReservoirs(r, prevR, historyTargetPDF, next_float(rng))) {
                    selectedPDF = historyTargetPDF;
                }

            }
            if (r.M > 30.0f) { 
                r.w_sum *= (30.0f / r.M); 
                r.M = 30.0f; 
            }
        }
    }

    // Normalize reservoir weight
    if (r.M > 0.0f && selectedPDF > 0.0f) {
        r.W = r.w_sum / (r.M * selectedPDF);
    } else {
        r.W = 0.0f;
    }

    currReservoirs[pixelIndex] = r;
}
