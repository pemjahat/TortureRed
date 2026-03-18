#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<BindlessIndices> g_Indices : register(b1);
StructuredBuffer<LightConstants> g_Lights : register(t0, space2);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims = uint2(g_Frame.screenWidth, g_Frame.screenHeight);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    LightConstants mainLight = g_Lights[0];

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    // Accessing texture bindless
    RWTexture2D<float4> accumulationBuffer = ResourceDescriptorHeap[g_Indices.OutputIdx0];
    RWTexture2D<float4> outputBuffer = ResourceDescriptorHeap[g_Indices.OutputIdx1];

    float3 accumulatedColor = 0;
    float3 indirectRadianceAccum = 0;
    float3 throughput = 1;

    Surface primarySurface;
    float primaryRayT;
    bool hasPrimaryHit = TracePrimarySurface(launchIndex, launchDims, g_Frame, rng, primarySurface, primaryRayT);

    if (!hasPrimaryHit) {
        accumulatedColor = float3(0.5f, 0.7f, 1.0f) * 0.2f;
    } else {
        float3 primaryHitPos = primarySurface.worldPos;
        float3 primaryNormal = primarySurface.normal;
        float3 primaryAlbedo = primarySurface.albedo;
        float primaryRoughness = primarySurface.roughness;
        float primaryMetallic = primarySurface.metallic;
        float3 V = primarySurface.viewDir;
        bool isPathDiffuse = false;

        // --- Step 1: Direct Lighting for Primary Hit ---
        // Dispatch based on lightSamplingMode: 0=Uniform, 1=ImportancePDF, 2=BruteForce
        accumulatedColor = GetDirectLightingHybrid(primaryHitPos, primaryNormal, V, primaryAlbedo,
                                                   primaryMetallic, primaryRoughness, g_Scene,
                                                   g_Lights, g_Frame.numLights, g_Frame, false, rng);

        // --- Step 2: Sample First Indirect Ray from Primary Hit ---
        float3 rayDir;
        float3 firstThroughput;
        float firstPDF;
        SampleIndirectRay(primaryNormal, V, primaryAlbedo, primaryMetallic, primaryRoughness, rng, rayDir, firstThroughput, firstPDF, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

        throughput *= firstThroughput;
        float3 rayPos = primaryHitPos + primaryNormal * 0.001f;

        // --- Step 3: Indirect Bounces ---
        float next_pdf = firstPDF;
        for (int bounce = 1; bounce < 4; bounce++) {
            if (all(throughput <= 0.0f)) break;

            RayDesc ray;
            ray.Origin = rayPos;
            ray.Direction = rayDir;
            ray.TMin = 0.001f;
            ray.TMax = 10000.0f;

            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
            while (q.Proceed()) {
                PROCESS_ALPHA_MASK(q, rng);
            }

            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                Surface hitSurf;
                ResolveHitSurface(ray, q.CommittedRayT(), q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), hitSurf, 0.15f);

                // NEE - Stochastic light sampling for indirect bounces (with MIS)
                float3 ndl = GetDirectLightingHybrid(hitSurf.worldPos, hitSurf.normal, hitSurf.viewDir, hitSurf.albedo,
                                                    hitSurf.metallic, hitSurf.roughness, g_Scene,
                                                    g_Lights, g_Frame.numLights, g_Frame, isPathDiffuse, rng) * throughput;
                indirectRadianceAccum += ndl;

                // Sample next bounce
                float3 nextDir;
                float3 nextThroughput;
                SampleIndirectRay(hitSurf.normal, hitSurf.viewDir, hitSurf.albedo, hitSurf.metallic, hitSurf.roughness, rng, nextDir, nextThroughput, next_pdf, isPathDiffuse, g_Frame.enableIndirectSpecular != 0);

                throughput *= nextThroughput;
                rayPos = hitSurf.worldPos + hitSurf.normal * 0.001f;
                rayDir = nextDir;

                // Russian Roulette
                if (bounce > 2) {
                    float p = max(throughput.r, max(throughput.g, throughput.b));
                    if (next_float(rng) > p) break;
                    throughput /= p;
                }
            } else {
                float3 skyRadiance = float3(0.5f, 0.7f, 1.0f) * 0.2f;
                indirectRadianceAccum += skyRadiance * throughput;
                break;
            }
        }
    }

    accumulatedColor += indirectRadianceAccum;

    // Progressive accumulation
    if (g_Frame.frameIndex <= 1) {
        accumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = accumulationBuffer[launchIndex].rgb;
        float n = (float)g_Frame.frameIndex;
        float lerpFactor = min( (n - 1.0f) / min(n, 2000.0f), 1.0f );  // Clamp to <= 1
        accumulatedColor = lerp(accumulatedColor, prevColor, lerpFactor);
        accumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
    }

    float3 exposedColor = accumulatedColor * g_Frame.exposure;
    outputBuffer[launchIndex] = float4(exposedColor / (exposedColor + 1.0f), 1.0f);
}
