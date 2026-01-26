
#include "Common.hlsl"

RWTexture2D<float4> g_AccumulationBuffer : register(u0);
RWTexture2D<float4> g_Output : register(u1);
RaytracingAccelerationStructure g_Scene : register(t2, space1);
StructuredBuffer<DrawNodeData> g_DrawNodeBuffer : register(t1, space1);
StructuredBuffer<MaterialConstants> g_Materials : register(t0, space1);
StructuredBuffer<GLTFVertex> g_GlobalVertices : register(t4, space1);
StructuredBuffer<uint> g_GlobalIndices : register(t3, space1);
ByteAddressBuffer g_Buffers[] : register(t0, space2);
Texture2D g_Textures[] : register(t0, space0);

ConstantBuffer<FrameConstants> g_Frame : register(b0);
ConstantBuffer<LightConstants> g_Light : register(b1);

SamplerState g_LinearSampler : register(s0);

// Random number generator (PCG)
struct RNG {
    uint state;
    uint inc;
};

uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float next_float(inout RNG rng) {
    rng.state = rng.state * 747796405u + 1u;
    uint res = pcg_hash(rng.state);
    return float(res) / 4294967296.0f;
}

float3 sample_cosine_weighted(float2 u) {
    float phi = 2.0f * 3.14159265f * u.x;
    float sinTheta = sqrt(u.y);
    float cosTheta = sqrt(1.0f - u.y);
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float3 align_to_normal(float3 v, float3 n) {
    float3 up = abs(n.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);
    return v.x * tangent + v.y * bitangent + v.z * n;
}

void seed_rng(out RNG rng, uint2 screenPos, uint frameIndex) {
    rng.state = pcg_hash(screenPos.y * 65536 + screenPos.x + pcg_hash(frameIndex));
    rng.inc = 1;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    RNG rng;
    seed_rng(rng, launchIndex, g_Frame.frameIndex);

    // Camera ray generation with jitter for anti-aliasing
    float2 subpixel = float2(next_float(rng), next_float(rng));
    float2 d = (((float2)launchIndex + subpixel) / (float2)launchDims) * 2.0f - 1.0f;
    // Screen space to NDC
    d.y = -d.y;

    // NDC to world space
    float4 target = mul(float4(d, 1.0f, 1.0f), g_Frame.projectionInverse);
    float3 rayDir = mul(float4(normalize(target.xyz / target.w), 0.0f), g_Frame.viewInverse).xyz;
    float3 rayPos = g_Frame.cameraPosition.xyz;

    float3 accumulatedColor = 0;    // store total light energy (radiance) reaches camera along path
    float3 throughput = 1;  // represent percentage light "survives" after bounce surface along path

    for (int bounce = 0; bounce < 3; bounce++) {
        RayDesc ray;
        ray.Origin = rayPos;
        ray.Direction = rayDir;
        ray.TMin = 0.001f;
        ray.TMax = 10000.0f;

        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            uint instanceIdx = q.CommittedInstanceID();
            uint triIdx = q.CommittedPrimitiveIndex();
            float2 barys = q.CommittedTriangleBarycentrics();
            
            DrawNodeData nodeData = g_DrawNodeBuffer[instanceIdx];
            MaterialConstants mat = g_Materials[nodeData.materialID];

            // Fetch vertices and interpolate
            uint i0 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 0];
            uint i1 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 1];
            uint i2 = g_GlobalIndices[nodeData.indexOffset + triIdx * 3 + 2];

            GLTFVertex v0 = g_GlobalVertices[nodeData.vertexOffset + i0];
            GLTFVertex v1 = g_GlobalVertices[nodeData.vertexOffset + i1];
            GLTFVertex v2 = g_GlobalVertices[nodeData.vertexOffset + i2];

            // Reconstruct attributes
            // WorldNormal ~ for lighting calculations
            // UV ~ sample material textures
            // hitPos ~ origin for next ray
            float3 worldNormal = normalize(mul(v0.normal * (1.0f - barys.x - barys.y) + v1.normal * barys.x + v2.normal * barys.y, (float3x3)nodeData.world));
            float2 uv = v0.texCoord * (1.0f - barys.x - barys.y) + v1.texCoord * barys.x + v2.texCoord * barys.y;
            float3 hitPos = ray.Origin + ray.Direction * q.CommittedRayT();

            float4 baseColor = mat.baseColorFactor;
            if (mat.baseColorTextureIndex >= 0) {
                baseColor *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, uv, 0);
            }

            // NEE for Directional Light
            {
                float3 L = -normalize(g_Light.direction.xyz);
                float ndotl = max(0.0f, dot(worldNormal, L));
                if (ndotl > 0) {
                    RayDesc shadowRay;
                    shadowRay.Origin = hitPos + worldNormal * 0.001f;
                    shadowRay.Direction = L;
                    shadowRay.TMin = 0.001f;
                    shadowRay.TMax = 10000.0f;

                    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
                    sq.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, shadowRay);
                    sq.Proceed();

                    if (sq.CommittedStatus() == COMMITTED_NOTHING) {
                        accumulatedColor += throughput * baseColor.xyz * g_Light.color.xyz * 10.f * ndotl * (1.0f / 3.14159f);
                    }
                }
            }

            // Path continuation (diffuse reflection)
            throughput *= baseColor.xyz;
            // Cosine weighted ensure rays more likely bounce towards normal direction
            float3 nextDirLocal = sample_cosine_weighted(float2(next_float(rng), next_float(rng)));
            // Convert sample direction from local "up" to surface normal
            rayDir = align_to_normal(nextDirLocal, worldNormal);
            rayPos = hitPos + worldNormal * 0.001f;

            // Russian Roulette
            if (bounce > 2) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                if (next_float(rng) > p) break;
                throughput /= p;
            }
        } else {
            // Environment / Simple Sky
            accumulatedColor += throughput * float3(0.5f, 0.7f, 1.0f) * 0.2f;
            break;
        }
    }

    if (g_Frame.frameIndex <= 1) {
        g_AccumulationBuffer[launchIndex] = float4(accumulatedColor, 1.0f);
        g_Output[launchIndex] = float4(accumulatedColor, 1.0f);
    } else {
        float3 prevColor = g_AccumulationBuffer[launchIndex].rgb;
        
        // Progressive accumulation logic similar to DXRPathTracer
        float n = (float)g_Frame.frameIndex;
        // Limit n to prevent floating point precision issues at very high frame counts
        float lerpFactor = (n - 1.0f) / n;
        if (n > 2000.0f) lerpFactor = 1999.0f / 2000.0f;

        float3 finalColor = lerp(accumulatedColor, prevColor, lerpFactor);
        
        g_AccumulationBuffer[launchIndex] = float4(finalColor, 1.0f);
        
        // Final output (could add tonemapping here)
        g_Output[launchIndex] = float4(finalColor, 1.0f);
    }
}
