
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

float3 FresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265 * denom * denom;
    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * 3.14159265 * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
    return align_to_normal(H, N);
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

    for (int bounce = 0; bounce < 4; bounce++) {
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

            float metallic = mat.metallicFactor;
            float roughness = max(0.01f, mat.roughnessFactor);
            float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
            float3 V = -ray.Direction;

            // NEE for Directional Light
            {
                float3 L = -normalize(g_Light.direction.xyz);
                float3 H = normalize(V + L);
                float ndotl = max(0.0001f, dot(worldNormal, L));
                float ndotv = max(0.0001f, dot(worldNormal, V));

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
                        float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
                        float D = DistributionGGX(worldNormal, H, roughness);
                        float G = GeometrySmith(worldNormal, V, L, roughness);
                        
                        float3 numerator = D * G * F;
                        float denominator = 4.0 * ndotv * ndotl + 0.0001;
                        float3 specular = numerator / denominator;
                        
                        float3 kD = (1.0 - F) * (1.0 - metallic);
                        float3 diffuse = kD * baseColor.rgb / 3.14159265;
                        
                        accumulatedColor += throughput * (diffuse + specular) * g_Light.color.rgb * g_Light.intensity * ndotl;
                    }
                }
            }

            // Path continuation (stochastic selection between diffuse and specular)
            float3 F_prob = FresnelSchlick(max(dot(worldNormal, V), 0.0), F0);
            float probSpecular = clamp(max(F_prob.r, max(F_prob.g, F_prob.b)), 0.1, 0.9);
            float rnd = next_float(rng);

            if (rnd < probSpecular) {
                // Specular reflection (Importance sampling GGX)
                float3 H = ImportanceSampleGGX(float2(next_float(rng), next_float(rng)), worldNormal, roughness);
                rayDir = reflect(-V, H);
                
                float VdotH = max(dot(V, H), 0.0);
                float NdotV = max(dot(worldNormal, V), 0.0001);
                float NdotH = max(dot(worldNormal, H), 0.0001);
                float G = GeometrySmith(worldNormal, V, rayDir, roughness);
                float3 F_spec = FresnelSchlick(VdotH, F0);
                
                throughput *= (F_spec * G * VdotH) / (NdotV * NdotH * probSpecular);
            } else {
                // Diffuse reflection (Cosine weighted sampling)
                float3 nextDirLocal = sample_cosine_weighted(float2(next_float(rng), next_float(rng)));
                rayDir = align_to_normal(nextDirLocal, worldNormal);
                
                float3 F_at_surface = FresnelSchlick(max(dot(worldNormal, V), 0.0), F0);
                float3 kD = (1.0 - F_at_surface) * (1.0 - metallic);
                throughput *= (kD * baseColor.rgb) / (1.0 - probSpecular);
            }

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
        
        float3 exposedColor = accumulatedColor * g_Frame.exposure;
        float3 ldrColor = exposedColor / (exposedColor + 1.0f);
        g_Output[launchIndex] = float4(ldrColor, 1.0f);
    } else {
        float3 prevColor = g_AccumulationBuffer[launchIndex].rgb;
        
        // Progressive accumulation logic similar to DXRPathTracer
        float n = (float)g_Frame.frameIndex;
        // Limit n to prevent floating point precision issues at very high frame counts
        float lerpFactor = (n - 1.0f) / n;
        if (n > 2000.0f) lerpFactor = 1999.0f / 2000.0f;

        float3 finalColor = lerp(accumulatedColor, prevColor, lerpFactor);
        
        g_AccumulationBuffer[launchIndex] = float4(finalColor, 1.0f);
        
        // Basic Tone Mapping for output
        float3 exposedColor = finalColor * g_Frame.exposure;
        float3 ldrColor = exposedColor / (exposedColor + 1.0f);
        
        g_Output[launchIndex] = float4(ldrColor, 1.0f);
    }
}
