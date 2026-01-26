
#include "Common.hlsl"

RWTexture2D<float4> g_Output : register(u0);
RaytracingAccelerationStructure g_Scene : register(t2, space1);
StructuredBuffer<DrawNodeData> g_DrawNodeBuffer : register(t1, space1);
StructuredBuffer<MaterialConstants> g_Materials : register(t0, space1);
StructuredBuffer<GLTFVertex> g_GlobalVertices : register(t4, space1);
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
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0f - u.y);
    return float3(cosTheta * cos(phi), cosTheta * sin(phi), sinTheta);
}

float3 align_to_normal(float3 v, float3 n) {
    float3 up = abs(n.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);
    return v.x * tangent + v.y * bitangent + v.z * n;
}

// GLTFVertex GetVertex(PrimitiveData prim, uint vertexIndex) {
//     ByteAddressBuffer buf = g_Buffers[prim.vertexBufferIndex];
//     uint offset = vertexIndex * 32; // sizeof(GLTFVertex) = 32 bytes (3*4 + 3*4 + 2*4)
//     GLTFVertex v;
//     v.position = asfloat(buf.Load3(offset));
//     v.normal = asfloat(buf.Load3(offset + 12));
//     v.texCoord = asfloat(buf.Load2(offset + 24));
//     return v;
// }

// uint GetIndex(PrimitiveData prim, uint triangleIndex, uint vertexInTriangle) {
//     ByteAddressBuffer buf = g_Buffers[prim.indexBufferIndex];
//     // Assuming 32-bit indices for now. If 16-bit, this needs change.
//     return buf.Load((triangleIndex * 3 + vertexInTriangle) * 4);
// }

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 launchIndex = dispatchThreadID.xy;
    uint2 launchDims;
    g_Output.GetDimensions(launchDims.x, launchDims.y);

    if (launchIndex.x >= launchDims.x || launchIndex.y >= launchDims.y) return;

    // Camera ray generation
    float2 d = (((float2)launchIndex + 0.5f) / (float2)launchDims) * 2.0f - 1.0f;
    d.y = -d.y;

    float4 target = mul(float4(d, 1.0f, 1.0f), g_Frame.projectionInverse);
    float3 rayDir = mul(float4(normalize(target.xyz / target.w), 0.0f), g_Frame.viewInverse).xyz;
    float3 rayPos = g_Frame.cameraPosition.xyz;

    RayDesc ray;
    ray.Origin = rayPos;
    ray.Direction = rayDir;
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;

    // Use RayQuery for an inline "Hello World" trace
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(g_Scene, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    float3 finalColor = float3(0.05f, 0.05f, 0.2f); // Default background

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        // Get hit information
        uint instanceIdx = q.CommittedInstanceID();
        float2 barys = q.CommittedTriangleBarycentrics();
        
        // Fetch primitive data for this instance
        //PrimitiveData prim = g_Primitives[instanceIdx];
        DrawNodeData nodeData = g_DrawNodeBuffer[instanceIdx];
        
        // Basic visualization: iterate colors based on material index
        uint matIdx = nodeData.materialID;
        float3 baseColor = float3(0.5f, 0.5f, 0.5f);
        
        // If we can access materials, use the base color factor
        // g_Materials is bound to t0 space1 in root signature
        baseColor = g_Materials[matIdx].baseColorFactor.rgb;

        // Simple world-space normal visualization if we had vertices
        // Since fetching vertices is more complex, let's just do N dot L with a dummy light
        float3 dummyLightDir = normalize(float3(1, 1, -1));
        
        // We can't easily get the geometry normal without fetching vertices and indices
        // but for testing AS validataion, visualizing InstanceID is very useful:
        float3 instanceColor = float3(
            (instanceIdx * 0.13f) % 1.0f,
            (instanceIdx * 0.27f) % 1.0f,
            (instanceIdx * 0.53f) % 1.0f
        );

        finalColor = baseColor * instanceColor;
    }

    g_Output[launchIndex] = float4(finalColor, 1.0f);
}
    //             PrimitiveData prim = g_Primitives[geomIdx];
    //             MaterialConstants mat = g_Materials[prim.materialIndex];
                
    //             if (mat.baseColorTextureIndex >= 0) {
    //                 uint triIdx = q.CandidatePrimitiveIndex();
    //                 float2 bary = q.CandidateTriangleBarycentrics();
                    
    //                 uint i0 = GetIndex(prim, triIdx, 0);
    //                 uint i1 = GetIndex(prim, triIdx, 1);
    //                 uint i2 = GetIndex(prim, triIdx, 2);
                    
    //                 float2 uv0 = GetVertex(prim, i0).texCoord;
    //                 float2 uv1 = GetVertex(prim, i1).texCoord;
    //                 float2 uv2 = GetVertex(prim, i2).texCoord;
                    
    //                 float2 uv = uv0 * (1.0f - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
    //                 float alpha = g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, uv, 0).a;
                    
    //                 if (alpha < 0.5f) q.IgnoreHit();
    //                 else q.CommitCandidateHit();
    //             } else {
    //                 q.CommitCandidateHit();
    //             }
    //         }
    //     }

    //     if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    //         uint geomIdx = q.CommittedGeometryIndex();
    //         uint triIdx = q.CommittedPrimitiveIndex();
    //         float2 bary = q.CommittedTriangleBarycentrics();
            
    //         PrimitiveData prim = g_Primitives[geomIdx];
    //         MaterialConstants mat = g_Materials[prim.materialIndex];
            
    //         uint i0 = GetIndex(prim, triIdx, 0);
    //         uint i1 = GetIndex(prim, triIdx, 1);
    //         uint i2 = GetIndex(prim, triIdx, 2);
            
    //         GLTFVertex v0 = GetVertex(prim, i0);
    //         GLTFVertex v1 = GetVertex(prim, i1);
    //         GLTFVertex v2 = GetVertex(prim, i2);
            
    //         float3 normal = normalize(v0.normal * (1.0f - bary.x - bary.y) + v1.normal * bary.x + v2.normal * bary.y);
    //         float2 uv = v0.texCoord * (1.0f - bary.x - bary.y) + v1.texCoord * bary.x + v2.texCoord * bary.y;
            
    //         float4 baseColor = mat.baseColorFactor;
    //         if (mat.baseColorTextureIndex >= 0) {
    //             baseColor *= g_Textures[mat.baseColorTextureIndex].SampleLevel(g_LinearSampler, uv, 0);
    //         }

    //         rayPos = ray.Origin + ray.Direction * q.CommittedRayT();
            
    //         // NEE (Next Event Estimation)
    //         {
    //             float3 lightDir = -g_Light.direction.xyz;
                
    //             // Add jitter for soft shadows (angular diameter approx 0.5 degrees like the sun)
    //             float3 spread = float3(next_float(rng) - 0.5f, next_float(rng) - 0.5f, next_float(rng) - 0.5f);
    //             lightDir = normalize(lightDir + spread * 0.01f); 

    //             RayDesc shadowRay;
    //             shadowRay.Origin = rayPos + normal * 0.001f;
    //             shadowRay.Direction = lightDir;
    //             shadowRay.TMin = 0.001f;
    //             shadowRay.TMax = 10000.0f;

    //             RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    //             sq.TraceRayInline(g_Scene, RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, 0xFF, shadowRay);
                
    //             while(sq.Proceed()) {
    //                  if (sq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
    //                     uint sGeomIdx = sq.CandidateGeometryIndex();
    //                     PrimitiveData sPrim = g_Primitives[sGeomIdx];
    //                     MaterialConstants sMat = g_Materials[sPrim.materialIndex];
                        
    //                     if (sMat.baseColorTextureIndex >= 0) {
    //                         uint sTriIdx = sq.CandidatePrimitiveIndex();
    //                         float2 sBary = sq.CandidateTriangleBarycentrics();
    //                         uint si0 = GetIndex(sPrim, sTriIdx, 0);
    //                         uint si1 = GetIndex(sPrim, sTriIdx, 1);
    //                         uint si2 = GetIndex(sPrim, sTriIdx, 2);
    //                         float2 suv0 = GetVertex(sPrim, si0).texCoord;
    //                         float2 suv1 = GetVertex(sPrim, si1).texCoord;
    //                         float2 suv2 = GetVertex(sPrim, si2).texCoord;
    //                         float2 suv = suv0 * (1.0f - sBary.x - sBary.y) + suv1 * sBary.x + suv2 * sBary.y;
    //                         float sAlpha = g_Textures[sMat.baseColorTextureIndex].SampleLevel(g_LinearSampler, suv, 0).a;
    //                         if (sAlpha < 0.5f) sq.IgnoreHit();
    //                         else sq.CommitCandidateHit();
    //                     } else {
    //                         sq.CommitCandidateHit();
    //                     }
    //                  }
    //             }

    //             if (sq.CommittedStatus() == COMMITTED_NOTHING) {
    //                 float ndotl = max(0.0f, dot(normal, lightDir));
    //                 accumulatedColor += throughput * baseColor.xyz * g_Light.color.xyz * ndotl;
    //             }
    //         }

    //         // Path continuation
    //         throughput *= baseColor.xyz;
    //         float3 nextDirLocal = sample_cosine_weighted(float2(next_float(rng), next_float(rng)));
    //         rayDir = align_to_normal(nextDirLocal, normal);
    //         rayPos += normal * 0.001f;
    //     } else {
    //         // Environment/Sky color
    //         accumulatedColor += throughput * float3(0.5, 0.7, 1.0) * 0.5; // Simple sky
    //         break;
    //     }

    //     // Russian Roulette
    //     if (bounce > 1) {
    //         float p = max(throughput.r, max(throughput.g, throughput.b));
    //         if (next_float(rng) > p) break;
    //         throughput /= p;
    //     }
    // }

    // g_Output[launchIndex] = float4(accumulatedColor, 1.0f);
