#ifndef PBR_HLSL
#define PBR_HLSL

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

// Evaluate the PBR BSDF (Diffuse and Specular components)
void EvaluateBSDF(float3 N, float3 V, float3 L, float3 baseColor, float metallic, float roughness, out float3 diffuseBRDF, out float3 specularBRDF) {
    float3 H = normalize(V + L);
    float dotNL = max(0.0001f, dot(N, L));
    float dotNV = max(0.0001f, dot(N, V));
    float dotVH = max(0.0001f, dot(V, H));

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float3 F = FresnelSchlick(dotVH, F0);
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);

    specularBRDF = (D * G * F) / (4.0f * dotNV * dotNL + 0.0001f);
    float3 kD = (1.0f - F) * (1.0f - metallic);
    diffuseBRDF = kD * baseColor / 3.14159265f;
}

#endif // PBR_HLSL
