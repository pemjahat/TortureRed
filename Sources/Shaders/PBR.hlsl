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

// ============================================================================
// SH9 Irradiance Evaluation — Tier 2: rasterizer ambient term
// ============================================================================
// The sky system stores irradiance SH coefficients (A_l * L_lm) in a 9-element
// StructuredBuffer. At evaluation time we reconstruct irradiance via:
//   E(N) = Σ_i skySH9[i] * Y_lm(N)
// (plain SH basis — the A_l cosine convolution is already baked into the
// stored coefficients by Sky_ProjectSH9.hlsl).
//
// Reference: Ramamoorthi & Hanrahan 2001, "An Efficient Representation for
//            Irradiance Environment Maps."
// ============================================================================

// Plain real SH basis Y_lm (order-3, 9 coefficients), WITHOUT cosine
// convolution A_l factors. The A_l scaling is pre-applied to the stored
// coefficients during the projection pass.
void EvalSH9Basis(float3 N, out float sh[9])
{
    // Band 0: Y00 = 1/(2*sqrt(pi))
    sh[0] = 0.28209479177387814f;

    // Band 1: Y1m = sqrt(3/(4*pi)) * {y, z, x}
    float c1 = 0.4886025119029199f;
    sh[1] = c1 * N.y;
    sh[2] = c1 * N.z;
    sh[3] = c1 * N.x;

    // Band 2
    sh[4] = 1.0925484305920792f * N.x * N.y;                    // Y2,-2
    sh[5] = 1.0925484305920792f * N.y * N.z;                    // Y2,-1
    sh[6] = 0.31539156525252005f * (3.0f * N.z * N.z - 1.0f);   // Y2,0
    sh[7] = 1.0925484305920792f * N.x * N.z;                    // Y2,+1
    sh[8] = 0.5462742152960396f  * (N.x * N.x - N.y * N.y);     // Y2,+2
}

// Evaluates SH9 irradiance at normal N from a structured buffer of 9 float4
// coefficients. The coefficients are pre-convolved with the cosine kernel.
// Returns RGB irradiance (HDR). Multiply by albedo/PI for Lambertian ambient.
float3 EvalSH9Irradiance(float3 N, StructuredBuffer<float4> skySH9)
{
    float basis[9];
    EvalSH9Basis(N, basis);
    float3 irradiance = 0.0f;
    for (int i = 0; i < 9; ++i)
        irradiance += skySH9[i].rgb * basis[i];
    return max(irradiance, 0.0f);
}

// Overload that takes a bindless index into ResourceDescriptorHeap.
float3 EvalSH9IrradianceIndex(float3 N, uint skySH9BufferIndex)
{
    StructuredBuffer<float4> skySH9 = ResourceDescriptorHeap[skySH9BufferIndex];
    return EvalSH9Irradiance(N, skySH9);
}

#endif // PBR_HLSL
