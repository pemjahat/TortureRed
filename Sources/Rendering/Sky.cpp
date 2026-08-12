#include "pch.h"

#include "Sky.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"
#include "ArHosekSkyModel.h"

#include <cmath>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>

using namespace DirectX;

// ---------------------------------------------------------------------------
// Hosek-Wilkie helpers
// ---------------------------------------------------------------------------

namespace
{
    // Fixed defaults per the task spec — no runtime UI in this pass.
    constexpr double kDefaultTurbidity    = 3.0;
    constexpr double kDefaultGroundAlbedo = 0.3;

    // Physical sun half-angle (radians): 0.27° angular radius.
    // Solid angle Ω = 2π(1 - cos(α)) ≈ π · α² for small α.
    constexpr double kPhysicalSunAngularRadius = 0.00471;   // 0.27°
    constexpr double kSunSolidAngle            = 6.97e-5;   // π · α² ≈ 6.97×10⁻⁵ sr

    constexpr double kSunIrradianceBaseScale  = 683.0;  // luminous efficacy, lm/W
    constexpr double kSunIrradianceSceneScale = 100.0;  // scene-relative tuning factor
    constexpr double kSkyRadianceScale = kSunIrradianceBaseScale * kSunIrradianceSceneScale;

    // -----------------------------------------------------------------------
    // CIE 1931 2° Standard Observer (x̄, ȳ, z̄) at the 11 Hosek spectral
    // wavelength centres: 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720 nm.
    // Δλ = 40 nm. Wavelengths below 360 nm contribute negligibly.
    // -----------------------------------------------------------------------
    struct CIEObserver
    {
        double x, y, z;
    };

    constexpr CIEObserver kCIE_1931_2deg[11] =
    {
        //   λ=320nm (UV, invisible)
        { 0.0000, 0.0000, 0.0000 },
        //   λ=360nm
        { 0.0001, 0.0000, 0.0006 },
        //   λ=400nm
        { 0.0191, 0.0020, 0.0860 },
        //   λ=440nm
        { 0.3483, 0.0230, 1.7471 },
        //   λ=480nm
        { 0.0956, 0.1390, 0.8130 },
        //   λ=520nm
        { 0.0633, 0.7100, 0.0782 },
        //   λ=560nm
        { 0.5945, 0.9950, 0.0039 },
        //   λ=600nm
        { 1.0622, 0.6310, 0.0008 },
        //   λ=640nm
        { 0.4479, 0.1750, 0.0000 },
        //   λ=680nm
        { 0.0468, 0.0170, 0.0000 },
        //   λ=720nm
        { 0.0114, 0.0041, 0.0000 },
    };

    // XYZ (D65 white point) → sRGB linear matrix.
    // Standard Bradford-adapted sRGB from CIE XYZ.
    inline void XYZToSRGBLinear(double X, double Y, double Z, double& R, double& G, double& B)
    {
        R =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
        G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
        B =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
    }

    // -----------------------------------------------------------------------
    // Cubemap face → world-space ray direction.
    // Follows D3D12 convention: +X, -X, +Y, -Y, +Z, -Z.
    // texelUV in [0,1]^2 per face.
    // -----------------------------------------------------------------------
    XMFLOAT3 CubemapTexelToDirection(uint32_t face, float u, float v)
    {
        // Map [0,1]^2 → [-1,1]^2 with the correct face mapping.
        float uc = 2.0f * u - 1.0f;
        float vc = 2.0f * v - 1.0f;
        XMFLOAT3 dir;
        switch (face)
        {
        case 0: dir = XMFLOAT3( 1.0f, -vc, -uc); break; // +X
        case 1: dir = XMFLOAT3(-1.0f, -vc,  uc); break; // -X
        case 2: dir = XMFLOAT3( uc,  1.0f,  vc); break; // +Y
        case 3: dir = XMFLOAT3( uc, -1.0f, -vc); break; // -Y
        case 4: dir = XMFLOAT3( uc, -vc,  1.0f); break; // +Z
        case 5: dir = XMFLOAT3(-uc, -vc, -1.0f); break; // -Z
        default: dir = XMFLOAT3(0,0,0); break;
        }

        // Normalize
        XMVECTOR vdir = XMLoadFloat3(&dir);
        vdir = XMVector3Normalize(vdir);
        XMStoreFloat3(&dir, vdir);
        return dir;
    }

    // Convert a world-space direction to Hosek-Wilkie (theta, gamma).
    // theta = angle from zenith (0 = zenith, pi/2 = horizon, pi = nadir)
    // gamma = angle between view direction and TO-SUN direction
    // toSunDir should be the negation of the engine light direction.
    void DirectionToThetaGamma(const XMFLOAT3& dir, const XMFLOAT3& toSunDir,
                               double& theta, double& gamma)
    {
        XMVECTOR vDir    = XMLoadFloat3(&dir);
        XMVECTOR vSunDir = XMLoadFloat3(&toSunDir);

        // Zenith is +Y in this project (Y-up).
        XMVECTOR vZenith = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

        float cosTheta = std::max(-1.0f, std::min(1.0f,
            XMVectorGetX(XMVector3Dot(vDir, vZenith))));
        theta = std::acos((double)cosTheta);

        float cosGamma = std::max(-1.0f, std::min(1.0f,
            XMVectorGetX(XMVector3Dot(vDir, vSunDir))));
        gamma = std::acos((double)cosGamma);
    }

    // Solar elevation from sun direction.
    // sunLight.direction stores the direction the light TRAVELS (game convention:
    // from source toward the scene, typically downward).
    // The Hosek model needs the TO-SUN direction (upward, on the celestial sphere).
    // So we negate: toSun = -lightDir.
    double SunElevationFromDirection(const LightConstants& sunLight)
    {
        float toSunY = -sunLight.direction.y;
        return (double)std::asin(std::max(-1.0f, std::min(1.0f, toSunY)));
    }
} // anonymous namespace

// ---------------------------------------------------------------------------
// ComputeSunIrradiance — CPU-side Hosek-Wilkie solar irradiance
//
// Evaluates the spectral solar radiance function arhosekskymodel_solar_radiance()
// at the sun-centre direction (gamma=0, theta = π/2 − elevation), integrates over
// the physical solar disc solid angle, converts the 11-wavelength spectral result
// to sRGB via CIE 1931 2° observer + D65 XYZ→sRGB matrix, and stores the result
// in m_SunIrradiance.
//
// Reference: MJP's DXRPathTracer, SampleFramework12/v1.02/Graphics/Skybox.cpp
//            SkyCache::Init() lines 81-136.
// ---------------------------------------------------------------------------

void Sky::ComputeSunIrradiance(const LightConstants& sunLight, float turbidity, float groundAlbedo)
{
    // --- Elevation & sun-centre angles ---
    double elevation = SunElevationFromDirection(sunLight);
    double thetaS    = (XM_PI / 2.0) - elevation;  // zenith angle of sun centre
    double gamma     = 0.0;                            // looking directly at sun centre

    // --- Allocate spectral sky states for all 11 wavelengths ---
    // NumSpectralSamples = 11 (320, 360, ..., 720 nm)
    static constexpr int kNumSpectralSamples = 11;
    ArHosekSkyModelState* skyStates[kNumSpectralSamples] = {};
    for (int i = 0; i < kNumSpectralSamples; ++i)
    {
        skyStates[i] = arhosekskymodelstate_alloc_init(elevation, (double)turbidity, (double)groundAlbedo); // first param expect elevation, not zenith angle
        if (!skyStates[i])
        {
            // Free already-allocated states on failure
            for (int j = 0; j < i; ++j)
            {
                arhosekskymodelstate_free(skyStates[j]);
                skyStates[j] = nullptr;
            }
            std::cerr << "[Sky] Failed to allocate solar spectral states" << std::endl;
            return;
        }
    }

    // --- Evaluate solar radiance at sun centre for each wavelength ---
    double solarRadiance[kNumSpectralSamples] = {};
    for (int i = 0; i < kNumSpectralSamples; ++i)
    {
        double wavelength = 320.0 + 40.0 * i;  // 320, 360, ..., 720 nm
        solarRadiance[i] = arhosekskymodel_solar_radiance(skyStates[i], thetaS, gamma, wavelength);
        solarRadiance[i] = std::max(0.0, solarRadiance[i]);
    }

    // --- Convert spectral radiance → CIE XYZ via 2° observer ---
    // X = Σ L(λ) · x̄(λ) · Δλ, similarly for Y, Z.  Δλ = 40 nm.
    constexpr double kDeltaLambda = 40.0;  // nm
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (int i = 0; i < kNumSpectralSamples; ++i)
    {
        X += solarRadiance[i] * kCIE_1931_2deg[i].x * kDeltaLambda;
        Y += solarRadiance[i] * kCIE_1931_2deg[i].y * kDeltaLambda;
        Z += solarRadiance[i] * kCIE_1931_2deg[i].z * kDeltaLambda;
    }

    // --- XYZ → sRGB linear ---
    double R, G, B;
    XYZToSRGBLinear(X, Y, Z, R, G, B);

    // Clamp to non-negative (sRGB gamut clipping handles negatives poorly)
    R = std::max(0.0, R);
    G = std::max(0.0, G);
    B = std::max(0.0, B);

    // --- Convert radiance (W·m⁻²·sr⁻¹) → irradiance (W·m⁻²) ---
    // Multiply by sun solid angle (sr). Surface perpendicular to the sun.
    R *= kSunSolidAngle;
    G *= kSunSolidAngle;
    B *= kSunSolidAngle;

    // --- Scale to match rendering unit expectations ---
    // The Hosek output is in physical W/m².  The renderer uses arbitrary linear
    // units.  We apply luminous efficacy (683 lm/W for photopic vision).
    // Tune kSunIrradianceSceneScale to taste — larger = brighter sun.
    R *= kSkyRadianceScale;
    G *= kSkyRadianceScale;
    B *= kSkyRadianceScale;

    // --- Pre-scale for fp16 HDR range ---
    // Multiply by FP16Scale so downstream GPU fp16 writes stay in range.
    // Inverted at exposure time via exp2(exposure + 10).
    R *= FP16Scale;
    G *= FP16Scale;
    B *= FP16Scale;

    // --- Free spectral states ---
    for (int i = 0; i < kNumSpectralSamples; ++i)
    {
        arhosekskymodelstate_free(skyStates[i]);
        skyStates[i] = nullptr;
    }

    m_SunIrradiance = DirectX::XMFLOAT3((float)R, (float)G, (float)B);

    std::cout << "[Sky] Sun irradiance computed: R=" << m_SunIrradiance.x
              << " G=" << m_SunIrradiance.y << " B=" << m_SunIrradiance.z
              << " (turbidity=" << turbidity << ", albedo=" << groundAlbedo
              << ", elevation=" << elevation << " rad)" << std::endl;
}

// ---------------------------------------------------------------------------
// CreateResources
// ---------------------------------------------------------------------------

void Sky::CreateResources(ID3D12Device* device, uint32_t /*internalWidth*/, uint32_t /*internalHeight*/)
{
    m_ResourcesCreated = false;

    // --- 128×128×6 cubemap, R16G16B16A16_FLOAT ---
    if (!CreateTexture(m_SkyCubemap, kCubemapSize, kCubemapSize, kCubemapFormat,
                       D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COMMON, nullptr, 1, kCubemapFaces,
                       "Tex_SkyCubemap", true))  // isCubemap -> SRV_DIMENSION_TEXTURECUBE
    {
        std::cerr << "[Sky] Failed to create sky cubemap" << std::endl;
        return;
    }

    // --- SH9 buffer: 9×float4 = 144 bytes ---
    if (!CreateStructuredBuffer(m_SkySH9Buffer, sizeof(float) * 4, 9,
                                D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_SkySH9"))
    {
        std::cerr << "[Sky] Failed to create SH9 buffer" << std::endl;
        return;
    }

    // --- Staging buffer for cubemap upload ---
    // 128×128×6 RGBA16F = 128*128*6*8 bytes = 786,432 bytes
    constexpr UINT64 kStagingSize = kCubemapSize * kCubemapSize * kCubemapFaces * 8ULL;
    if (!CreateBuffer(m_StagingBuffer, (kStagingSize + 255ULL) & ~255ULL,
                      D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                      false, false, "Buf_SkyStaging"))
    {
        std::cerr << "[Sky] Failed to create sky staging buffer" << std::endl;
        return;
    }

    m_Dirty = true;
    m_ResourcesCreated = true;
    std::cout << "[Sky] Resources created (cubemap " << kCubemapSize
              << "x" << kCubemapSize << "x" << kCubemapFaces << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// CreatePipelines
// ---------------------------------------------------------------------------

void Sky::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
    computeDesc.pRootSignature = rootSignature;

    auto sh9CS = GraphicsHelper::CompileShader("Shaders/Sky_ProjectSH9.hlsl", "main", "cs_6_6");
    if (!sh9CS.empty())
    {
        computeDesc.CS = { sh9CS.data(), sh9CS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_ProjectSH9PSO));
        std::cout << "[Sky] SH9 projection PSO created" << std::endl;
    }
    else
    {
        std::cerr << "[Sky] Failed to compile Sky_ProjectSH9.hlsl" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Transition helpers
// ---------------------------------------------------------------------------

void Sky::TransitionCubemapToSRV(ID3D12GraphicsCommandList* cmdList)
{
    GraphicsHelper::TransitionResource(cmdList, m_SkyCubemap,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

void Sky::TransitionSH9ToUAV(ID3D12GraphicsCommandList* cmdList)
{
    GraphicsHelper::TransitionResource(cmdList, m_SkySH9Buffer,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void Sky::TransitionSH9ToSRV(ID3D12GraphicsCommandList* cmdList)
{
    GraphicsHelper::TransitionResource(cmdList, m_SkySH9Buffer,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

// ---------------------------------------------------------------------------
// BakeCubemap — CPU-side Hosek-Wilkie evaluation
// ---------------------------------------------------------------------------

bool Sky::BakeCubemap(ID3D12GraphicsCommandList* cmdList,
                      const LightConstants& sunLight, float turbidity, float groundAlbedo)
{
    if (!m_ResourcesCreated) return false;

    double elevation = SunElevationFromDirection(sunLight);

    // Compute TO-SUN direction for gamma angle computation.
    // sunLight.direction stores the light travel direction (downward in game
    // convention). Negate to get the celestial-sphere sun direction.
    XMFLOAT3 toSunDir;
    {
        XMVECTOR vSun = XMLoadFloat4(&sunLight.direction);
        vSun = XMVector3Normalize(XMVectorNegate(vSun));
        XMStoreFloat3(&toSunDir, vSun);
    }

    // Allocate HK model states for R, G, B channels.
    ArHosekSkyModelState* stateR = arhosek_rgb_skymodelstate_alloc_init(
        (double)turbidity, (double)groundAlbedo, elevation);
    ArHosekSkyModelState* stateG = arhosek_rgb_skymodelstate_alloc_init(
        (double)turbidity, (double)groundAlbedo, elevation);
    ArHosekSkyModelState* stateB = arhosek_rgb_skymodelstate_alloc_init(
        (double)turbidity, (double)groundAlbedo, elevation);

    if (!stateR || !stateG || !stateB)
    {
        std::cerr << "[Sky] Failed to allocate HK model states" << std::endl;
        if (stateR) arhosekskymodelstate_free(stateR);
        if (stateG) arhosekskymodelstate_free(stateG);
        if (stateB) arhosekskymodelstate_free(stateB);
        return false;
    }

    // CreateBuffer already mapped the UPLOAD heap; cpuPtr is always valid.
    // See GraphicsTypes.cpp line 31-34.
    if (!m_StagingBuffer.cpuPtr)
    {
        std::cerr << "[Sky] Staging buffer cpuPtr is null" << std::endl;
        arhosekskymodelstate_free(stateR);
        arhosekskymodelstate_free(stateG);
        arhosekskymodelstate_free(stateB);
        return false;
    }

    // D3D12 requires row pitch to be aligned to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256 bytes).
    constexpr UINT kBytesPerTexel   = 8; // RGBA16F = 8 bytes/texel
    constexpr UINT kFaceRowBytes    = kCubemapSize * kBytesPerTexel;
    constexpr UINT kAlignedRowPitch = ((kFaceRowBytes + 255) / 256) * 256;
    constexpr UINT kFaceSize        = kAlignedRowPitch * kCubemapSize;

    // Build the cubemap data in a local buffer, then memcpy to the persistent mapped ptr.
    constexpr UINT kTotalSize = kFaceSize * kCubemapFaces;
    std::vector<uint8_t> texelData(kTotalSize);

    for (uint32_t face = 0; face < kCubemapFaces; ++face)
    {
        for (uint32_t y = 0; y < kCubemapSize; ++y)
        {
            for (uint32_t x = 0; x < kCubemapSize; ++x)
            {
                // Texel center UV in [0,1]^2.
                float u = ((float)x + 0.5f) / (float)kCubemapSize;
                float v = ((float)y + 0.5f) / (float)kCubemapSize;

                XMFLOAT3 dir = CubemapTexelToDirection(face, u, v);

                double theta, gamma;
                DirectionToThetaGamma(dir, toSunDir, theta, gamma);

                // Evaluate per-channel radiance.
                // channel: 0=R, 1=G, 2=B (matches arhosek_tristim_skymodel_radiance).
                double radR = arhosek_tristim_skymodel_radiance(stateR, theta, gamma, 0);
                double radG = arhosek_tristim_skymodel_radiance(stateG, theta, gamma, 1);
                double radB = arhosek_tristim_skymodel_radiance(stateB, theta, gamma, 2);

                // Clamp to reasonable range, avoid negative values.
                float fr = (float)std::max(0.0, radR);
                float fg = (float)std::max(0.0, radG);
                float fb = (float)std::max(0.0, radB);

                // Pre-scale for fp16 HDR range — see SharedTypes.h FP16Scale.
                fr *= (kSkyRadianceScale * FP16Scale);
                fg *= (kSkyRadianceScale * FP16Scale);
                fb *= (kSkyRadianceScale * FP16Scale);

                // Pack RGBA16F — write 4 uint16_t half-float channels manually.
                uint8_t* dst = texelData.data()
                             + face * kFaceSize
                             + y * kAlignedRowPitch
                             + x * kBytesPerTexel;
                uint16_t* halfDst = reinterpret_cast<uint16_t*>(dst);
                halfDst[0] = DirectX::PackedVector::XMConvertFloatToHalf(fr);
                halfDst[1] = DirectX::PackedVector::XMConvertFloatToHalf(fg);
                halfDst[2] = DirectX::PackedVector::XMConvertFloatToHalf(fb);
                halfDst[3] = DirectX::PackedVector::XMConvertFloatToHalf(1.0f);
            }
        }
    }

    memcpy(m_StagingBuffer.cpuPtr, texelData.data(), kTotalSize);

    arhosekskymodelstate_free(stateR);
    arhosekskymodelstate_free(stateG);
    arhosekskymodelstate_free(stateB);

    // --- Upload staging → cubemap via CopyTextureRegion ---
    // Transition cubemap to COPY_DEST.
    GraphicsHelper::TransitionResource(cmdList, m_SkyCubemap, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource        = m_SkyCubemap.resource.Get();
    dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource        = m_StagingBuffer.resource.Get();
    srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint.Offset = 0;
    srcLoc.PlacedFootprint.Footprint.Format   = kCubemapFormat;
    srcLoc.PlacedFootprint.Footprint.Width    = kCubemapSize;
    srcLoc.PlacedFootprint.Footprint.Height   = kCubemapSize;
    srcLoc.PlacedFootprint.Footprint.Depth    = 1;
    srcLoc.PlacedFootprint.Footprint.RowPitch = kAlignedRowPitch;

    for (uint32_t face = 0; face < kCubemapFaces; ++face)
    {
        dstLoc.SubresourceIndex = face;
        srcLoc.PlacedFootprint.Offset = face * kFaceSize;

        cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    }

    std::cout << "[Sky] Cubemap baked and uploaded (turbidity=" << turbidity
              << ", albedo=" << groundAlbedo << ")" << std::endl;

    return true;
}

// ---------------------------------------------------------------------------
// DispatchSH9Projection
// ---------------------------------------------------------------------------

void Sky::DispatchSH9Projection(ID3D12GraphicsCommandList* cmdList,
                                 ID3D12RootSignature* rootSignature,
                                 D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress)
{
    if (!m_ProjectSH9PSO) return;

    // Compute bindings must be set independently from the graphics pipeline.
    // Layout matches the common root signature: b0=FrameCB, t3=BindlessSRV.
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    cmdList->SetPipelineState(m_ProjectSH9PSO.Get());

    // 1 thread group (64 threads), performs group-shared-memory reduction
    // over all cubemap texels and writes 9 float4s to the SkySH9 buffer.
    cmdList->Dispatch(1, 1, 1);

    std::cout << "[Sky] SH9 projection dispatched" << std::endl;
}

// ---------------------------------------------------------------------------
// Execute — called once per frame before G-Buffer pass
// ---------------------------------------------------------------------------

void Sky::Execute(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                  D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                  const LightConstants& sunLight,
                  float turbidity, float groundAlbedo)
{
    if (!m_ResourcesCreated) return;

    double elevation = SunElevationFromDirection(sunLight);

    // Check dirty flag.
    bool needsRebake = m_Dirty
        || std::abs(m_LastTurbidity    - turbidity)    > 0.01
        || std::abs(m_LastGroundAlbedo - groundAlbedo) > 0.01
        || std::abs(m_LastSunElevation - elevation)    > 1e-5;

    // todo: testing
    if (!needsRebake) return;

    // --- Re-bake ---
    if (!BakeCubemap(cmdList, sunLight, turbidity, groundAlbedo))
    {
        std::cerr << "[Sky] Bake failed, keeping previous cubemap" << std::endl;
        return;
    }

    m_LastTurbidity    = turbidity;
    m_LastGroundAlbedo = groundAlbedo;
    m_LastSunElevation = (float)elevation;
    m_Dirty            = false;

    // --- Compute sun irradiance from spectral solar radiance ---
    ComputeSunIrradiance(sunLight, turbidity, groundAlbedo);

    // --- SH9 Projection ---
    TransitionCubemapToSRV(cmdList);
    TransitionSH9ToUAV(cmdList);

    DispatchSH9Projection(cmdList, rootSignature, frameCBAddress);

    TransitionSH9ToSRV(cmdList);
}
