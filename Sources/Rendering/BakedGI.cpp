#include "pch.h"

#include "BakedGI.h"
#include "Core/Model.h"
#include "Graphics/GraphicsHelper.h"

#include <DirectXCollision.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>
#include <cstddef>
#include <cassert>

// =============================================================================
// Root-constant offset guards (C++ side only).
//
// SharedTypes.h is compiled by BOTH C++ and HLSL, so the guard cannot live
// there: offsetof and sizeof do not exist in HLSL. It lives here instead, against
// values measured from `dxc -Fc` on the BakedGI shaders. A mismatch makes a shader
// read one member as another with no build error, which is the class of defect
// that costs a day to find.
//
// Measured 2026-10-29, after the dead-slot cleanup.
//   BakedGIUpdateParams  (BakedGI_Update.hlsl): 0, 12, 16, 32, 60, 64, Size 68
//     and, appended 2026-11-04 for the sky-visibility A/B: skyVisSHSRVIdx 68,
//     skyVisMode 72, Size 76.
//   BakedGIVisParams     (BakedGI_Visibility.hlsl): skyVisSHUAVIdx 40, Size 44
//   BakedGIBakeParams    (BakedGI_Bake.hlsl):   seedDir 68, sunState 72,
//                            batchIndex 76, Size 80
//   BakedGIRayDebugParams (BakedGI_RayDebug.hlsl): seedDir 56, sunState 60,
//                            source 64, Size 68
//   BakedGIDilationParams (BakedGI_Dilate.hlsl): writeMeta 36, Size 40
// Every field in these blocks is a 4-byte scalar, so these are plain cumulative
// sums; the guards exist to catch a field inserted in the middle.
// =============================================================================
static_assert(offsetof(BakedGIUpdateParams, sunIrradianceX)    == 0,  "update params: sun irradiance moved");
static_assert(offsetof(BakedGIUpdateParams, hour)              == 12, "update params: hour moved");
static_assert(offsetof(BakedGIUpdateParams, responseSRVIdx)    == 16, "update params: response SRV moved");
static_assert(offsetof(BakedGIUpdateParams, litSunUAVIdx)      == 32, "update params: lit sun UAV moved");
static_assert(offsetof(BakedGIUpdateParams, dimZ)              == 60, "update params: dimZ moved");
static_assert(offsetof(BakedGIUpdateParams, sunResponseSRVIdx) == 64, "update params: sun response SRV moved");
static_assert(offsetof(BakedGIUpdateParams, skyVisSHSRVIdx)    == 68, "update params: sky-vis SH SRV moved");
static_assert(offsetof(BakedGIUpdateParams, skyVisMode)        == 72, "update params: sky-vis mode moved");
static_assert(sizeof(BakedGIUpdateParams)                      == 76, "update params: size changed");

// BakedGIVisParams: offsets measured with `dxc -Fc` too. skyVisSHUAVIdx 40,
// sizeof 44. The shared root slot 12 is sized by the 30-dword DDGIParams, so
// 11 dwords leave room for the next field as well.
static_assert(offsetof(BakedGIVisParams, metaUAVIdx)           == 36, "vis params: meta UAV moved");
static_assert(offsetof(BakedGIVisParams, skyVisSHUAVIdx)       == 40, "vis params: sky-vis SH UAV moved");
static_assert(sizeof(BakedGIVisParams)                         == 44, "vis params: size changed");

static_assert(offsetof(BakedGIBakeParams, seedDir)             == 68, "bake params: seedDir moved");
static_assert(offsetof(BakedGIBakeParams, sunState)            == 72, "bake params: sunState moved");
static_assert(offsetof(BakedGIBakeParams, batchIndex)          == 76, "bake params: batchIndex moved");
static_assert(sizeof(BakedGIBakeParams)                        == 80, "bake params: size changed");

static_assert(offsetof(BakedGIRayDebugParams, seedDir)         == 56, "ray-debug params: seedDir moved");
static_assert(offsetof(BakedGIRayDebugParams, sunState)        == 60, "ray-debug params: sunState moved");
static_assert(offsetof(BakedGIRayDebugParams, source)          == 64, "ray-debug params: source moved");
static_assert(sizeof(BakedGIRayDebugParams)                    == 68, "ray-debug params: size changed");

static_assert(offsetof(BakedGIDilationParams, recordCount)     == 16, "dilation params: recordCount moved");
static_assert(offsetof(BakedGIDilationParams, tableSRVIdx)     == 20, "dilation params: table SRV moved");
static_assert(offsetof(BakedGIDilationParams, metaSRVIdx)      == 28, "dilation params: meta SRV moved");
static_assert(offsetof(BakedGIDilationParams, writeMeta)       == 36, "dilation params: writeMeta moved");
static_assert(sizeof(BakedGIDilationParams)                    == 40, "dilation params: size changed");

// -----------------------------------------------------------------------------
// Sky-visibility SH9 normalization (the A/B's other V source).
//
// The visibility bake projects a binary visibility into raw SH9 with equal
// weight for each of its 16x16x4 = 1024 rays and the factor 4*pi/N
// (SKYVIS_SH_NORMALIZE, BakedGI_Visibility.hlsl):
//
//     L_lm = (4pi / N) * sum_s V_s * Y_lm(d_s)
//
// The reconstruction is then sum_lm L_lm Y_lm(d) with no cosine factor. For a
// FULLY OPEN probe (V = 1 on every ray) only the DC term survives, and it must
// reconstruct to exactly 1 at every direction:
//
//     (4pi/N) * N * Y00 * Y00 = 4pi * Y00^2 = 1
//
// Y00 = 0.28209479177387814 comes from EvalSH9Basis (Shaders/PBR.hlsl). This
// check holds that identity, so a change to the normalization constant or to
// the basis scale fails the build instead of biasing every probe's V.
// -----------------------------------------------------------------------------
namespace
{
constexpr double kSH9Y00       = 0.28209479177387814; // EvalSH9Basis, band 0
constexpr double kSH9OpenValue = 4.0 * 3.14159265358979323846 * kSH9Y00 * kSH9Y00;
}
static_assert(kSH9OpenValue > 0.999999999 && kSH9OpenValue < 1.000000001,
              "sky-vis SH9: the 4pi normalization does not reconstruct a fully open probe to 1");

static_assert(offsetof(FrameConstants, bakedGIUseFilled)       == 1168, "runtime FILLED flag moved");
// The two ray-debug source fields are appended after the last 4-byte scalar,
// so no pad is needed on either side. Measured with dxc as well: 1172, 1176,
// and the struct size 1180. Re-measure both sides when a field before these
// changes size or count.
static_assert(offsetof(FrameConstants, bakedGIRayDebugSource)  == 1172, "ray-debug source field moved");
static_assert(offsetof(FrameConstants, bakedGIRayDebugSkyDir)  == 1176, "ray-debug sky bin field moved");
// The photon-map fields are appended after the sky-bin field, all 4-byte
// scalars, so no pad is needed on either side. Measured with dxc as well:
// 1180, 1184, 1188, 1192, 1196, and the struct size 1200.
static_assert(offsetof(FrameConstants, bakedGIPhotonSRVIndex)  == 1180, "photon SRV field moved");
static_assert(offsetof(FrameConstants, bakedGIPhotonCount)     == 1184, "photon count field moved");
static_assert(offsetof(FrameConstants, bakedGIPhotonSunState)  == 1188, "photon state field moved");
static_assert(offsetof(FrameConstants, bakedGIPhotonFluxScale) == 1192, "photon flux scale moved");
static_assert(offsetof(FrameConstants, bakedGIPhotonIdxSRVIndex) == 1196, "photon index list field moved");
static_assert(sizeof(FrameConstants)                           == 1200, "FrameConstants size changed");
// The SH9 sky-visibility SRV index. The two V views are its only readers, and the
// response table carries no visibility. Measured with dxc on BakedGI_Debug.hlsl:
// 732. Nothing downstream moved, so the pads further on still hold.
static_assert(offsetof(FrameConstants, bakedGISkyVisSRVIndex)  == 732, "sky-vis SRV index moved");

// -----------------------------------------------------------------------------
// Ray-debug RECORD layout. The stride comes from BAKED_GI_RAY_RECORD_F4
// (SharedTypes.h), which both shader files also read, so this block guards the
// CPU arithmetic that the shader compiler cannot see: the field float4 indices
// used by the readback, and the two buffer sizes the dispatch and the copy use.
// Every number below is also printed in the layout comment of SharedTypes.h.
// -----------------------------------------------------------------------------
static_assert(BakedGI::kRayRecordF4 == 8u,        "ray record: stride changed");
static_assert(BakedGI::kRayRecordBytes == 128u,   "ray record: a record is not 128 B");
// The first six float4 keep their original offsets and meanings. A change here
// moves every reader at once, so the guard is explicit about all six.
static_assert(BakedGI::kRayRecF4GatherDir  == 0u, "ray record: rec0 moved");
static_assert(BakedGI::kRayRecF4Albedo     == 1u, "ray record: rec1 moved");
static_assert(BakedGI::kRayRecF4L          == 2u, "ray record: rec2 moved");
static_assert(BakedGI::kRayRecF4Efb        == 3u, "ray record: rec3 moved");
static_assert(BakedGI::kRayRecF4HitNormal  == 4u, "ray record: rec4 moved");
static_assert(BakedGI::kRayRecF4HitPos     == 5u, "ray record: rec5 moved");
static_assert(BakedGI::kRayRecF4Shadow     == 6u, "ray record: rec6 moved");
static_assert(BakedGI::kRayRecF4ShadowInfo == 7u, "ray record: rec7 moved");
// 16 directions x 64 records x 8 float4 = 8,192, then 16 x 9 for the table.
static_assert(BakedGI::kRayDebugTableBase == 8192u, "ray debug: table base moved");
static_assert(BakedGI::kRayDebugBufF4s    == 8336u, "ray debug: buffer float4 count moved");
static_assert(BakedGI::kRayDebugBufBytes  == 133376u, "ray debug: buffer size moved");
static_assert(BakedGI::kRayDebugReadbackBytes == 135680u, "ray debug: readback size moved");
// The response record shrank from 10 float4 to 9 when the cone visibility left
// slot 9. The stride is read by the bake, the dilation, the term fold, the ray
// debug and the CPU readback, so a regression must fail somewhere that names it.
static_assert(BakedGI::kResponseFloat4s == 9u, "response record: stride changed");
static_assert(BakedGI::kRayDebugTableBytes == 2304u, "ray debug: stored table size moved");
// The three secondary status values are read by the shader writer, the shader
// viewer and this readback, so a renumber has to fail somewhere that names it.
static_assert(BAKED_GI_RAY_SHADOW_NONE    == 0u, "ray record: shadow status NONE moved");
static_assert(BAKED_GI_RAY_SHADOW_OPEN    == 1u, "ray record: shadow status OPEN moved");
static_assert(BAKED_GI_RAY_SHADOW_BLOCKED == 2u, "ray record: shadow status BLOCKED moved");
// The channel number is written in three files and read by the renderer's
// instance-count branch, so it is checked against the mirror it must agree with.
static_assert(BakedGI::kRayFanChannelDeltaDirs == 7u, "ray fan: DeltaDirections channel moved");
static_assert(BakedGI::kRayFanChannelVHitRay   == 8u, "ray fan: VHit Ray channel moved");
static_assert(BakedGI::kSkyVisModeCount        == 2u, "sky visibility: mode count changed");
static_assert((uint32_t)BakedGI::SkyVisMode::SH9          == 0u, "sky visibility: SH9 is not mode 0");
static_assert((uint32_t)BakedGI::SkyVisMode::NoVisibility == 1u, "sky visibility: NoVisibility is not mode 1");
// The photon deposit stride and capacity are read by the emit dispatch's
// buffer sizing, the shader's slot clamp and the debug view's record walk, so
// a change must fail somewhere that names it. The shader's mirrors are
// BAKED_GI_PHOTON_CAP / BAKED_GI_PHOTON_F4 (BakedGI.hlsli).
static_assert(BakedGI::kPhotonCap == BAKED_GI_PHOTON_CAP, "photon map: capacity mirror changed");
static_assert(BakedGI::kPhotonF4  == BAKED_GI_PHOTON_F4,  "photon map: record stride mirror changed");
static_assert(BakedGI::kPhotonNoSRV == 0xFFFFFFFFu, "photon map: sentinel changed");

// =============================================================================
// Probes/ disk cache — the ShaderCache pattern applied to the bake outputs.
// =============================================================================
namespace
{

constexpr uint32_t kProbeCacheMagic   = 0x42504744; // 'BGPD'
// Version 2 added the 17 sun states. Version 3 appends the actual dilation
// count and, for a positive count, the final dilated sky and sun tables.
// Version 4 drops the tenth float4 of both response tables: the record is 9
// float4 now, so a version-3 payload is the wrong length and its strides are
// wrong. Older files cannot restore the runtime data this build requires.
// Version 5 changes the estimator to eight rotated fibonacci batches. Reject
// older files so a cache load cannot silently bypass the new sampling.
constexpr uint32_t kProbeCacheVersion = 5;

#pragma pack(push, 4)
struct ProbeCacheHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t dimX, dimY, dimZ;
    uint32_t probeCount;
    uint32_t finalResponse; // ping-pong index holding the final table
    float    spacing;
    float    gridMinX, gridMinY, gridMinZ;

    // ---- Appended for version 2. The header is read as sizeof(header), so a
    // longer header is self-consistent: the payload starts after it either way.
    //
    // The 17 state directions follow from ONE number, acos of the authored sun's
    // vertical component, so a cache baked at another latitude holds a valid
    // table of the wrong directions. Nothing else in this header detects that.
    uint32_t sunStates; // BakedGI::kSunStates at write time
    float    sunUpCos;  // the authored sun's Y component the directions derive from

    // Appended for version 3. Positive counts append the two final dilated
    // tables after the raw sun table. The final meta section contains FILLED.
    uint32_t dilationIterations;
};
#pragma pack(pop)

// 14 fields of 4 bytes, no padding under pack(4).
static_assert(sizeof(ProbeCacheHeader) == 56, "cache header: fields are 4-byte, no padding expected");

// FNV-1a 64 over a byte range. Used ONLY for the sun-table round-trip check: two
// runs of the app print one hash each, and equal hashes prove that the file
// carries the table the bake wrote. It is a test hook, not a data check — the
// loader still validates the header.
//
// The constants are the FNV-1a 64 offset basis and prime. The codebase had no
// hash helper before this one.
uint64_t Fnv1a64(const uint8_t* data, size_t bytes)
{
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < bytes; ++i)
    {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ull;
    }
    return h;
}

// One line, printed on BOTH cache paths, so the round-trip is provable without
// RenderDoc: run once with an empty Probes/ folder and once with the file in
// place, then compare the two hashes.
void LogSunTableHash(const uint8_t* sunBytes, uint64_t probeCount, const char* source)
{
    const uint64_t bytes = probeCount * BakedGI::kSunStates * BakedGI::kResponseFloat4s * 16ull;
    std::cout << "[BakedGI] sun table " << source << ": " << BakedGI::kSunStates
              << " states, " << (BakedGI::kSunStates * BakedGI::kResponseFloat4s * 16ull)
              << " B for each probe, " << bytes << " B total, FNV1a64 "
              << std::hex << Fnv1a64(sunBytes, (size_t)bytes) << std::dec << std::endl;
}

std::filesystem::path ProbeCachePath(const std::string& scenePath, float spacing)
{
    const std::filesystem::path dir = GraphicsHelper::GetCacheDir(L"Probes");
    if (dir.empty())
        return {};
    char sp[32];
    snprintf(sp, sizeof(sp), "%.2f", static_cast<double>(spacing));
    return dir / (std::filesystem::path(scenePath).stem().string() + "_s" + sp + ".probebin");
}

} // namespace

void BakedGI::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    // --- Bake (compute, uses inline ray queries via CommonTracing) ---
    {
        std::cout << "[BakedGI] compiling bake shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_Bake.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] bake shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_BakePSO));
        }
    }

    // --- Step-3 visibility bake: octahedral depth moments (compute, ray queries) ---
    {
        std::cout << "[BakedGI] compiling visibility bake shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_Visibility.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] visibility bake shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_VisBakePSO));
        }
    }

    // --- Per-frame lit-probe update (compute) ---
    {
        std::cout << "[BakedGI] compiling update shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_Update.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] update shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_UpdatePSO));
        }
    }

    // --- Fullscreen backface-contamination leak map (post-composite, alpha-blended) ---
    {
        std::cout << "[BakedGI] compiling leak-map shaders..." << std::endl;
        auto vs = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "LeakVS", "vs_6_6");
        auto ps = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "LeakPS", "ps_6_6");
        std::cout << "[BakedGI] leak-map shaders: " << (vs.empty() || ps.empty() ? "FAILED" : "ok") << std::endl;
        if (!vs.empty() && !ps.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature         = rootSignature;
            desc.VS                      = { vs.data(), vs.size() };
            desc.PS                      = { ps.data(), ps.size() };
            desc.BlendState              = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            desc.SampleMask              = UINT_MAX;
            desc.RasterizerState         = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT); // FillMode=SOLID etc.
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;                  // fullscreen triangle
            desc.PrimitiveTopologyType   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets        = 1;
            desc.SampleDesc.Count        = 1;
            desc.RTVFormats[0]           = DXGI_FORMAT_R8G8B8A8_UNORM; // backbuffer (post-composite)
            desc.DepthStencilState.DepthEnable = FALSE;
            // Alpha blend: sky pixels write a=0 so the tonemapped scene shows
            // through; geometry pixels (a=1) replace.
            desc.BlendState.RenderTarget[0].BlendEnable    = TRUE;
            desc.BlendState.RenderTarget[0].SrcBlend       = D3D12_BLEND_SRC_ALPHA;
            desc.BlendState.RenderTarget[0].DestBlend      = D3D12_BLEND_INV_SRC_ALPHA;
            desc.BlendState.RenderTarget[0].BlendOp        = D3D12_BLEND_OP_ADD;
            desc.BlendState.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
            desc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
            desc.BlendState.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
            HRESULT hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_LeakDebugPSO));
            if (FAILED(hr))
                std::cerr << "[BakedGI] leak-map PSO creation failed (hr = 0x"
                          << std::hex << hr << std::dec << ")" << std::endl;
        }
    }

    // --- Probe placement debug cubes (post-composite LDR overlay) ---
    {
        std::cout << "[BakedGI] compiling debug shaders..." << std::endl;
        auto vs = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "VSMain", "vs_6_6");
        auto ps = GraphicsHelper::CompileShader("Shaders/BakedGI_Debug.hlsl", "PSMain", "ps_6_6");
        std::cout << "[BakedGI] debug shaders: " << (vs.empty() || ps.empty() ? "FAILED" : "ok") << std::endl;
        if (!vs.empty() && !ps.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature         = rootSignature;
            desc.VS                      = { vs.data(), vs.size() };
            desc.PS                      = { ps.data(), ps.size() };
            desc.BlendState              = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            desc.SampleMask              = UINT_MAX;
            desc.RasterizerState         = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            // Engine convention (GBuffer/DeferredLighting/Shadow/Transparency/
            // Meshlet all set this): geometry is CCW-from-outside.
            desc.RasterizerState.FrontCounterClockwise = TRUE;
            desc.PrimitiveTopologyType   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets        = 1;
            desc.SampleDesc.Count        = 1;
            desc.RTVFormats[0]           = DXGI_FORMAT_R8G8B8A8_UNORM; // backbuffer (post-composite)
            // No DSV: occlusion is a manual reverse-Z compare in PSMain against
            // the GBuffer depth SRV (internal-res, mapped in the shader).
            desc.DepthStencilState.DepthEnable = FALSE;
            device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_DebugPSO));
        }
    }

    // --- Ray-debug re-trace (compute, on demand — debug only) ---
    {
        std::cout << "[BakedGI] compiling ray-debug shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_RayDebug.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] ray-debug shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_RayDebugPSO));
        }
    }

    // --- Per-term response-ball fold (compute, debug view 8) ---
    // Rides the bake list, so it is compiled here with the bake: a shader error
    // then shows up in this log line rather than only in a missing view.
    {
        std::cout << "[BakedGI] compiling iteration-fold shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_IterFold.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] iteration-fold shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_IterFoldPSO));
        }
    }

    // --- Bake-time transport dilation (compute) ---
    {
        std::cout << "[BakedGI] compiling dilation shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_Dilate.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] dilation shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_DilatePSO));
        }
    }

    // --- First-bounce photon-map emit (compute, on demand, debug-first) ---
    {
        std::cout << "[BakedGI] compiling photon emit shader..." << std::endl;
        D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
        computeDesc.pRootSignature = rootSignature;
        auto cs = GraphicsHelper::CompileShader("Shaders/BakedGI_PhotonEmit.hlsl", "main", "cs_6_6");
        std::cout << "[BakedGI] photon emit shader: " << (cs.empty() ? "FAILED" : "ok") << std::endl;
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_PhotonEmitPSO));
        }
    }
}

bool BakedGI::RecordBake(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                         ID3D12RootSignature* rootSignature, Model* model,
                         D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                         D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                         float spacing, uint32_t dilationIterations)
{
    if (!device || !cmdList || !rootSignature || !model || !m_BakePSO)
        return false;

    // The caller synced the GPU — release the previous bake's cache staging
    // now (the upload buffer must only die between executions, never while a
    // recorded list still references it).
    m_CacheUpload.Reset();
    m_CacheReadback.Reset();
    m_LoadedFromCache = false;
    // The sun table belongs to the previous bake's sun and grid. Clear the flag
    // first, so an early return cannot leave a stale sun table readable.
    m_SunResponseValid = false;
    // Dilation resources belong to one positive-count bake. Release them even
    // on a zero-count request, but retain descriptor slots for the next bake.
    // The descriptor heap has no free operation. Assigning {} here would leak slots.
    auto releaseDilationBuffer = [](GPUBuffer& buffer)
    {
        buffer.resource.Reset();
        buffer.size = 0;
        buffer.state = D3D12_RESOURCE_STATE_COMMON;
        buffer.cpuPtr = nullptr;
        buffer.gpuAddress = 0;
    };
    m_DilatedTransportValid = false;
    m_DilationIterations = 0;
    m_FinalDilatedResponse = 0;
    m_FinalDilatedSunResponse = 0;
    releaseDilationBuffer(m_DilatedResponse[0]);
    releaseDilationBuffer(m_DilatedResponse[1]);
    releaseDilationBuffer(m_DilatedSunResponse[0]);
    releaseDilationBuffer(m_DilatedSunResponse[1]);
    releaseDilationBuffer(m_DilationMeta);
    // The per-term capture is produced by the dispatch path below. Cleared here
    // so the cache-load early return cannot leave a previous bake's balls
    // standing as if they described this grid.
    m_EfbBallsValid   = false;
    // The photon map keys on the PREVIOUS bake's grid. Void it here, before
    // any early return, so no view can read deposits against a grid that no
    // longer exists. Both flags, so the validity pair never disagrees. A new
    // map is one button press.
    m_PhotonValid     = false;
    m_PhotonIdxValid  = false;

    // --- Grid from the scene bounds (auto-grow spacing to fit the caps) ---
    const DirectX::BoundingBox bounds = model->GetSceneWorldBounds();
    DirectX::XMFLOAT3 mn, mx;
    {
        DirectX::XMFLOAT3 c = bounds.Center, e = bounds.Extents;
        DirectX::XMFLOAT3 sceneMin = { c.x - e.x, c.y - e.y, c.z - e.z };
        DirectX::XMFLOAT3 sceneMax = { c.x + e.x, c.y + e.y, c.z + e.z };
        std::cout << "[BakedGI] Scene bounds: min=(" << sceneMin.x << ", " << sceneMin.y << ", " << sceneMin.z
                  << ") max=(" << sceneMax.x << ", " << sceneMax.y << ", " << sceneMax.z << ") "
                  << "size=(" << sceneMax.x - sceneMin.x << ", " << sceneMax.y - sceneMin.y << ", "
                  << sceneMax.z - sceneMin.z << " m)" << std::endl;
        mn = { sceneMin.x - spacing, sceneMin.y - spacing, sceneMin.z - spacing };
        mx = { sceneMax.x + spacing, sceneMax.y + spacing, sceneMax.z + spacing };
    }

    m_Spacing = spacing;
    for (uint32_t guard = 0; guard < 32; ++guard)
    {
        m_DimX = std::max(2u, (uint32_t)std::ceil((mx.x - mn.x) / m_Spacing) + 1u);
        m_DimY = std::max(2u, (uint32_t)std::ceil((mx.y - mn.y) / m_Spacing) + 1u);
        m_DimZ = std::max(2u, (uint32_t)std::ceil((mx.z - mn.z) / m_Spacing) + 1u);
        uint64_t count = (uint64_t)m_DimX * m_DimY * m_DimZ;
        if (m_DimX <= kMaxDim && m_DimY <= kMaxDim && m_DimZ <= kMaxDim && count <= kMaxProbeCount)
            break;
        m_Spacing *= 1.25f;
    }
    m_ProbeCount = m_DimX * m_DimY * m_DimZ;
    m_GridMin = mn;
    m_Valid = false; // set at the end; the buffers below are the source of truth

    // Beyond the Manhattan diameter no axial iteration can reach a new cell.
    const uint32_t dilationDiameter = (m_DimX - 1u) + (m_DimY - 1u) + (m_DimZ - 1u);
    m_DilationIterations = std::min(dilationIterations, dilationDiameter);
    if (dilationIterations != 0u && !m_DilatePSO)
    {
        std::cerr << "[BakedGI] Dilation requested but the dilation PSO is missing" << std::endl;
        return false;
    }

    // The grid identity is (re)established. Bumped here rather than on the
    // success path so the cache-load early return below is covered too: both
    // paths end up with this grid. A bump on a bake that later fails only costs
    // the UI one rebuild.
    ++m_GridRevision;

    if (m_Spacing != spacing)
        std::cout << "[BakedGI] Spacing auto-grew " << spacing << "m -> " << m_Spacing
                  << "m to fit the probe cap (" << kMaxProbeCount << " probes / "
                  << kMaxDim << " per axis)" << std::endl;

    std::cout << "[BakedGI] Baking probe grid: " << m_DimX << "x" << m_DimY << "x" << m_DimZ
              << " : " << m_Spacing << "m (" << m_ProbeCount << " probes, "
              << kBakeIterations << " series iterations, "
              << kBakeGatherBatches << " batches x " << kBakeGatherRays
              << " rotated fibonacci rays (" << (kBakeGatherBatches * kBakeGatherRays)
              << " total) for each of " << kDirections << " directions"
              << " + " << kSunStates << " sun hour-state passes, "
              << m_DilationIterations << " dilation iterations)"
              << " — world " << (mx.x - mn.x) << "x" << (mx.y - mn.y) << "x" << (mx.z - mn.z) << "m" << std::endl;

    // --- Probes/ disk cache: on a hit (file exists + grid header matches),
    // restore the outputs with plain buffer copies and skip the dispatches
    // entirely. Gated off right now (BakedGI::kProbeCacheEnabled) so the bake
    // always runs; the load path itself is unchanged. ---
    if constexpr (kProbeCacheEnabled)
    {
        if (TryLoadCache(device, cmdList, model->GetSourcePath()))
            return true;
    }

    if (!CreateProbeBuffers())
        return false;
    // The sky-visibility SH9 payload, reported once for each bake. 36 B for each
    // probe: 418,176 B at 11,616 probes, 4.72 MB at the kMaxProbeCount cap.
    std::cout << "[BakedGI] sky-visibility SH9 buffer: " << GetSkyVisSHBytes() << " B ("
              << kSkyVisSHCoeffs * sizeof(float) << " B for each probe, "
              << m_ProbeCount << " probes)" << std::endl;
    if (m_DilationIterations != 0u && !CreateDilationBuffers())
        return false;

    // Debug-only capture buffers (term snapshots + folded balls). A failure here
    // does not void the bake: the capture is a debug aid, so record without it
    // and let the "Efb Contribution" view report that it has nothing to read.
    const bool iterCapture = CreateIterationBuffers();
    if (!iterCapture)
        std::cerr << "[BakedGI] iteration capture unavailable — the Efb Contribution view will be empty" << std::endl;

    // --- Record the series-expansion iterations (ping-pong) ---
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, materialsAddress);
    cmdList->SetComputeRootShaderResourceView(2, drawNodesAddress);
    cmdList->SetComputeRootShaderResourceView(3, tlasAddress);
    cmdList->SetComputeRootShaderResourceView(4, indicesAddress);
    cmdList->SetComputeRootShaderResourceView(5, verticesAddress);

    // --- Step-3 visibility bake FIRST: the transport iterations' feedback
    // reads are visibility-weighted (BakedGI_Bake.hlsl EvalResponseIndirect
    // gates each feedback cage probe with backface + Chebyshev weights read
    // from these moment maps), so the moments must exist before iteration 1.
    // Rides the same list and root bindings (CBV b0 + material/draw-node/
    // TLAS/index/vertex SRVs) as the transport dispatches; only the b2
    // root-constants block and PSO change. One 16x16-thread group per probe;
    // probes tile across (x = kVisDispatchX, y) to respect the 65535-per-axis
    // dispatch cap (kMaxProbeCount is 131072).
    if (m_VisBakePSO)
    {
        BakedGIVisParams vp = {};
        vp.gridMinX = m_GridMin.x; vp.gridMinY = m_GridMin.y; vp.gridMinZ = m_GridMin.z;
        vp.spacing  = m_Spacing;
        vp.dimX = m_DimX; vp.dimY = m_DimY; vp.dimZ = m_DimZ;
        vp.probeCount = m_ProbeCount;
        vp.visUAVIdx  = (uint32_t)m_Visibility.uavIndex;
        vp.metaUAVIdx = (uint32_t)m_Meta.uavIndex;
        vp.skyVisSHUAVIdx = (uint32_t)m_SkyVisSH.uavIndex;

        cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIVisParams) / 4, &vp, 0);
        cmdList->SetPipelineState(m_VisBakePSO.Get());
        cmdList->Dispatch(kVisDispatchX, (m_ProbeCount + kVisDispatchX - 1) / kVisDispatchX, 1);

        // Barrier ALL THREE outputs: the transport iterations read the meta bits
        // the classification lane wrote (the visibility barrier alone would
        // leave that write-read unordered), and the SH buffer is read by the
        // per-frame fold after this list completes.
        D3D12_RESOURCE_BARRIER visBarriers[3] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_Visibility.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_Meta.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_SkyVisSH.resource.Get()),
        };
        cmdList->ResourceBarrier(3, visBarriers);
    }
    else
    {
        std::cerr << "[BakedGI] Visibility bake PSO missing — Chebyshev weights skipped" << std::endl;
        // No dispatch wrote the SH buffer, so its bytes are undefined. Mark the
        // SH data unavailable: the fold then zeroes V and the UI says so,
        // instead of reading garbage coefficients.
        m_SkyVisSHValid = false;
    }

    uint32_t cur = 0;
    for (uint32_t it = 0; it < kBakeIterations; ++it)
    {
        BakedGIBakeParams p = {};
        p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
        p.spacing  = m_Spacing;
        p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
        p.probeCount = m_ProbeCount;
        p.responseUAVIdx    = (uint32_t)m_Response[cur].uavIndex;
        p.responsePrevSRVIdx = (uint32_t)m_Response[1 - cur].srvIndex;
        p.metaUAVIdx         = (uint32_t)m_Meta.uavIndex;
        p.iteration          = it;
        p.visSRVIdx          = (uint32_t)m_Visibility.srvIndex;
        p.isSunPass          = 0u; // 16-direction sky transport
        // The binned table must not move, ever. It is the Probe method's data and
        // the reference the sun table is compared against, so this pass keeps the
        // shipped seed even when the caller pairs the sun with a bin.
        p.seedDir            = kSunSeedOwnDir;

        // Each batch overwrites or updates the running mean in the same output
        // record. The other ping-pong buffer remains the feedback source.
        for (uint32_t batch = 0; batch < kBakeGatherBatches; ++batch)
        {
            p.batchIndex = batch;
            cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIBakeParams) / 4, &p, 0);
            cmdList->SetPipelineState(m_BakePSO.Get());
            cmdList->Dispatch((m_ProbeCount + 7) / 8, kDirections / 8, 1);

            D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Response[cur].resource.Get());
            cmdList->ResourceBarrier(1, &barrier);

            // ---- Sun: one response for EACH hour state, toward that state's direction.
            // isSunPass redirects the delta direction, table and record stride.
            // The feedback operator and the batch sequence stay unchanged.
            //
            // seedDir = st, NOT the own-direction default: each state gets an
            // independent batch sequence. The hour fold then averages independent
            // errors instead of inheriting one correlated draw.
            if (m_SunResponse[0].resource)
            {
                for (uint32_t st = 0; st < kSunStates; ++st)
                {
                    BakedGIBakeParams sp = p;
                    sp.isSunPass          = 1u;
                    sp.sunState           = st;
                    const DirectX::XMFLOAT3 d = SunDirectionAtHourCPU(st);
                    sp.sunDirX            = d.x;
                    sp.sunDirY            = d.y;
                    sp.sunDirZ            = d.z;
                    sp.responseUAVIdx     = (uint32_t)m_SunResponse[cur].uavIndex;
                    sp.responsePrevSRVIdx = (uint32_t)m_SunResponse[1 - cur].srvIndex;
                    sp.seedDir            = st;

                    cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIBakeParams) / 4, &sp, 0);
                    cmdList->SetPipelineState(m_BakePSO.Get());
                    cmdList->Dispatch((m_ProbeCount + 7) / 8, 1, 1);
                }

                // Order this batch before the next running-mean read or feedback read.
                D3D12_RESOURCE_BARRIER sunBarrier =
                    CD3DX12_RESOURCE_BARRIER::UAV(m_SunResponse[cur].resource.Get());
                cmdList->ResourceBarrier(1, &sunBarrier);
            }
        }

        // Snapshot this term's table for the debug view. The LAST term needs no
        // copy: it is m_Response[m_FinalResponse] once the loop ends, so
        // snapshot[kBakeIterations - 1] is deliberately absent.
        if (iterCapture && it + 1u < kBakeIterations)
            SnapshotResponseTable(cmdList, cur, it);

        m_FinalResponse = cur;
        cur = 1 - cur;
    }

    m_Valid = true;

    // The sun table ping-pongs on the same schedule as the binned one, so the
    // final sun record sits in the same index the final response does.
    m_SunResponseValid = (m_SunResponse[0].resource != nullptr);
    m_FinalSunResponse = m_FinalResponse;

    // --- Per-term response balls (debug view 8: the Efb contribution) ---
    // One fold for each term. The view differences ball[k] and ball[0]; by the
    // bake's own identity R^k = R^0 + G(Efb_k) that difference IS the series-
    // feedback contribution of term k. Uses the bindings already on this list
    // (descriptor heap, root signature, b0, b2) — only the PSO and the b2 block
    // change, the same way the visibility bake rides the transport dispatches.
    if (iterCapture && m_IterFoldPSO)
    {
        // The snapshots were written by CopyBufferRegion, so they are COPY_DEST;
        // the fold reads them as an SRV. The final table needs no transition here
        // because the transport iterations already read their tables the same
        // way (a UAV barrier, no state change).
        for (uint32_t s = 0; s < kIterSnapshotCount; ++s)
            m_ResponseIter[s].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        for (uint32_t t = 0; t < kBakeIterations; ++t)
        {
            BakedGIIterFoldParams fp = {};
            fp.probeCount  = m_ProbeCount;
            fp.tableSRVIdx = (t + 1u < kBakeIterations)
                           ? (uint32_t)m_ResponseIter[t].srvIndex
                           : (uint32_t)m_Response[m_FinalResponse].srvIndex;
            fp.ballUAVIdx  = (uint32_t)m_EfbBall[t].uavIndex;

            cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIIterFoldParams) / 4, &fp, 0);
            cmdList->SetPipelineState(m_IterFoldPSO.Get());
            cmdList->Dispatch((m_ProbeCount + 63u) / 64u, 1, 1);
        }

        // One barrier for all three outputs: the debug view transitions them for
        // read anyway, but the bake must not end with unordered writes.
        D3D12_RESOURCE_BARRIER ballBarriers[kBakeIterations];
        for (uint32_t t = 0; t < kBakeIterations; ++t)
            ballBarriers[t] = CD3DX12_RESOURCE_BARRIER::UAV(m_EfbBall[t].resource.Get());
        cmdList->ResourceBarrier(kBakeIterations, ballBarriers);

        m_EfbBallsValid = true;
    }

    // Dilation runs only after all raw transport iterations and raw debug folds.
    // It writes separate runtime tables and never modifies a raw response pair.
    if (m_DilationIterations != 0u && !RecordDilation(cmdList))
    {
        m_Valid = false;
        return false;
    }

    // ---- Sky-visibility SH9: leave the write state ONCE, on the bake list ----
    //
    // The visibility dispatch wrote this buffer as a UAV and its barrier is
    // recorded above, so this transition is ordered after that write and after
    // every later UAV barrier of the bake. The per-frame fold reads the buffer
    // through an SRV on a different command list, so the read state must be
    // established here and once, not in RecordUpdate.
    //
    // A re-bake calls CreateProbeBuffers, which replaces the resource and sets
    // its tracked state to UNORDERED_ACCESS again, so no stale state survives a
    // bake. The buffer is transitioned only when a dispatch actually filled it:
    // without the visibility PSO the bytes are undefined and nothing reads them.
    if (m_SkyVisSHValid)
        m_SkyVisSH.Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    std::cout << "[BakedGI] Bake recorded (" << m_ProbeCount << " probes, "
              << "visibility: 16x16 depth moments x" << kVisStride << " texels/probe, "
              << (m_EfbBallsValid ? "per-term response balls captured"
                                  : "no per-term capture")
              << ")" << std::endl;
    return true;
}

bool BakedGI::RecordMetaReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList)
{
    if (!m_Valid || !device || !cmdList || !m_Meta.resource)
        return false;

    const UINT64 byteSize = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&m_MetaReadback))))
    {
        std::cerr << "[BakedGI] Failed to create the meta readback buffer" << std::endl;
        return false;
    }
    GraphicsHelper::SetObjectName(m_MetaReadback.Get(), "RB_BakedGIMeta");

    // Appended to the bake list: meta (UAV after the dispatches) -> readback.
    // Round-trips back to UAV so the runtime update pass — which reads the
    // meta SRV without transitions — is unaffected.
    m_Meta.Transition(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->CopyBufferRegion(m_MetaReadback.Get(), 0, m_Meta.resource.Get(), 0, byteSize);
    m_Meta.Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return true;
}

void BakedGI::LogBakeStats() const
{
    if (!m_Valid || !m_MetaReadback)
        return;

    uint32_t* meta = nullptr;
    if (FAILED(m_MetaReadback->Map(0, nullptr, reinterpret_cast<void**>(&meta))))
    {
        std::cerr << "[BakedGI] Failed to map the meta readback buffer" << std::endl;
        return;
    }

    uint32_t ddgiValid = 0, oldValid = 0, newlyBad = 0, newlyValid = 0;
    uint32_t deadBackface = 0, deadFar = 0, filled = 0, unfilled = 0;
    for (uint32_t i = 0; i < m_ProbeCount; ++i)
    {
        const uint32_t m = meta[i];
        const bool v  = (m & 0x1u) != 0; // BAKED_GI_META_VALID (DDGI criteria)
        const bool ov = (m & 0x2u) != 0; // BAKED_GI_META_OLD_VALID (1cm test)
        const bool f  = (m & 0x10u) != 0; // BAKED_GI_META_FILLED (donor-backed data)
        ddgiValid += v ? 1u : 0u;
        filled    += f ? 1u : 0u;
        unfilled  += (!v && !f) ? 1u : 0u;
        oldValid  += ov ? 1u : 0u;
        newlyBad   += (!v && ov) ? 1u : 0u;
        newlyValid += (v && !ov) ? 1u : 0u;
        deadBackface += (m & 0x4u) ? 1u : 0u;
        deadFar      += (m & 0x8u) ? 1u : 0u;
    }
    m_MetaReadback->Unmap(0, nullptr);

    const float pctNew = 100.0f * (float)ddgiValid / (float)m_ProbeCount;
    const float pctOld = 100.0f * (float)oldValid / (float)m_ProbeCount;
    std::cout << "[BakedGI] Probe coverage (DDGI criteria): " << ddgiValid << "/" << m_ProbeCount
              << " valid (" << pctNew << "%) — dead: " << deadBackface
              << " by >= 25% backfaces, " << deadFar << " by > 3x spacing far rule" << std::endl;
    std::cout << "[BakedGI] Old 1cm test for comparison:      " << oldValid << "/" << m_ProbeCount
              << " valid (" << pctOld << "%)" << std::endl;
    std::cout << "[BakedGI] Criteria swap flips: " << newlyBad
              << " valid -> BAD (blue in placement view), " << newlyValid
              << " bad -> valid (yellow)" << std::endl;
    std::cout << "[BakedGI] Dilation (" << m_DilationIterations << " iterations): "
              << filled << " filled, " << unfilled << " invalid and unfilled" << std::endl;
}

// -----------------------------------------------------------------------------
// Photon map — first-bounce sun deposits. Debug only, on demand. The transport
// bake never reads these buffers; the "Photon Map" view and LogPhotonStats do.
// -----------------------------------------------------------------------------

bool BakedGI::RecordPhotonEmit(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                               ID3D12RootSignature* rootSignature, Model* model,
                               D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                               D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                               D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                               D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                               D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                               D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                               uint32_t sunState, uint32_t raysPerAxis)
{
    if (!device || !cmdList || !rootSignature || !model || !m_PhotonEmitPSO)
    {
        std::cerr << "[BakedGI] Photon emit rejected: device, list, signature, model or PSO missing"
                  << std::endl;
        return false; // nothing was touched: the resident map, if any, stands
    }
    // The deposit grid keys on the probe grid, so a resident bake is a
    // prerequisite: without one there is no grid to hash into.
    if (!m_Valid)
    {
        std::cerr << "[BakedGI] Photon emit needs a valid probe bake first" << std::endl;
        return false; // nothing was touched: the resident map, if any, stands
    }
    if (sunState >= kSunStates)
    {
        std::cerr << "[BakedGI] Photon emit rejected: sun state " << sunState
                  << " out of range" << std::endl;
        return false; // nothing was touched: the resident map, if any, stands
    }
    const uint32_t rays = std::clamp(raysPerAxis, 16u, 4096u);
    const uint64_t rayTotal = (uint64_t)rays * rays;

    // --- VOID FIRST, PUBLISH LAST (defect 2) ---
    //
    // The buffers below are recreated on EVERY press. A failure after this
    // point leaves a recreated — therefore EMPTY — deposit buffer under the
    // previous generation's numbers, which is exactly the captured defect
    // (67,197 published instances over an all-zero buffer). Void the
    // published state before the first creation, and re-publish only on the
    // success path at the end. A failed press means NO map: the view declines
    // to draw, the panel reports none. The returns above this block are
    // different: they touch nothing, so the resident map stands.
    m_PhotonValid    = false;
    m_PhotonIdxValid = false;
    m_PhotonTotal    = 0;
    m_PhotonDropped  = 0;
    m_PhotonCells    = 0;

    // --- Buffers: deposits + append counters. The caller synced the GPU, and
    // CreateStructuredBuffer reuses the existing descriptor slots (the
    // ray-debug dump's property, so repeated emits leak nothing). ---
    const uint64_t photonElems = (uint64_t)m_ProbeCount * kPhotonCap * kPhotonF4;
    if (!CreateStructuredBuffer(m_Photons, sizeof(float) * 4, photonElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIPhotons") ||
        !CreateStructuredBuffer(m_PhotonCounts, sizeof(uint32_t), (uint64_t)m_ProbeCount + 1u,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIPhotonCounts"))
    {
        std::cerr << "[BakedGI] Failed to create the photon buffers" << std::endl;
        return false;
    }

    // The counter clear below needs a UAV descriptor in the CPU-ONLY heap.
    // CreateStructuredBuffer allocates only the shader-visible one: its
    // cpuUavIndex path lives in CreateBuffer's createUAV branch, which no
    // structured buffer takes, so m_PhotonCounts.cpuUavIndex is otherwise -1
    // and ClearUnorderedAccessViewUint would dereference a wild CPU handle
    // (access violation, found at runtime). Create the descriptor here,
    // mirroring the shader-visible view exactly.
    if (m_PhotonCounts.cpuUavIndex < 0)
        m_PhotonCounts.cpuUavIndex = (int)GraphicsHelper::AllocateCpuUAV();
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC cpuUavDesc = {};
        cpuUavDesc.Format = DXGI_FORMAT_UNKNOWN;
        cpuUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        cpuUavDesc.Buffer.FirstElement = 0;
        cpuUavDesc.Buffer.NumElements = m_ProbeCount + 1u;
        cpuUavDesc.Buffer.StructureByteStride = sizeof(uint32_t);
        device->CreateUnorderedAccessView(
            m_PhotonCounts.resource.Get(), nullptr, &cpuUavDesc,
            GraphicsHelper::GetCpuUAVHandle((UINT)m_PhotonCounts.cpuUavIndex));
    }

    // --- Emission quad: the scene's projection perpendicular to the sun,
    // placed outside the scene on the sun side. The CPU builds it, so the
    // shader only interpolates (the CPU-is-the-authority rule). The AABB
    // support function gives the projected half-extents without a corner
    // loop: the box spans ±(sum |e_i * axis_i|) along each quad axis. ---
    using namespace DirectX;
    const XMFLOAT3 sunDir = SunDirectionAtHourCPU(sunState);
    const XMFLOAT3 bc = model->GetSceneWorldBounds().Center;
    const XMFLOAT3 be = model->GetSceneWorldBounds().Extents;

    const XMVECTOR s = XMLoadFloat3(&sunDir);
    XMVECTOR right = XMVector3Cross(XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f), s);
    if (XMVectorGetX(XMVector3LengthSq(right)) < 1e-6f)
        right = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f); // vertical sun: any perpendicular axis
    right = XMVector3Normalize(right);
    const XMVECTOR up = XMVector3Normalize(XMVector3Cross(s, right));

    XMFLOAT3 r3, u3;
    XMStoreFloat3(&r3, right);
    XMStoreFloat3(&u3, up);
    const float halfR = std::abs(be.x * r3.x) + std::abs(be.y * r3.y) + std::abs(be.z * r3.z);
    const float halfU = std::abs(be.x * u3.x) + std::abs(be.y * u3.y) + std::abs(be.z * u3.z);
    const float radius = std::sqrt(be.x * be.x + be.y * be.y + be.z * be.z);
    const float dist = radius + 1.0f; // quad clearance on the sun side

    const float quadW = 2.0f * halfR;
    const float quadH = 2.0f * halfU;
    const float quadArea = quadW * quadH;
    XMFLOAT3 qo, qr, qu;
    XMStoreFloat3(&qo, XMLoadFloat3(&bc) + s * dist - right * halfR - up * halfU);
    XMStoreFloat3(&qr, right * quadW);
    XMStoreFloat3(&qu, up * quadH);

    BakedGIPhotonEmitParams p = {};
    p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
    p.spacing  = m_Spacing;
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
    p.probeCount = m_ProbeCount;
    p.sunDirX = sunDir.x; p.sunDirY = sunDir.y; p.sunDirZ = sunDir.z;
    p.sunState = sunState;
    p.quadOriginX = qo.x; p.quadOriginY = qo.y; p.quadOriginZ = qo.z;
    p.quadRightX = qr.x;  p.quadRightY = qr.y;  p.quadRightZ = qr.z;
    p.quadUpX = qu.x;     p.quadUpY = qu.y;     p.quadUpZ = qu.z;
    p.raysX = rays; p.raysY = rays;
    p.photonUAVIdx = (uint32_t)m_Photons.uavIndex;
    p.countUAVIdx = (uint32_t)m_PhotonCounts.uavIndex;
    p.photonCap = kPhotonCap;
    // From the quad, through the whole scene, with slack: dist + 2*radius is
    // the geometric worst case and this is comfortably above it.
    p.rayTMax = 2.0f * (radius + dist);

    // Root bindings: the bake list's own set (CBV b0 + material/draw-node/
    // TLAS/index/vertex SRVs), so the shader's CommonTracing includes resolve.
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, materialsAddress);
    cmdList->SetComputeRootShaderResourceView(2, drawNodesAddress);
    cmdList->SetComputeRootShaderResourceView(3, tlasAddress);
    cmdList->SetComputeRootShaderResourceView(4, indicesAddress);
    cmdList->SetComputeRootShaderResourceView(5, verticesAddress);

    // Counters to zero before the dispatch, including the out-of-grid slot
    // (the GPUCulling counter-clear pattern).
    const UINT zeroes[4] = { 0, 0, 0, 0 };
    cmdList->ClearUnorderedAccessViewUint(
        GraphicsHelper::GetSRVGPUHandle((UINT)m_PhotonCounts.uavIndex),
        GraphicsHelper::GetCpuUAVHandle((UINT)m_PhotonCounts.cpuUavIndex),
        m_PhotonCounts.resource.Get(), zeroes, 0, nullptr);

    cmdList->SetComputeRoot32BitConstants(12, sizeof(p) / 4, &p, 0);
    cmdList->SetPipelineState(m_PhotonEmitPSO.Get());
    cmdList->Dispatch((uint32_t)((rayTotal + 63) / 64), 1, 1);

    D3D12_RESOURCE_BARRIER photonBarriers[2] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_Photons.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_PhotonCounts.resource.Get()),
    };
    cmdList->ResourceBarrier(2, photonBarriers);

    // Count readback rides the same list (the meta-readback pattern), so
    // LogPhotonStats can map it right after the caller's wait.
    const UINT64 countBytes = ((uint64_t)m_ProbeCount + 1u) * sizeof(uint32_t);
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(countBytes);
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &rd,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_PhotonCountReadback))))
    {
        std::cerr << "[BakedGI] Failed to create the photon count readback" << std::endl;
        return false;
    }
    GraphicsHelper::SetObjectName(m_PhotonCountReadback.Get(), "RB_BakedGIPhotonCounts");
    m_PhotonCounts.Transition(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->CopyBufferRegion(m_PhotonCountReadback.Get(), 0, m_PhotonCounts.resource.Get(), 0, countBytes);
    m_PhotonCounts.Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // PUBLISH (the second half of void-first/publish-last). The getters'
    // numbers describe the RESIDENT map. Total stays 0 until LogPhotonStats
    // maps the readback, so the view draws nothing before that — correct,
    // because the counts are not known yet. The index list is voided for the
    // same reason: the new deposits have no list until the readout builds
    // one, and a stale list would index the new buffer with the old
    // generation's slots.
    m_PhotonValid = true;
    m_PhotonSunState = sunState;
    m_PhotonFluxScale = (float)rayTotal / quadArea;
    m_PhotonTotal = 0;
    m_PhotonDropped = 0;
    m_PhotonCells = 0;
    m_PhotonIdxValid = false;

    const float hour = kSunStateHour0 + kSunStateStepHours * (float)sunState;
    std::cout << "[BakedGI] Photon emit recorded: state " << sunState << " (hour "
              << hour << ", to-sun " << sunDir.x << ", " << sunDir.y << ", " << sunDir.z
              << "), " << rays << "x" << rays << " = " << rayTotal << " rays, quad "
              << quadW << "x" << quadH << " m, cap " << kPhotonCap << "/cell, deposits "
              << (photonElems * 16ull / (1024ull * 1024ull)) << " MB" << std::endl;
    return true;
}

void BakedGI::LogPhotonStats(ID3D12Device* device)
{
    m_PhotonTotal = 0;
    m_PhotonDropped = 0;
    m_PhotonCells = 0;
    m_PhotonIdxValid = false;
    if (!m_PhotonValid || !m_PhotonCountReadback)
    {
        // No valid map: the normal post-re-bake state, nothing to report. A
        // valid map with no readback cannot be counted, and a silent zero
        // would read as "the map is empty" — say so instead. Unreachable in
        // the current flow (the emit that published the map made the
        // readback), kept for the invariant, not for expectation.
        if (m_PhotonValid)
            std::cerr << "[BakedGI] Photon stats skipped: no count readback for the resident map"
                      << std::endl;
        return;
    }

    uint32_t* counts = nullptr;
    if (FAILED(m_PhotonCountReadback->Map(0, nullptr, reinterpret_cast<void**>(&counts))))
    {
        // The map EXISTS (m_PhotonValid stays true) but its count is unknown,
        // so the published totals stay 0 and the index list stays void: the
        // view stays dark rather than drawing a wrong generation. A re-press
        // of "Bake Photon Map" is the recovery.
        std::cerr << "[BakedGI] Failed to map the photon count readback — totals stay 0"
                     " and the view stays dark until a new bake" << std::endl;
        return;
    }

    uint32_t saturated = 0;
    for (uint32_t i = 0; i < m_ProbeCount; ++i)
    {
        const uint32_t c = counts[i];
        const uint32_t stored = std::min(c, kPhotonCap);
        m_PhotonTotal  += stored;
        m_PhotonDropped += c - stored;
        m_PhotonCells  += (c > 0u) ? 1u : 0u;
        saturated      += (c >= kPhotonCap) ? 1u : 0u;
    }
    const uint32_t outOfGrid = counts[m_ProbeCount];
    m_PhotonDropped += outOfGrid;

    // --- The view's index list: instance -> STORED deposit record ---
    //
    // The deposits are SPARSE (record slot = cell * cap + slot), so a dense
    // instance * kPhotonF4 walk reads only the first cells of the grid: a draw
    // of `total` instances covers cells 0 to total/cap - 1, and no cell past
    // them can draw (capture-proven: 67,197 instances, cells 0..1,049 of
    // 11,616). The list below enumerates every stored deposit in cell order —
    // one entry for each, so the list length IS m_PhotonTotal — and the view
    // reads record idx[instance]. Built HERE because the counts are mapped
    // here; the entries come from the same readout the stats above came from,
    // so the list can never disagree with the totals.
    //
    // UPLOAD heap with a persistent Map (the SB_RtxdiNeighborOffsets pattern,
    // PathTracing.cpp): written ONCE per bake on a synced GPU — the caller
    // executed and waited before this call — and read by every later frame's
    // view, so a copy queue or a default-heap staging hop buys nothing.
    // GENERIC_READ is already a valid shader-read state, so the list needs no
    // transition in TransitionPhotonForRead. Worst case at the probe cap:
    // 131072 * 64 * 4 B = 33.5 MB, debug-only. Sponza scale is about 270 KB.
    std::vector<uint32_t> indices;
    if (m_PhotonTotal > 0u)
    {
        indices.reserve(m_PhotonTotal);
        for (uint32_t i = 0; i < m_ProbeCount; ++i)
        {
            const uint32_t stored = std::min(counts[i], kPhotonCap);
            const uint32_t cellBase = i * kPhotonCap;
            for (uint32_t slot = 0; slot < stored; ++slot)
                indices.push_back(cellBase + slot);
        }
    }
    m_PhotonCountReadback->Unmap(0, nullptr);

    if (!indices.empty())
    {
        if (!device ||
            !CreateStructuredBuffer(m_PhotonIndices, sizeof(uint32_t), indices.size(),
                                    D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                                    "SB_BakedGIPhotonIndices"))
        {
            std::cerr << "[BakedGI] Failed to create the photon index list" << std::endl;
            return; // m_PhotonIdxValid stays false: the view declines to draw
        }
        // CreateStructuredBuffer keeps the upload heap persistently mapped
        // (CreateBuffer maps on HEAP_TYPE_UPLOAD), so one memcpy publishes the
        // list to every later frame.
        memcpy(m_PhotonIndices.cpuPtr, indices.data(), indices.size() * sizeof(uint32_t));
        m_PhotonIdxValid = true;
    }
    // A zero-deposit bake leaves no buffer at all: m_PhotonTotal is 0, so the
    // view draws no instance and never reads the (sentinel) index.

    const float hour = kSunStateHour0 + kSunStateStepHours * (float)m_PhotonSunState;
    const float cellPct = 100.0f * (float)m_PhotonCells / (float)m_ProbeCount;
    std::cout << "[BakedGI] Photon map (state " << m_PhotonSunState << ", hour " << hour
              << "): " << m_PhotonTotal << " deposits in " << m_PhotonCells << "/"
              << m_ProbeCount << " cells (" << cellPct << "%), " << saturated
              << " cells at cap, " << m_PhotonDropped << " dropped ("
              << outOfGrid << " outside the grid)" << std::endl;
}

void BakedGI::TransitionPhotonForRead(ID3D12GraphicsCommandList* cmdList)
{
    // The index list is NOT transitioned: it lives on an UPLOAD heap in
    // GENERIC_READ, which is already a shader-read state, so no barrier ever
    // applies to it. Only the default-heap deposit buffer needs the move.
    if (m_Photons.resource)
        m_Photons.Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

// -----------------------------------------------------------------------------
// Ray-debug tool — debug only, on demand
// -----------------------------------------------------------------------------

namespace
{

// The shader's luminance weights (CommonTracing.hlsl Luminance). Kept local so
// this debug path shares no symbol with the renderer.
inline float RayDebugLuminance(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// One float4 of the response table for a direction and an SH term.
// The stride is BakedGI::kResponseFloat4s (9). It is NOT a literal: the record
// lost its tenth float4 when the cone visibility left slot 9, and a stale 10
// here would read every direction one float4 past where it belongs.
inline const float* RayDebugTable(const uint8_t* base, uint32_t dir, uint32_t lm)
{
    return reinterpret_cast<const float*>(base)
         + ((size_t)dir * BakedGI::kResponseFloat4s + lm) * 4u;
}

// The float offset of one record field. A record is indexed in float4s
// (BakedGI::kRayRecF4*, which mirrors the shader layout) and this readback walks
// floats, so this is the ONE place that converts.
inline const float* RayDebugField(const float* rec, uint32_t f4)
{
    return rec + (size_t)f4 * 4u;
}

// The record `r` of record group `dir`, inside the mapped readback. ONE place
// multiplies by the stride, so a stride change reaches one line.
inline const float* RayDebugRecord(const uint8_t* mapped, uint32_t dir, uint32_t r)
{
    return reinterpret_cast<const float*>(mapped)
         + ((size_t)dir * BakedGI::kDebugGatherRays + r) * BakedGI::kRayRecordF4 * 4u;
}

// rec3.w packs TWO values into one float: status + 4 * sourceRayIndex.
// BakedGI_RayDebug.hlsl owns the encode. Status has three values (0 = sky miss,
// 1 = backface reject, 2 = shaded), so a reader decodes instead of comparing.
//
// Exactness: fp32 represents every integer up to 2^24, and the largest packed
// value is 2 + 4 * 255 = 1022, so std::fmod is exact and the round trip loses
// nothing. Read rec3.w directly and you get the packed value, not the status.
inline float RayDebugStatus(const float* rec)
{
    return std::fmod(RayDebugField(rec, BakedGI::kRayRecF4Efb)[3], 4.0f);
}

inline uint32_t RayDebugSourceRay(const float* rec)
{
    return static_cast<uint32_t>(
        std::lround((RayDebugField(rec, BakedGI::kRayRecF4Efb)[3] - RayDebugStatus(rec)) * 0.25f));
}

// The SECONDARY visibility ray of a record: the shadow ray the gather hit fired
// toward the active delta direction W. rec6.w is the status and it is a plain
// float, so no decode is needed; the three values come from SharedTypes.h, which
// the shader writer reads as well.
//
//   NONE     no shadow ray was traced. The record is not shaded, or NdotD <= 0
//   OPEN     nothing was hit, so Vhit is 1 and rec6.xyz is W
//   BLOCKED  the ray stopped, so Vhit is 0 and rec6.xyz is that hit's world pos
inline float RayDebugShadowStatus(const float* rec)
{
    return RayDebugField(rec, BakedGI::kRayRecF4Shadow)[3];
}

inline const float* RayDebugShadowPayload(const float* rec)
{
    return RayDebugField(rec, BakedGI::kRayRecF4Shadow); // .xyz = W (open) or hit pos (blocked)
}

// rec7 = (distance, instance, primitive, reserved). -1 in all three means the
// field carries no value, which is the rec4/rec5 convention: instance 0 and
// distance 0 are legal and must not also mean "none".
inline const float* RayDebugShadowInfo(const float* rec)
{
    return RayDebugField(rec, BakedGI::kRayRecF4ShadowInfo);
}

} // namespace

bool BakedGI::RecordRayDebug(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                             ID3D12RootSignature* rootSignature,
                             D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS tlasAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS materialsAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS drawNodesAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS indicesAddress,
                             D3D12_GPU_VIRTUAL_ADDRESS verticesAddress,
                             uint32_t probeIndex,
                             RayDebugSource source,
                             uint32_t skyDirection,
                             uint32_t sunState)
{
    if (!device || !cmdList || !rootSignature || !m_RayDebugPSO)
        return false;
    if (!m_Valid || m_ProbeCount == 0)
        return false;

    // The mirror check runs here, on the one path that prints a direction. It is
    // two sin/cos calls for each dump, and it turns a formula drift into a failed
    // dump instead of a direction in the log that the shader never traced.
    const bool mirrorOk = VerifyDeltaDirectionMirror();
    assert(mirrorOk);
    (void)mirrorOk;

    // Clamp into the grid: a stale UI value must not read outside the table.
    if (probeIndex >= m_ProbeCount)
        probeIndex = m_ProbeCount - 1u;
    m_RayDebugProbe = probeIndex;
    // Clamp the bin. A stale value must not select a bin that does not exist, and
    // it is the DISPLAYED bin after the dump, so it also clamps what the panel may
    // show next.
    if (skyDirection >= kDirections)
        skyDirection = kDirections - 1u;
    // Record what THIS dump replays, before any early return below, so the labels
    // name the source the buffer will hold. A dump never changes the caller's
    // selection: the source is an argument, so it can only be echoed here.
    //
    // For a SKY dump the bin is the DISPLAYED bin, not the set of bins traced: the
    // dispatch below records all kDirections of them, so the request only decides
    // which one is on screen when the dump finishes.
    m_RayDebugSource       = source;
    m_RayDebugSkyDirection = (source == RayDebugSource::Sky) ? skyDirection : kRayDebugNoDir;
    // Clear the verdict for this dump. A skipped assignment then yields no
    // verdict, whatever the index value is. The pairing check alone cannot catch
    // a re-bake that flips the verdict at the SAME index.
    m_RayDebugValidProbe = UINT32_MAX;

    // The Cage Trace pool describes the OLD records. Clear it here, not in the
    // readout: a dump whose readout never runs — no readback, or a failed Map —
    // must leave an EMPTY pool, so the UI cannot offer a ray index that belongs
    // to a buffer that is not resident.
    for (auto& list : m_RayCandidates)
        list.clear();
    m_RayCandidateDirs = 0u;

    // The caller synced the GPU before recording, so releasing the previous
    // readback here is safe (the meta-readback pattern). m_RayDebugBuf is NOT
    // cleared: CreateStructuredBuffer reuses the existing srvIndex/uavIndex, so
    // a repeated dump does not leak descriptor slots (the heap has no free).
    m_RayDebugReadback.Reset();

    // --- GPU output: records, then the re-accumulated table (one UAV). ---
    // CreateStructuredBuffer allocates the SRV and UAV role descriptors, so the
    // shader reads the UAV through ResourceDescriptorHeap[m_RayDebugBuf.uavIndex].
    if (!CreateStructuredBuffer(m_RayDebugBuf, 16u, kRayDebugBufF4s,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                "SB_BakedGIRayDebug"))
    {
        std::cerr << "[BakedGI] Failed to create the ray-debug buffer" << std::endl;
        return false;
    }

    // --- CPU readback: debug buffer | stored table. ---
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(kRayDebugReadbackBytes);
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&m_RayDebugReadback))))
        {
            std::cerr << "[BakedGI] Failed to create the ray-debug readback buffer" << std::endl;
            return false;
        }
        GraphicsHelper::SetObjectName(m_RayDebugReadback.Get(), "RB_BakedGIRayDebug");
    }

    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, materialsAddress);
    cmdList->SetComputeRootShaderResourceView(2, drawNodesAddress);
    cmdList->SetComputeRootShaderResourceView(3, tlasAddress);
    cmdList->SetComputeRootShaderResourceView(4, indicesAddress);
    cmdList->SetComputeRootShaderResourceView(5, verticesAddress);

    BakedGIRayDebugParams p = {};
    p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
    p.spacing  = m_Spacing;
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
    p.probeIndex         = m_RayDebugProbe;
    p.rayRecUAVIdx       = (uint32_t)m_RayDebugBuf.uavIndex;
    // The final iteration (kBakeIterations - 1 = 2) reads the previous
    // iteration's table for its feedback term. With the 3-iteration odd
    // ping-pong, that buffer is 1 - m_FinalResponse. This is the SKY feedback
    // source, and the sky replay keeps it: the sky transport ping-pongs on the same
    // schedule as the sun series.
    p.responsePrevSRVIdx = (uint32_t)m_Response[1 - m_FinalResponse].srvIndex;
    p.visSRVIdx          = (uint32_t)m_Visibility.srvIndex;

    // The one field that selects the table, the stride, the record index, the delta
    // direction and the seed TOGETHER. The shader reads anything that is not 1 as
    // SUN, so the zero-initialised struct would already mean SUN; the assignment is
    // explicit so a reader sees the source travel. There is no sky-bin field: a SKY
    // dispatch takes the bin from each group's own index, which is what lets ONE
    // dispatch fill all 16 groups.
    p.source       = (uint32_t)m_RayDebugSource;

    // SUN: replay ONE sun hour state. Its feedback AND its stored record both come
    // from the sun table (stride kSunStates, indexed by sunState), which is what the
    // bake's sun pass wrote. Everything below is the shipped path, unchanged.
    if (sunState >= kSunStates)
        sunState = kSunStates - 1u;
    m_RayDebugSunState = sunState;
    p.sunState = m_RayDebugSunState;

    if (m_RayDebugSource == RayDebugSource::Sky)
    {
        // SKY: the bake's sky transport, for EVERY fibonacci bin in one dispatch.
        // Each group d traces bin d, so the seed must be d as well — the bake seeds
        // bin j from j. The own-direction sentinel is exactly that request, and it
        // is the only value that can be shared by 16 groups: a fixed bin here would
        // seed all of them from one index and 15 would report MISMATCH.
        // The feedback table stays the sky table set above (stride kDirections,
        // indexed by the group index in the shader). sunDir stays zero: the SKY
        // branch of the shader never reads it.
        p.seedDir = kSunSeedOwnDir;
    }
    else
    {
        // The seed, resolved exactly as the bake resolves it. The bake seeds each
        // hour state from its own index, so the replay must carry that same state
        // index. Under the binned fallback it is fixed to kSunSeedOwnDir, because
        // that is what the binned bake always uses — passing a bin here would
        // compare this replay against a table baked from `dir`.
        p.seedDir = m_SunResponseValid ? m_RayDebugSunState : kSunSeedOwnDir;
        if (m_SunResponseValid)
        {
            // The state's own direction, from the one formula. Derived, never
            // passed in: a caller-supplied direction could describe another hour
            // and the self-check would then report a MISMATCH with no visible
            // cause.
            const DirectX::XMFLOAT3 d = SunDirectionAtHourCPU(m_RayDebugSunState);
            p.sunDirX = d.x;
            p.sunDirY = d.y;
            p.sunDirZ = d.z;
            p.responsePrevSRVIdx = (uint32_t)m_SunResponse[1 - m_FinalSunResponse].srvIndex;
        }
    }

    cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIRayDebugParams) / 4, &p, 0);
    cmdList->SetPipelineState(m_RayDebugPSO.Get());
    // kDirections groups, always. A SUN dump works in group 0 and returns at once
    // from the other 15. A SKY dump works in every group, because group d traces
    // bin d — so ONE click fills all 16 record groups and the whole re-accumulated
    // table, and the bin browser then needs no second dump.
    //
    // COST. A SKY dump therefore replays kDirections x kBakeGatherRays ray
    // traversals instead of one direction's worth, which is about 16x the GPU work
    // of the single-bin dump. It is a synchronous debug click, it never runs in a
    // normal frame, and the extra time is the price of browsing all 16 bins without
    // a wait. It is NOT a cost per frame.
    cmdList->Dispatch(1, kDirections, 1);

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_RayDebugBuf.resource.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }

    // Debug buffer -> readback.
    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
            m_RayDebugBuf.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b);
        cmdList->CopyBufferRegion(m_RayDebugReadback.Get(), 0, m_RayDebugBuf.resource.Get(), 0, kRayDebugBufBytes);
        D3D12_RESOURCE_BARRIER b2 = CD3DX12_RESOURCE_BARRIER::Transition(
            m_RayDebugBuf.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);
    }

    // The probe's slice of the stored table -> readback, for the self-check.
    {
        // WHICH TABLE, and which slice of it. The source decides, and nothing else
        // does: the two tables have different strides, so the wrong branch copies
        // bytes that belong to another probe or another record.
        //
        //   SUN  ONE record: the probe's record for the replayed hour state. The
        //        full 16-direction slice would read past the sun row.
        //   SKY  the probe's whole 16-record sky slice, so the readout can rank all
        //        sixteen bins by stored DC luminance AND check every one of them
        //        against its own re-accumulated record.
        const bool sunSource  = (m_RayDebugSource != RayDebugSource::Sky);
        const bool sunTable   = sunSource && m_SunResponseValid;
        ID3D12Resource* stored = sunTable
            ? m_SunResponse[m_FinalSunResponse].resource.Get()
            : m_Response[m_FinalResponse].resource.Get();
        const UINT64 srcOffset = sunTable
            ? (UINT64)(m_RayDebugProbe * kSunStates + m_RayDebugSunState) * kResponseFloat4s * 16u
            : (UINT64)m_RayDebugProbe * kDirections * kResponseFloat4s * 16u;
        const UINT64 tableBytes = sunTable
            ? (UINT64)kResponseFloat4s * 16u : kRayDebugTableBytes;
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
            stored, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &b);
        cmdList->CopyBufferRegion(m_RayDebugReadback.Get(), kRayDebugStoredOff, stored,
                                  srcOffset, tableBytes);
        D3D12_RESOURCE_BARRIER b2 = CD3DX12_RESOURCE_BARRIER::Transition(
            stored, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &b2);
    }

    return true;
}

void BakedGI::TransitionRayDebugForRead(ID3D12GraphicsCommandList* cmdList)
{
    if (!cmdList || !m_RayDebugBuf.resource)
        return;
    // PLACEMENT. The transition lives at the read site, not inside
    // RecordRayDebug. Two reasons:
    //   1. CreateStructuredBuffer calls CreateCommittedResource on every dump,
    //      so each dump builds a NEW resource with state = UNORDERED_ACCESS.
    //      The tracker therefore cannot go stale between dumps, and the next
    //      dispatch needs no repair.
    //   2. The read site owns its own read barrier, the way
    //      Renderer::DrawBakedGIProbeDebug transitions the GBuffer depth.
    // GPUResource::Transition guards on a state change, so drawing the fan on
    // two frames in a row emits one barrier, not two.
    m_RayDebugBuf.Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

bool BakedGI::SetRayDebugSkyDirection(uint32_t dir)
{
    // CPU STATE ONLY. This function deliberately contains no dispatch, no copy, no
    // Map and no wait, because a sky dump records every bin: another bin is already
    // in the buffer. Moving the display therefore costs nothing and cannot stall
    // the frame. If a future change ever needs a dispatch here, the browser becomes
    // a dump and the UI promise ("no dump, no wait") becomes false.
    if (m_RayDebugSource != RayDebugSource::Sky)
        return false; // a SUN dump holds one group; there is no sky bin to show
    if (!m_RayDebugBuf.resource)
        return false; // no records at all
    if (dir >= kDirections)
        return false; // out of range: keep the displayed bin, never clamp silently

    m_RayDebugSkyDirection = dir;
    return true;
}

BakedGI::ProbeMetaStatus BakedGI::ReadProbeMeta(uint32_t probe, uint32_t& metaWord) const
{
    // The meta readback is created by RecordMetaReadback on both the cold-bake
    // and the cache-load path, and it persists for the life of the bake. Map it
    // again here: the buffer is CPU-readable and nothing writes it after the
    // bake, so a second Map is safe (LogBakeStats uses the same buffer).
    if (!m_MetaReadback)
        return ProbeMetaStatus::NoReadback;
    if (probe >= m_ProbeCount)
        return ProbeMetaStatus::ProbeOutOfRange;

    uint32_t* meta = nullptr;
    if (FAILED(m_MetaReadback->Map(0, nullptr, reinterpret_cast<void**>(&meta))))
        return ProbeMetaStatus::MapFailed;
    metaWord = meta[probe];
    m_MetaReadback->Unmap(0, nullptr);
    return ProbeMetaStatus::Ok;
}

const std::vector<BakedGI::RayCandidate>& BakedGI::GetRayCandidates(uint32_t dir) const
{
    // An out-of-range direction gets an empty list rather than a clamp: the
    // caller asked about a direction that does not exist, and answering with
    // direction 0's pool would put another direction's rays under that label.
    static const std::vector<RayCandidate> kEmpty;
    return (dir < kDirections) ? m_RayCandidates[dir] : kEmpty;
}

void BakedGI::LogRayDebugReadout(const DirectX::XMFLOAT3* sunToDir) const
{
    if (!m_RayDebugReadback)
    {
        std::cerr << "[BakedGI RayDebug] no readback buffer — run RecordRayDebug first" << std::endl;
        return;
    }

    uint8_t* mapped = nullptr;
    if (FAILED(m_RayDebugReadback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
    {
        std::cerr << "[BakedGI RayDebug] failed to map the readback buffer" << std::endl;
        return;
    }

    const uint8_t* tableReacc  = mapped + (UINT64)kRayDebugTableBase * 16u;
    const uint8_t* tableStored = mapped + kRayDebugStoredOff;

    // --- WHICH SOURCE THIS READOUT DESCRIBES ---
    // Every label, index and verdict below derives from these three values, so the
    // readout cannot name a table the dump did not read. The two tables have
    // different strides and different meanings for the same integer, so a single
    // confused index would compare a sun record against a sky record and report a
    // MISMATCH that is purely an addressing difference.
    //
    //   skySource  the dump replayed the sky table. It traced EVERY fibonacci bin,
    //              so all kDirections record groups hold records and every bin can
    //              be checked and displayed.
    //   sunTable   a sun dump whose sun table exists, so ONE record was copied.
    //              Without the sun table the shipped binned fallback runs.
    //   focusDir   the bin ON DISPLAY, which is also the record GROUP the Ray Fan
    //              and the Cage Trace read. It is where the detailed tables below
    //              start, not the set of bins that were traced.
    const bool     skySource = (m_RayDebugSource == RayDebugSource::Sky);
    const bool     sunTable  = !skySource && m_SunResponseValid;
    const uint32_t focusDir  = skySource ? m_RayDebugSkyDirection : 0u;

    std::cout << "[BakedGI RayDebug] probe " << m_RayDebugProbe
              << " (grid " << m_DimX << "x" << m_DimY << "x" << m_DimZ
              << " @ " << m_Spacing << "m)" << std::endl;
    // Name the source BEFORE any number. Every number below describes one record of
    // one table, and the reader must know which one.
    if (skySource)
    {
        // The direction is the CPU mirror of BakedGIDeltaDirection, so the reader
        // can see the vector the shader traced instead of having to derive it.
        const DirectX::XMFLOAT3 d = DeltaDirection(focusDir);
        std::cout << "    source SKY — all " << kDirections
                  << " sky directions were traced; the display is bin " << focusDir
                  << " of " << kDirections << ", direction (" << d.x << ", " << d.y << ", " << d.z
                  << "), record group " << focusDir << std::endl;
    }
    else if (sunTable)
    {
        std::cout << "    source SUN — hour state " << m_RayDebugSunState
                  << " (hour " << (kSunStateHour0 + kSunStateStepHours * (float)m_RayDebugSunState)
                  << "), record group 0" << std::endl;
    }
    else
    {
        std::cout << "    source SUN — binned fallback: no sun table, so the replay seeded from"
                     " the direction index and compared against the sky table" << std::endl;
    }
    // The two counts differ on purpose, so print both. Confusing them is how a
    // reader would size a ranking against the wrong set. The GROUP count differs by
    // source: a SUN dump writes one group, a SKY dump writes all sixteen.
    std::cout << "    " << kBakeGatherRays << " gather rays for each traced direction were"
                 " replayed; the " << kDebugGatherRays
              << " RECORD slots of a direction hold the rays that carry a direct term first."
              << std::endl;
    std::cout << "    Record groups written by this dump: "
              << (skySource ? kDirections : 1u) << " of " << kDirections << std::endl;

    // --- Validity gate. Read the probe's meta bit up front, so the reader has
    // the verdict before step 1. Do NOT infer invalidity from an all-zero
    // stored row. That inference is false in two cases:
    //   1. METAL. A fully metallic surface has diffAlbedo = albedo *
    //      (1 - metallic) = 0 (BakedGI_Bake.hlsl:241), so a probe in a closed
    //      cavity lined with metal stores an exactly zero row. That probe is
    //      still VALID under the DDGI backface rule. The re-trace reproduces
    //      the zero, so the row prints MATCH. A row test would stamp "invalid"
    //      on a MATCH, which is a false claim.
    //   2. SEALED VOLUME. A valid probe inside a sealed volume stores zero too,
    //      because a sky miss contributes nothing under a delta environment.
    // The bake zeroes EVERY stored value of a genuinely invalid probe
    // (BakedGI_Bake.hlsl:147-155), so for such a probe the step-1 ranking
    // describes nothing and the step-3 comparison must fail. That failure is
    // not a drifted RNG stream, and a debug tool must not invent a defect. ---
    uint32_t   probeMeta  = 0;
    const ProbeMetaStatus metaStatus = ReadProbeMeta(m_RayDebugProbe, probeMeta);
    const bool haveMeta   = (metaStatus == ProbeMetaStatus::Ok);
    const bool probeValid = haveMeta && (probeMeta & 0x1u) != 0u; // BAKED_GI_META_VALID
    // Complete the verdict for the Ray Fan through GetRayDebugKnownInvalid.
    // RecordRayDebug already cleared the pairing for this dump, so writing both
    // here is what makes the verdict exist. An early return above (no buffer, or
    // a failed map on the debug readback) leaves the pairing cleared, and the
    // getter then reports no verdict.
    // Unknown validity means "do NOT dim". Dimming states that the probe is
    // degenerate, and that claim needs the meta bit. The text readout assumes a
    // valid probe for this same case, so the picture agrees with the text.
    m_RayDebugValid      = !haveMeta || probeValid;
    m_RayDebugValidProbe = m_RayDebugProbe;
    if (haveMeta && !probeValid)
    {
        std::cout << "[BakedGI RayDebug] WARNING — probe " << m_RayDebugProbe
                  << " is INVALID under the DDGI criteria (meta 0x" << std::hex << probeMeta
                  << std::dec << ")." << std::endl;
        std::cout << "    The bake zeroed this probe's stored table, so step 1 and step 3 describe"
                  << " nothing." << std::endl;
        std::cout << "    The re-traced rays themselves are live, so step 2 and the Ray Fan view"
                  << " still show the geometry." << std::endl;
    }

    // --- Step 1: rank the stored records by the luminance of the DC term. ---
    // The count follows what the dump COPIED. A sun dump copies ONE record (its own
    // hour state), so ranking 16 would read 2400 B past the copy and print numbers
    // that belong to no bake at all. A sky dump copies the probe's whole 16-record
    // slice, so all sixteen bins are rankable.
    const uint32_t rankDirs = sunTable ? 1u : kDirections;
    std::vector<std::pair<float, uint32_t>> dirRank;
    dirRank.reserve(rankDirs);
    for (uint32_t h = 0; h < rankDirs; ++h)
    {
        const float* dc = RayDebugTable(tableStored, h, 0u);
        dirRank.emplace_back(RayDebugLuminance(dc[0], dc[1], dc[2]), h);
    }
    std::sort(dirRank.begin(), dirRank.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "[BakedGI RayDebug] step 1 — " << rankDirs
              << (rankDirs == 1u ? " stored record" : " directions")
              << ", stored DC luminance (high to low):" << std::endl;
    for (size_t i = 0; i < dirRank.size(); ++i)
    {
        const uint32_t h = dirRank[i].second;
        std::cout << "    " << (rankDirs == 1u ? "hour state " : "dir ")
                  << (h < 10u ? " " : "") << h
                  << "  dcLum " << dirRank[i].first << std::endl;
    }

    // Name the record the Ray Fan draws, next to the DC ranking. The two picks
    // differ on purpose: step 1 ranks by baked DC luminance, and the fan follows the
    // bin ON DISPLAY. Naming it unconditionally is safe: it is a fact of the dump and
    // of the panel's bin browser, not of the scene's light.
    if (skySource)
    {
        std::cout << "    the Ray Fan displays sky direction " << focusDir
                  << " (record group " << focusDir
                  << "); the panel's Sky Bin control moves it with no new dump" << std::endl;
    }
    else if (sunTable)
    {
        std::cout << "    the Ray Fan replays hour state " << m_RayDebugSunState
                  << " (hour " << (kSunStateHour0 + kSunStateStepHours * (float)m_RayDebugSunState)
                  << ")";
        // The scene's live to-sun direction is NOT what the replay traced: the
        // replay traces the baked hour state's own direction. Printing both answers
        // "is the light where the state says it is", which is the only question the
        // live light can answer here. A SKY replay ignores this argument, because no
        // sun record is involved.
        if (sunToDir)
        {
            std::cout << ", and the scene's to-sun direction is ("
                      << sunToDir->x << ", " << sunToDir->y << ", " << sunToDir->z << ")";
        }
        std::cout << std::endl;
    }
    // The binned fallback prints nothing here: it has no single replayed record to
    // name.

    // --- Step 2: rank the RECORDS of the displayed direction, by the luminance of
    // L (record component 2). ---
    // A SUN dump traced one hour state, so step 1's DC winner IS that record. A SKY
    // dump traced all sixteen bins while step 1 ranks the stored table, so the two
    // picks can differ. The detailed tables below describe ONE bin — the displayed
    // one — because 16 x 64 rows would bury the answer.
    //
    // A record is NOT a source ray. The bake traces kBakeGatherRays for each
    // direction and the debug pass keeps kDebugGatherRays of them, qualifiers
    // first (BakedGI_RayDebug.hlsl). So record 0..63 are an ORDERED SUBSET of
    // rays 0..255, and "ray" below is the source index read from rec3.w. A
    // reader who assumes record r is ray r will mis-name every ray.
    const uint32_t dcDir     = dirRank.empty() ? 0u : dirRank.front().second;
    // The displayed group: group 0 for a sun dump, the displayed bin's group for a
    // sky dump. In the shipped sun path dcDir is also 0, because a sun dump ranks
    // one record, so nothing moves there.
    const uint32_t recordDir = skySource ? focusDir : 0u;
    std::vector<std::pair<float, uint32_t>> rayRank;
    rayRank.reserve(kDebugGatherRays);
    for (uint32_t r = 0; r < kDebugGatherRays; ++r)
    {
        // The record index is in float4 units, so multiply by 4 for the float
        // offset (the same conversion RayDebugTable applies).
        const float* rec = RayDebugRecord(mapped, recordDir, r);
        const float* L = RayDebugField(rec, BakedGI::kRayRecF4L); // rec2 = (L.rgb, Vhit)
        rayRank.emplace_back(RayDebugLuminance(L[0], L[1], L[2]), r);
    }
    std::sort(rayRank.begin(), rayRank.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // This table describes ONE group: the displayed one. A SUN dump wrote group 0,
    // and a SKY dump wrote every group, so either way the rows below belong to a
    // record the dump traced.
    if (skySource)
    {
        std::cout << "[BakedGI RayDebug] step 2 — the DISPLAYED sky direction " << focusDir
                  << " (step 1's DC winner is dir " << dcDir
                  << (dcDir == focusDir ? ", the same bin)"
                                        : ", a different bin; every bin has records)")
                  << ", " << kDebugGatherRays << " records by L luminance (high to low), top 16:"
                  << std::endl;
    }
    else if (sunTable)
    {
        std::cout << "[BakedGI RayDebug] step 2 — hour state " << m_RayDebugSunState
                  << " (one direction, no DC-luminance pick; the 16 bins "
                     "carry no sun weight), "
                  << kDebugGatherRays << " records by L luminance (high to low), top 16:"
                  << std::endl;
    }
    else
    {
        // The binned fallback: no sun table, so the replay wrote group 0 and step 1
        // ranked the 16 stored directions. Naming a DC winner as the record source
        // would name a group that holds no records.
        std::cout << "[BakedGI RayDebug] step 2 — binned fallback: record group 0, the only group"
                     " the replay wrote (step 1 ranks all " << kDirections << " stored directions), "
                  << kDebugGatherRays << " records by L luminance (high to low), top 16:"
                  << std::endl;
    }
    for (size_t i = 0; i < rayRank.size() && i < 16u; ++i)
    {
        const uint32_t r = rayRank[i].second;
        const float* rec = RayDebugRecord(mapped, recordDir, r);
        const float* gdir = RayDebugField(rec, BakedGI::kRayRecF4GatherDir);
        const float* alb  = RayDebugField(rec, BakedGI::kRayRecF4Albedo);
        const float* L    = RayDebugField(rec, BakedGI::kRayRecF4L);
        // rec4 = (hit normal.xyz, instance index), rec5 = (hit world pos.xyz,
        // primitive index). -1 in w means the ray carried no surface.
        const float* idn  = RayDebugField(rec, BakedGI::kRayRecF4HitNormal);
        const float* idp  = RayDebugField(rec, BakedGI::kRayRecF4HitPos);
        // rec6/rec7 = the SECONDARY visibility ray of that hit. shw names its
        // status, so "Vhit 1" is not the only evidence a row carries.
        const float* shw  = RayDebugField(rec, BakedGI::kRayRecF4Shadow);
        const float* shi  = RayDebugField(rec, BakedGI::kRayRecF4ShadowInfo);
        std::cout << "    rec " << (r < 10u ? " " : "") << r
                  << "  ray " << RayDebugSourceRay(rec)
                  << "  L " << rayRank[i].first
                  << "  dist " << gdir[3]
                  << "  NdotD " << alb[3]
                  << "  Vhit " << L[3]
                  << "  status " << RayDebugStatus(rec)
                  << "  inst " << (int)idn[3]
                  << "  prim " << (int)idp[3]
                  << "  Lrgb(" << L[0] << ", " << L[1] << ", " << L[2] << ")"
                  << "  | shw " << RayDebugShadowStatusName(RayDebugShadowStatus(rec))
                  << " shdist " << shi[0]
                  << " shinst " << (int)shi[1]
                  << " shprim " << (int)shi[2]
                  << " shxyz(" << shw[0] << ", " << shw[1] << ", " << shw[2] << ")"
                  << std::endl;
    }

    // --- Step 2b: the hit-surface census. ---
    // Group every HIT of the probe's re-traced rays by the instance it landed on,
    // and print that group's ray count, mean diffuse albedo, mean hit position and
    // mean hit normal. rec4/rec5 carry that surface identity, so naming the mesh
    // behind a coloured ray needs no offline EvalResponseIndirect replay and no
    // normal inferred from NdotD. It also makes a tinted surface visible as a
    // NUMBER — the instance whose mean albedo is red is the one to go looking for.
    //
    // The scan covers all kDirections groups. A SKY dump fills every one of them,
    // so the census is the whole sphere. A SUN dump fills group 0 only, and the
    // other groups are a freshly zeroed buffer with status 0 (sky miss), so they
    // add nothing and the census stays the shipped one direction.
    //
    // The count covers every non-miss ray (status >= 0.5), backface rejects
    // included: a reject names a real surface too, and it is often the one being
    // hunted. The albedo mean is valid for a reject as well, because the record
    // fills rec1 BEFORE the backface test.
    // CAUTION: rec1 carries diffAlbedo = albedo * (1 - metallic), so a METAL
    // reads as black here. A black albedo is not a missing surface.
    struct HitGroup
    {
        uint32_t count = 0;
        float    albedo[3] = { 0.0f, 0.0f, 0.0f };
        float    pos[3]    = { 0.0f, 0.0f, 0.0f };
        float    nrm[3]    = { 0.0f, 0.0f, 0.0f };
    };
    std::vector<std::pair<uint32_t, HitGroup>> groups; // (instance index, group)
    for (uint32_t h = 0; h < kDirections; ++h)
    {
        for (uint32_t r = 0; r < kDebugGatherRays; ++r)
        {
            const float* rec = RayDebugRecord(mapped, h, r);
            if (RayDebugStatus(rec) < 0.5f) // status 0 = sky miss
                continue;
            const float* alb = RayDebugField(rec, BakedGI::kRayRecF4Albedo);
            const float* nrm = RayDebugField(rec, BakedGI::kRayRecF4HitNormal);
            const float* pos = RayDebugField(rec, BakedGI::kRayRecF4HitPos);
            const uint32_t inst = (uint32_t)std::max(0.0f, nrm[3]);
            HitGroup* g = nullptr;
            for (auto& e : groups)
            {
                if (e.first == inst) { g = &e.second; break; }
            }
            if (!g)
            {
                groups.emplace_back(inst, HitGroup{});
                g = &groups.back().second;
            }
            ++g->count;
            for (uint32_t c = 0; c < 3u; ++c)
            {
                g->albedo[c] += alb[c];
                g->nrm[c]    += nrm[c];
                g->pos[c]    += pos[c];
            }
        }
    }
    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.second.count > b.second.count; });

    std::cout << "[BakedGI RayDebug] step 2b — hit surfaces, " << kDebugGatherRays
              << " records of " << kBakeGatherRays << " replayed rays, source "
              << RayDebugSourceName(m_RayDebugSource);
    if (skySource)
        std::cout << ", all " << kDirections << " sky directions ("
                  << (kDirections * kDebugGatherRays)
                  << " records), grouped by instance:" << std::endl;
    else if (sunTable)
        std::cout << " hour state " << m_RayDebugSunState << ", grouped by instance:" << std::endl;
    else
        std::cout << " (binned fallback), grouped by instance:" << std::endl;
    if (groups.empty())
        std::cout << "    no hit rays — this probe sees sky only" << std::endl;
    for (size_t i = 0; i < groups.size() && i < 12u; ++i)
    {
        const uint32_t inst = groups[i].first;
        const HitGroup& g   = groups[i].second;
        const float    inv  = 1.0f / (float)g.count;
        std::cout << "    inst " << inst
                  << "  rays " << g.count
                  << "  albedo(" << g.albedo[0] * inv << ", " << g.albedo[1] * inv
                                 << ", " << g.albedo[2] * inv << ")"
                  << "  pos(" << g.pos[0] * inv << ", " << g.pos[1] * inv << ", " << g.pos[2] * inv << ")"
                  << "  normal(" << g.nrm[0] * inv << ", " << g.nrm[1] * inv << ", " << g.nrm[2] * inv << ")"
                  << std::endl;
    }
    if (groups.size() > 12u)
        std::cout << "    ... " << (groups.size() - 12u) << " more instance(s)" << std::endl;

    // --- Step 2c: the Cage Trace pool — the CPU side of the ray picker (view
    // 9). One entry for each shaded ray whose feedback term beats its direct term
    // at the hit ("direct < Efb"). ---
    //
    // The pool is built for EVERY group, and a SKY dump fills every group, so all
    // sixteen bins have a filled pool. A SUN dump writes group 0 only, so groups
    // 1..15 are zero in a freshly zeroed buffer and their lists come out empty by
    // construction. The UI reads the pool of GetRayDebugDir(), which is the
    // DISPLAYED group, so the combo enumerates the rays on screen and never another
    // direction's.
    //
    // The score is luminance(diffAlbedo * Efb) — the shader's own key. The bake's
    // band-0 coefficient is one constant times the ray's radiance (all 64 rays
    // share the solid angle 4*PI/64), so ranking by carried radiance IS ranking by
    // contribution to the stored response; diffAlbedo stays INSIDE the product,
    // because a ray onto a dark or blue surface carries a large Efb and
    // contributes almost nothing.
    {
        for (uint32_t h = 0; h < kDirections; ++h)
        {
            m_RayCandidates[h].clear();
            for (uint32_t r = 0; r < kDebugGatherRays; ++r)
            {
                const float* rec = RayDebugRecord(mapped, h, r);
                // rec3.w packs status + 4 * sourceRayIndex; decode the status.
                // The shader's test is |status - 2| <= 0.5.
                const float status = RayDebugStatus(rec);
                if (status < 1.5f || status > 2.5f)
                    continue; // a sky miss or a backface reject carries no L and no Efb

                const float* alb  = RayDebugField(rec, BakedGI::kRayRecF4Albedo); // (diffAlbedo.rgb, NdotD)
                const float* L    = RayDebugField(rec, BakedGI::kRayRecF4L);      // (L.rgb, Vhit)
                const float* efb  = RayDebugField(rec, BakedGI::kRayRecF4Efb);    // (Efb.rgb, packed)
                const float  ndotd = alb[3] * L[3]; // NdotD * Vhit, the direct term's weight

                RayCandidate c;
                c.ray       = r;
                c.sourceRay = RayDebugSourceRay(rec);
                c.albedoLum = RayDebugLuminance(alb[0], alb[1], alb[2]);
                c.directLum = RayDebugLuminance(alb[0] * ndotd, alb[1] * ndotd, alb[2] * ndotd);
                c.efbLum    = RayDebugLuminance(alb[0] * efb[0], alb[1] * efb[1], alb[2] * efb[2]);
                c.score     = c.efbLum; // identical on purpose: the key IS the carried feedback
                if (c.efbLum <= c.directLum)
                    continue; // the direct delta at the hit wins: this ray is not a borrow
                m_RayCandidates[h].push_back(c);
            }
        }
        m_RayCandidateDirs = kDirections;
    }

    {
        // The DISPLAYED group: it is the one whose records the Ray Fan and the Cage
        // Trace read, so the readout prints the same list the panel enumerates.
        // Sorted HERE only, on a copy: the stored list keeps ray order, so the UI
        // combo's rows do not re-sort under the user's cursor.
        std::vector<RayCandidate> rows = m_RayCandidates[recordDir];
        std::sort(rows.begin(), rows.end(),
                  [](const RayCandidate& a, const RayCandidate& b) { return a.score > b.score; });

        std::cout << "[BakedGI RayDebug] step 2c — Cage Trace pool, source "
                  << RayDebugSourceName(m_RayDebugSource);
        if (skySource)
            std::cout << " direction " << focusDir << " (displayed)";
        else if (sunTable)
            std::cout << " hour state " << m_RayDebugSunState;
        std::cout << " (record group " << recordDir << "): " << rows.size() << " of "
                  << kDebugGatherRays
                  << " records carry Efb > NdotD*Vhit (the picker's list, by score):" << std::endl;
        for (const RayCandidate& c : rows)
            std::cout << "    rec " << (c.ray < 10u ? " " : "") << c.ray
                      << "  ray " << c.sourceRay
                      << "  score " << c.score
                      << "  efb " << c.efbLum
                      << "  direct " << c.directLum
                      << "  alb " << c.albedoLum << std::endl;
        if (rows.empty())
            std::cout << "    none — the trace draws nothing in this direction" << std::endl;
    }

    // --- Step 2d: the SECONDARY visibility rays of the displayed group. ---
    // For each shaded record whose hit fired a shadow ray (NdotD > 0), this prints
    // where that ray ended. It exists because a white Vhit is the ray's own claim:
    // without an endpoint, a lit floor whose sky ray never stopped cannot be told
    // apart from a ray that leaves the RT scene. rec6/rec7 carry the answer, and
    // this prints it.
    //
    //   open     the ray committed nothing. shxyz is W, the unit active delta
    //            direction, and the drawn segment runs to the length cap.
    //   blocked  the ray committed a hit. shxyz is that hit's world position and
    //            shdist/shinst/shprim name it.
    //
    // The rows are ordered by section, then by distance: a blocked ray that stops
    // very close to its own hit is the interesting one, because the surface it
    // stops on is the surface that was supposed to be lit. Open rays follow, so the
    // section a reader is chasing does not move between dumps.
    {
        struct ShadowRow
        {
            uint32_t   rec;
            uint32_t   sourceRay;
            float      status;
            float      dist;
            float      inst;
            float      prim;
            float      xyz[3];
            float      ndotd;
        };
        std::vector<ShadowRow> shadowRows;
        shadowRows.reserve(kDebugGatherRays);
        uint32_t nNone = 0u, nOpen = 0u, nBlocked = 0u;
        for (uint32_t r = 0; r < kDebugGatherRays; ++r)
        {
            const float* rec = RayDebugRecord(mapped, recordDir, r);
            const float  shs = RayDebugShadowStatus(rec);
            if (shs < 0.5f)
            {
                ++nNone;
                continue;
            }
            const float* payload = RayDebugShadowPayload(rec);
            const float* info    = RayDebugShadowInfo(rec);
            ShadowRow row;
            row.rec       = r;
            row.sourceRay = RayDebugSourceRay(rec);
            row.status    = shs;
            row.dist      = info[0];
            row.inst      = info[1];
            row.prim      = info[2];
            row.xyz[0]    = payload[0];
            row.xyz[1]    = payload[1];
            row.xyz[2]    = payload[2];
            row.ndotd     = RayDebugField(rec, BakedGI::kRayRecF4Albedo)[3];
            if (shs > 1.5f) ++nBlocked; else ++nOpen;
            shadowRows.push_back(row);
        }
        // Blocked first, nearest first; then open, in record order. std::stable_sort
        // keeps the record order inside each key, so two dumps of the same buffer
        // print the same list.
        std::stable_sort(shadowRows.begin(), shadowRows.end(),
                         [](const ShadowRow& a, const ShadowRow& b)
                         {
                             const bool ab = (a.status > 1.5f);
                             const bool bb = (b.status > 1.5f);
                             if (ab != bb) return ab;
                             if (!ab) return false;       // both open: record order
                             return a.dist < b.dist;      // both blocked: nearest first
                         });

        std::cout << "[BakedGI RayDebug] step 2d — VHit secondary rays, source "
                  << RayDebugSourceName(m_RayDebugSource);
        if (skySource)
            std::cout << " direction " << focusDir << " (displayed)";
        else if (sunTable)
            std::cout << " hour state " << m_RayDebugSunState;
        std::cout << " (record group " << recordDir << "): " << nBlocked << " blocked, "
                  << nOpen << " open, " << nNone << " with no secondary ray"
                  << " — of " << kDebugGatherRays << " records:" << std::endl;
        if (shadowRows.empty())
        {
            std::cout << "    none — no shaded ray in this direction has NdotD > 0" << std::endl;
        }
        for (size_t i = 0; i < shadowRows.size() && i < 16u; ++i)
        {
            const ShadowRow& row = shadowRows[i];
            std::cout << "    rec " << (row.rec < 10u ? " " : "") << row.rec
                      << "  ray " << row.sourceRay
                      << "  NdotD " << row.ndotd
                      << "  shw " << RayDebugShadowStatusName(row.status);
            if (row.status > 1.5f)
            {
                std::cout << "  shdist " << row.dist
                          << "  shinst " << (int)row.inst
                          << "  shprim " << (int)row.prim;
            }
            std::cout << "  shxyz(" << row.xyz[0] << ", " << row.xyz[1] << ", "
                      << row.xyz[2] << ")" << std::endl;
        }
        if (shadowRows.size() > 16u)
            std::cout << "    ... " << (shadowRows.size() - 16u) << " more row(s)" << std::endl;
    }

    // --- Step 3: the self-check. Compare the re-accumulated table with the
    // stored table, element by element. ---
    //
    // WHICH RECORDS to compare. A SKY dump re-accumulated ALL kDirections groups
    // and copied the probe's all-kDirections stored slice, so it checks every bin
    // and prints a verdict for each one. One verdict for sixteen bins is not good
    // enough here: the panel can display any bin, and a single drifted bin would
    // hide behind fifteen good ones under a "worst of 16" MATCH.
    //
    // A SUN dump stays exactly as shipped: one verdict for the one record it
    // traced, or the shipped 16-way binned-fallback comparison.
    const uint32_t checkCount = skySource ? kDirections : (sunTable ? 1u : kDirections);
    const uint32_t checkFirst = 0u;

    float maxAbs = 0.0f, maxStored = 0.0f;
    uint32_t worstDir = 0, worstLm = 0;
    uint32_t mismatched = 0u; // bins (or records) that fail their own tolerance
    for (uint32_t h = checkFirst; h < checkFirst + checkCount; ++h)
    {
        // Per-record tolerance: a small stored row must not pass on the strength of
        // a large row elsewhere in the probe's slice.
        float recAbs = 0.0f, recStored = 0.0f;
        uint32_t recLm = 0;
        for (uint32_t lm = 0; lm < kResponseFloat4s; ++lm)
        {
            const float* a = RayDebugTable(tableReacc,  h, lm);
            const float* b = RayDebugTable(tableStored, h, lm);
            for (uint32_t c = 0; c < 3u; ++c) // xyz carry the data; w is always zero
            {
                const float d = std::fabs(a[c] - b[c]);
                if (d > recAbs) { recAbs = d; recLm = lm; }
                recStored = std::max(recStored, std::fabs(b[c]));
            }
        }
        const float recTol = 1e-4f * std::max(1.0f, recStored);
        const bool  recOk  = (recAbs <= recTol);
        if (!recOk)
            ++mismatched;
        if (recAbs > maxAbs) { maxAbs = recAbs; worstDir = h; worstLm = recLm; }
        maxStored = std::max(maxStored, recStored);
        if (skySource)
        {
            std::cout << "    bin " << (h < 10u ? " " : "") << h
                      << "  max |diff| " << recAbs
                      << "  max |stored| " << recStored
                      << "  " << (recOk ? "MATCH" : "MISMATCH") << std::endl;
        }
    }
    const float tol = 1e-4f * std::max(1.0f, maxStored);
    // Count-driven, not a second comparison of the same numbers: for one record this
    // is identical to "max |diff| <= tol", and for sixteen it cannot claim a MATCH
    // while a bin failed.
    const bool  pass = (mismatched == 0u);

    // The header names the SOURCE and WHICH RECORDS, so a MATCH below can never be
    // read as a verdict about another table or another record.
    std::cout << "[BakedGI RayDebug] step 3 — self-check, source "
              << RayDebugSourceName(m_RayDebugSource);
    if (skySource)
        std::cout << ", all " << kDirections << " sky directions (record index = direction)";
    else if (sunTable)
        std::cout << ", hour state " << m_RayDebugSunState << " (record index "
                  << m_RayDebugSunState << ")";
    std::cout << ": re-accumulated table vs stored table" << std::endl;
    std::cout << "    max |diff| " << maxAbs << "  (worst at dir " << worstDir
              << ", lm " << worstLm << ")   max |stored| " << maxStored
              << "   tol " << tol << std::endl;
    if (mismatched != 0u)
        std::cout << "    " << mismatched << " of " << checkCount
                  << (checkCount == 1u ? " record" : " records") << " exceed their own tolerance"
                  << std::endl;
    if (haveMeta && !probeValid)
        std::cout << "    NOT A VERDICT — the probe is invalid, so its stored table is all zero."
                  << " The mismatch above is expected." << std::endl;
    else
    {
        std::cout << "    " << (pass ? "MATCH — the re-trace reproduced the bake"
                                     : "MISMATCH — the RNG stream drifted; read the rays with care")
                  << " (source " << RayDebugSourceName(m_RayDebugSource);
        if (skySource)
            std::cout << ", all " << kDirections << " sky directions)" << std::endl;
        else if (sunTable)
            std::cout << ", hour state " << m_RayDebugSunState << ")" << std::endl;
        else
            std::cout << ", binned fallback)" << std::endl;
        if (!haveMeta)
        {
            // Name the cause. The two causes need different fixes, so one
            // shared "no meta readback" line would name the wrong reason after
            // a rebuild with a smaller grid.
            if (metaStatus == ProbeMetaStatus::NoReadback)
                std::cout << "    note: no meta readback buffer (the bake has not run), so the verdict"
                             " assumes a valid probe" << std::endl;
            else if (metaStatus == ProbeMetaStatus::ProbeOutOfRange)
                std::cout << "    note: probe " << m_RayDebugProbe << " is at or past the probe count "
                          << m_ProbeCount << " (the grid changed), so the verdict assumes a valid probe"
                          << std::endl;
            else
                std::cout << "    note: the meta readback could not be mapped, so the verdict assumes"
                             " a valid probe" << std::endl;
        }
    }

    m_RayDebugReadback->Unmap(0, nullptr);
}

// =============================================================================
// Equinox hour states — the CPU mirror
// =============================================================================
//
// ONE formula, stated once in the shader (BakedGISunDirectionAtHour,
// BakedGI.hlsli) and once here. Both call sites below go through this function,
// so the bake's 17 directions, the ray-debug replay direction and the frame
// constants cannot disagree with each other. A drift from the SHADER is the real
// risk: it leaves the bake and the fold on different arcs and the image stays
// plausible, so nothing reports it. The reference below is the guard.
//
// Reference values, kEquinoxSunY = 0.906106471, latitude = acos(kEquinoxSunY):
//   hour 12 -> (0.000000000, 0.906106471, 0.423049718)   <- the authored sun, EXACT
//   hour  6 -> (1.000000000, 0.000000000, 0.000000000)   <- sunrise, on the horizon
//   hour 18 -> (-1.000000000, 0.000000000, 0.000000000)  <- sunset, on the horizon
//   hour  6.75 -> (0.980785, 0.176773, 0.082533)
// Every one of them is unit length by construction: sin^2 H + cos^2 H = 1. There
// is no normalize here, so there is no near-zero denominator and no NaN case.
//
// The arithmetic is DOUBLE and the result is stored to float. That is what makes
// state 8 reproduce the authored float3 exactly: the latitude has to keep more
// than six decimals for the round trip, and a float latitude does not.
DirectX::XMFLOAT3 BakedGI::SunDirectionAtHour(float hour)
{
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double H      = 15.0 * ((double)hour - 12.0) * kDegToRad;
    const double latRad = std::acos(kEquinoxSunY);
    return DirectX::XMFLOAT3((float)(-std::sin(H)),
                             (float)(std::cos(latRad) * std::cos(H)),
                             (float)(std::sin(latRad) * std::cos(H)));
}

DirectX::XMFLOAT3 BakedGI::SunDirectionAtHourCPU(uint32_t state)
{
    return SunDirectionAtHour(kSunStateHour0 + kSunStateStepHours * (float)state);
}

// =============================================================================
// Fibonacci sky-quadrature directions — the CPU mirror
// =============================================================================
//
// ONE formula, stated once in the shader (BakedGIDeltaDirection,
// BakedGI.hlsli:33-39) and once here. The bake, the sky replay and this mirror all
// use it, so the direction the log prints is the direction the GPU traced.
//
// Every step is FLOAT, in the shader's own order, because the shader's literal
// 2.39996323f and its float divide are part of the answer: a double computation
// would land on a different last bit for the higher bins, where phi is near 28.8
// radians and the argument reduction of cos/sin amplifies the difference.
//
// The normalization divides by sqrt(dot) rather than multiplying by rsqrt. HLSL's
// normalize multiplies by rsqrt, whose hardware approximation can differ by about
// 2e-7 relative — far below the 1e-6 the reference check allows.
DirectX::XMFLOAT3 BakedGI::DeltaDirection(uint32_t i)
{
    const float y   = ((float)i + 0.5f) / (float)kDirections;
    const float phi = (float)i * 2.39996323f; // golden angle, the shader's literal
    const float r   = std::sqrt(std::max(0.0f, 1.0f - y * y));
    const float x   = r * std::cos(phi);
    const float z   = r * std::sin(phi);
    // Unit length by construction only in exact arithmetic: 1 - y^2 + y^2 = 1, but
    // the rounded sine and cosine break it. Normalize, as the shader does.
    const float len = std::sqrt(x * x + y * y + z * z);
    return DirectX::XMFLOAT3(x / len, y / len, z / len);
}

// Reference values of BakedGIDeltaDirection, read from DXC's OWN constant folding
// of that exact function, not from a second derivation:
//
//   dxc -T cs_6_6 -E main probe.hlsl -Fc probe.asm
//     Out[0] = float4(BakedGIDeltaDirection(0u),  0) -> ( 0.99951159954071045, 0.03125, 0)
//     Out[1] = float4(BakedGIDeltaDirection(12u), 0) -> (-0.54008126258850098, 0.78125, -0.31298670172691345)
//
// The measured difference between this mirror and the folded constants is exactly
// zero for both indices. The tolerance below is therefore for the rsqrt
// approximation on the GPU, not for the formula.
bool BakedGI::VerifyDeltaDirectionMirror()
{
    constexpr float kTol = 1e-6f;

    struct Ref
    {
        uint32_t i;
        float x, y, z;
    };
    static const Ref kRefs[] =
    {
        {  0u,  0.99951159954071045f, 0.03125f,  0.0f                  },
        { 12u, -0.54008126258850098f, 0.78125f, -0.31298670172691345f  },
    };

    for (const Ref& ref : kRefs)
    {
        const DirectX::XMFLOAT3 d = DeltaDirection(ref.i);
        if (std::fabs(d.x - ref.x) > kTol ||
            std::fabs(d.y - ref.y) > kTol ||
            std::fabs(d.z - ref.z) > kTol)
        {
            std::cerr << "[BakedGI] DELTA DIRECTION MIRROR DRIFT at direction " << ref.i
                      << ": got (" << d.x << ", " << d.y << ", " << d.z
                      << "), the shader folds (" << ref.x << ", " << ref.y << ", " << ref.z
                      << ")" << std::endl;
            return false;
        }
    }
    return true;
}

BakedGI::SunHourBlend BakedGI::SunBlendForHour(float hour)
{
    SunHourBlend b;
    b.u = (hour - kSunStateHour0) / kSunStateStepHours;
    // The clamp is what makes the night case safe with no branch: below hour 6 and
    // above hour 18 the end pair is returned and the sun irradiance is already
    // near zero, so the sun term fades out without a step.
    const float clamped = std::max(0.0f, std::min(b.u, (float)(kSunStates - 2)));
    b.i0 = (uint32_t)clamped;
    b.t  = std::max(0.0f, std::min(b.u - (float)b.i0, 1.0f));
    b.inDaylight = (hour >= kSunStateHour0) &&
                   (hour <= kSunStateHour0 + kSunStateStepHours * (float)(kSunStates - 1));
    return b;
}

uint32_t BakedGI::FindNearestValidProbes(const DirectX::XMFLOAT3& point, uint32_t k,
                                         std::vector<NearestProbe>& out) const
{
    out.clear();
    if (k == 0u || !m_Valid || m_ProbeCount == 0u || m_DimX == 0u || m_DimY == 0u)
        return 0u;
    // The meta readback is the validity source. Without it (the bake has not
    // run) there is no verified probe to report, and an unverified probe must
    // not be offered as valid.
    if (!m_MetaReadback)
        return 0u;

    // The grid is regular, so the cell of a probe index is a pure function of
    // the index. ONE definition, used by both the scan and the output pass, so
    // the two cannot disagree about which cell a row describes.
    const uint32_t layerSize = m_DimX * m_DimY; // probes in one z slice
    const auto cellOf = [this, layerSize](uint32_t p, uint32_t& cx, uint32_t& cy, uint32_t& cz)
    {
        cx = p % m_DimX;
        cy = (p / m_DimX) % m_DimY;
        cz = p / layerSize;
    };

    uint32_t* meta = nullptr;
    if (FAILED(m_MetaReadback->Map(0, nullptr, reinterpret_cast<void**>(&meta))))
        return 0u;

    // Keep the k smallest distances WITHOUT sorting the grid. A bounded max-heap
    // keyed on SQUARED distance: the front is the farthest survivor, so a closer
    // probe replaces it. k is a UI count (a handful); the grid is up to 128K.
    // Comparing d2 avoids a sqrt on every probe — the sqrt happens once per
    // surviving row, in the output pass below.
    std::vector<std::pair<float, uint32_t>> heap; // (distance^2, probe index)
    heap.reserve(k + 1u);

    for (uint32_t p = 0; p < m_ProbeCount; ++p)
    {
        if ((meta[p] & 0x1u) == 0u) // BAKED_GI_META_VALID
            continue;

        uint32_t cx = 0, cy = 0, cz = 0;
        cellOf(p, cx, cy, cz);
        const float dx = m_GridMin.x + (float)cx * m_Spacing - point.x;
        const float dy = m_GridMin.y + (float)cy * m_Spacing - point.y;
        const float dz = m_GridMin.z + (float)cz * m_Spacing - point.z;
        const float d2 = dx * dx + dy * dy + dz * dz;

        if (heap.size() < k)
        {
            heap.emplace_back(d2, p);
            std::push_heap(heap.begin(), heap.end());
        }
        else if (d2 < heap.front().first)
        {
            std::pop_heap(heap.begin(), heap.end()); // farthest lands at the back
            heap.back() = { d2, p };
            std::push_heap(heap.begin(), heap.end());
        }
    }

    m_MetaReadback->Unmap(0, nullptr);

    // Heap order is not distance order. sort_heap turns the max-heap into an
    // ascending run, which is the order the list prints.
    std::sort_heap(heap.begin(), heap.end());

    out.reserve(heap.size());
    for (const std::pair<float, uint32_t>& e : heap)
    {
        NearestProbe np;
        np.index  = e.second;
        cellOf(np.index, np.cellX, np.cellY, np.cellZ);
        np.center = { m_GridMin.x + (float)np.cellX * m_Spacing,
                      m_GridMin.y + (float)np.cellY * m_Spacing,
                      m_GridMin.z + (float)np.cellZ * m_Spacing };
        np.distance = std::sqrt(e.first);
        out.push_back(np);
    }
    return (uint32_t)out.size();
}

bool BakedGI::HasSunResponseFor(float hour) const
{
    if (!m_SunResponseValid)
        return false;

    // The table holds every hour state, so the only question left is whether the
    // hour is a value the fold can describe. The fold itself clamps, so the only
    // input it cannot survive is a NaN: NaN fails both compares and therefore
    // reports false here instead of reaching a descriptor index.
    return (hour >= 0.0f) && (hour <= 24.0f);
}

// -----------------------------------------------------------------------------
// Probes/ disk cache
// -----------------------------------------------------------------------------

bool BakedGI::CreateProbeBuffers()
{
    // The caller synced the GPU before recording, so releasing the previous
    // bake's buffers here is safe (same pattern as
    // Renderer::BuildAccelerationStructures).
    m_Response[0]   = {};
    m_Response[1]   = {};
    m_SunResponse[0] = {};
    m_SunResponse[1] = {};
    m_LitProbes     = {};
    m_LitProbesSky  = {};
    m_LitProbesSun  = {};
    m_Meta          = {};
    m_Visibility    = {};
    m_SkyVisSH      = {};
    m_SkyVisSHValid = false;

    const uint64_t responseElems = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s;
    // Sun table: one record for EACH hour state, per probe. 17 x 144 B = 2448 B
    // per probe per buffer. Created only when the sun pass runs, so a bake that
    // does not need the sun pays nothing.
    const uint64_t sunElems = (uint64_t)m_ProbeCount * kSunStates * kResponseFloat4s;
    if (!CreateStructuredBuffer(m_Response[0], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse0") ||
        !CreateStructuredBuffer(m_Response[1], sizeof(float) * 4, responseElems,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIResponse1") ||
        (!CreateStructuredBuffer(m_SunResponse[0], sizeof(float) * 4, sunElems,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGISun0") ||
         !CreateStructuredBuffer(m_SunResponse[1], sizeof(float) * 4, sunElems,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGISun1")) ||
        !CreateStructuredBuffer(m_LitProbes, sizeof(float) * 4, (uint64_t)m_ProbeCount * kLitFloat4s,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGILit") ||
        !CreateStructuredBuffer(m_LitProbesSky, sizeof(float) * 4, (uint64_t)m_ProbeCount * kLitFloat4s,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGILitSky") ||
        !CreateStructuredBuffer(m_LitProbesSun, sizeof(float) * 4, (uint64_t)m_ProbeCount * kLitFloat4s,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGILitSun") ||
        !CreateStructuredBuffer(m_Meta, sizeof(uint32_t), m_ProbeCount,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIMeta") ||
        !CreateStructuredBuffer(m_Visibility, sizeof(float) * 2, (uint64_t)m_ProbeCount * kVisStride,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGIVisibility") ||
        // Sky-visibility SH9 A/B data. Written by the SAME visibility dispatch,
        // which indexes this buffer through Vis.skyVisSHUAVIdx, so it is part of
        // the required chain and not an optional extra: a missing buffer would
        // make the shader write through descriptor 0. 36 B for each probe.
        !CreateStructuredBuffer(m_SkyVisSH, sizeof(float), (uint64_t)m_ProbeCount * kSkyVisSHCoeffs,
                                 D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_BakedGISkyVisSH"))
    {
        std::cerr << "[BakedGI] Failed to create probe buffers" << std::endl;
        return false;
    }
    // The buffer exists but holds no bake yet. The visibility dispatch below
    // fills it, and TryLoadCache clears the flag again because the cache has no
    // SH section.
    m_SkyVisSHValid = (m_SkyVisSH.resource != nullptr);
    return true;
}

bool BakedGI::CreateDilationBuffers()
{
    const uint64_t responseElems = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s;
    const uint64_t sunElems = (uint64_t)m_ProbeCount * kSunStates * kResponseFloat4s;

    // A positive/zero/positive bake sequence must reuse every descriptor slot.
    // First allocation assigns slots. Later allocations retain the same values.
    auto createDilationBuffer = [](GPUBuffer& buffer, uint64_t elementSize,
                                   uint64_t elementCount, const char* debugName)
    {
        const int previousSrv = buffer.srvIndex;
        const int previousUav = buffer.uavIndex;
        const int previousCpuUav = buffer.cpuUavIndex;
        if (!CreateStructuredBuffer(buffer, elementSize, elementCount,
                                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, debugName))
        {
            return false;
        }
        assert((previousSrv < 0 || buffer.srvIndex == previousSrv) &&
               (previousUav < 0 || buffer.uavIndex == previousUav) &&
               buffer.cpuUavIndex == previousCpuUav);
        return true;
    };

    if (!createDilationBuffer(m_DilatedResponse[0], sizeof(float) * 4, responseElems, "SB_BakedGIDilated0") ||
        !createDilationBuffer(m_DilatedResponse[1], sizeof(float) * 4, responseElems, "SB_BakedGIDilated1") ||
        !createDilationBuffer(m_DilatedSunResponse[0], sizeof(float) * 4, sunElems, "SB_BakedGIDilatedSun0") ||
        !createDilationBuffer(m_DilatedSunResponse[1], sizeof(float) * 4, sunElems, "SB_BakedGIDilatedSun1") ||
        !createDilationBuffer(m_DilationMeta, sizeof(uint32_t), m_ProbeCount, "SB_BakedGIDilationMeta"))
    {
        std::cerr << "[BakedGI] Failed to create dilation buffers" << std::endl;
        return false;
    }
    return true;
}

bool BakedGI::RecordDilation(ID3D12GraphicsCommandList* cmdList)
{
    if (m_DilationIterations == 0u)
        return true;
    if (!cmdList || !m_DilatePSO || !m_DilationMeta.resource ||
        !m_DilatedResponse[0].resource || !m_DilatedResponse[1].resource ||
        !m_DilatedSunResponse[0].resource || !m_DilatedSunResponse[1].resource)
    {
        return false;
    }

    GPUBuffer* metaSource = &m_Meta;
    GPUBuffer* metaDest   = &m_DilationMeta;
    for (uint32_t it = 0; it < m_DilationIterations; ++it)
    {
        const uint32_t dst = it & 1u;
        const uint32_t src = 1u - dst;
        const GPUBuffer& skySource = (it == 0u) ? m_Response[m_FinalResponse]
                                                : m_DilatedResponse[src];
        const GPUBuffer& sunSource = (it == 0u) ? m_SunResponse[m_FinalSunResponse]
                                                : m_DilatedSunResponse[src];

        BakedGIDilationParams p = {};
        p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
        p.probeCount = m_ProbeCount;
        p.recordCount = kDirections;
        p.tableSRVIdx = (uint32_t)skySource.srvIndex;
        p.tableUAVIdx = (uint32_t)m_DilatedResponse[dst].uavIndex;
        p.metaSRVIdx = (uint32_t)metaSource->srvIndex;
        p.metaUAVIdx = (uint32_t)metaDest->uavIndex;
        p.writeMeta = 1u;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIDilationParams) / 4, &p, 0);
        cmdList->SetPipelineState(m_DilatePSO.Get());
        cmdList->Dispatch((m_ProbeCount + 63u) / 64u, kDirections, 1);

        BakedGIDilationParams sp = p;
        sp.recordCount = kSunStates;
        sp.tableSRVIdx = (uint32_t)sunSource.srvIndex;
        sp.tableUAVIdx = (uint32_t)m_DilatedSunResponse[dst].uavIndex;
        sp.writeMeta = 0u;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIDilationParams) / 4, &sp, 0);
        cmdList->Dispatch((m_ProbeCount + 63u) / 64u, kSunStates, 1);

        D3D12_RESOURCE_BARRIER barriers[3] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_DilatedResponse[dst].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_DilatedSunResponse[dst].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(metaDest->resource.Get()),
        };
        cmdList->ResourceBarrier(3, barriers);
        m_FinalDilatedResponse = dst;
        m_FinalDilatedSunResponse = dst;
        std::swap(metaSource, metaDest);
    }

    // Keep the public meta resource final on both parities. The other resource
    // becomes scratch and retains no separate runtime consumer.
    if (metaSource != &m_Meta)
        std::swap(m_Meta, m_DilationMeta);
    m_DilatedTransportValid = true;
    std::cout << "[BakedGI] Dilation recorded (" << m_DilationIterations
              << " iterations, separate runtime tables)" << std::endl;
    return true;
}

// Debug-only capture, created on the dispatch path only: the term snapshots and
// the folded per-probe response balls that the "Efb Contribution" view reads.
//
// SIZING. One snapshot is a whole transport table: probeCount x 16 x 9 float4,
// 2304 B for each probe (about 27 MB at 11,616 probes, 50 MB at 21,952). The
// kIterSnapshotCount = kBakeIterations - 1 snapshots plus kBakeIterations balls
// (144 B for each probe — the ball is a per-probe SH9) is the whole cost, and it
// is what the view costs when it is never opened. It is paid on a bake, which is
// a user action, not on the frame path.
//
// The GPUBuffer structs are deliberately NOT reset first. CreateStructuredBuffer
// reuses an existing descriptor slot when it has one, so a re-bake replaces the
// resource without consuming a fresh SRV/UAV pair from a heap that has no free —
// the pattern the ray-debug buffer documents in RecordRayDebug.
bool BakedGI::CreateIterationBuffers()
{
    const uint64_t responseElems = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s;

    for (uint32_t s = 0; s < kIterSnapshotCount; ++s)
    {
        char name[48];
        snprintf(name, sizeof(name), "SB_BakedGIResponseIter%u", s);
        // COPY_DEST: these buffers only ever receive a CopyBufferRegion during
        // the bake, and the fold pass transitions them for its SRV read.
        if (!CreateStructuredBuffer(m_ResponseIter[s], sizeof(float) * 4, responseElems,
                                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, name))
        {
            std::cerr << "[BakedGI] Failed to create the term snapshot " << s << std::endl;
            return false;
        }
    }

    for (uint32_t t = 0; t < kBakeIterations; ++t)
    {
        char name[48];
        snprintf(name, sizeof(name), "SB_BakedGIEfbBall%u", t);
        if (!CreateStructuredBuffer(m_EfbBall[t], sizeof(float) * 4, (uint64_t)m_ProbeCount * kEfbBallFloat4s,
                                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, name))
        {
            std::cerr << "[BakedGI] Failed to create the response ball " << t << std::endl;
            return false;
        }
    }
    return true;
}

void BakedGI::SnapshotResponseTable(ID3D12GraphicsCommandList* cmdList, uint32_t src, uint32_t snapshot)
{
    GPUBuffer& source = m_Response[src];
    GPUBuffer& dest   = m_ResponseIter[snapshot];
    const UINT64 bytes = (UINT64)m_ProbeCount * kDirections * kResponseFloat4s * 16u;

    // The source is in UNORDERED_ACCESS (the bake dispatch wrote it), the
    // destination in COPY_DEST (CreationIterationBuffers). Round-trip the source
    // so the next transport iteration still finds the state it expects — the same
    // record-copy-record shape as the ray-debug dump's stored-table copy.
    source.Transition(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->CopyBufferRegion(dest.resource.Get(), 0, source.resource.Get(), 0, bytes);
    source.Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void BakedGI::TransitionEfbBallsForRead(ID3D12GraphicsCommandList* cmdList)
{
    if (!cmdList)
        return;
    // See TransitionRayDebugForRead for why this sits at the read site and not in
    // the bake: every bake builds NEW resources (CreateIterationBuffers calls
    // CreateCommittedResource), so the state tracker cannot go stale between
    // bakes, and GPUResource::Transition makes the call free after the first.
    for (uint32_t t = 0; t < kBakeIterations; ++t)
        m_EfbBall[t].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // The view's single-direction mode reads the TRANSPORT TABLES, not the balls:
    // one direction cannot be recovered from a 16-direction sum. The bake leaves
    // them tracked as UNORDERED_ACCESS and reads them as SRVs behind a UAV barrier
    // alone — the convention RecordBake documents at :373-374, which is fine for a
    // compute read on the bake's own list. This read happens on a LATER FRAME and
    // from the VERTEX stage, so the read site takes the real barrier rather than
    // inheriting that assumption. Transition guards the change, so this costs one
    // barrier per bake, not one per frame.
    for (uint32_t s = 0; s < kIterSnapshotCount; ++s)
        m_ResponseIter[s].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_Response[m_FinalResponse].Transition(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

bool BakedGI::TryLoadCache(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                           const std::string& scenePath)
{
    namespace fs = std::filesystem;
    if (scenePath.empty())
        return false;

    const fs::path cachePath = ProbeCachePath(scenePath, m_Spacing);

    // Hit = the file exists + the header validates against the grid this
    // run computed (the map stem + spacing are already in the path). No
    // mtime freshness during development — delete the cache or bump the
    // version after editing the bake.

    // Read + validate the header against the grid RecordBake just computed.
    std::ifstream in(cachePath, std::ios::binary);
    if (!in.is_open())
        return false;
    ProbeCacheHeader h = {};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in)
        return false;

    // The version test comes FIRST and it logs. An older file cannot restore the
    // raw and dilated data this build needs. A silent rejection would look like
    // a missing file and cost a session.
    if (h.magic != kProbeCacheMagic)
        return false;
    if (h.version != kProbeCacheVersion)
    {
        std::cout << "[BakedGI] Probe cache REJECTED: version " << h.version
                  << ", this build writes and reads version " << kProbeCacheVersion
                  << " (raw and dilated transport layout). Delete "
                  << cachePath.string() << " and bake again." << std::endl;
        return false;
    }
    if (h.dimX != m_DimX || h.dimY != m_DimY || h.dimZ != m_DimZ ||
        h.probeCount != m_ProbeCount || h.finalResponse > 1 ||
        std::fabs(h.spacing - m_Spacing) > 1e-6f ||
        std::fabs(h.gridMinX - m_GridMin.x) > 1e-4f ||
        std::fabs(h.gridMinY - m_GridMin.y) > 1e-4f ||
        std::fabs(h.gridMinZ - m_GridMin.z) > 1e-4f)
    {
        return false;
    }
    if (h.dilationIterations != m_DilationIterations)
    {
        std::cout << "[BakedGI] Probe cache REJECTED: dilation count " << h.dilationIterations
                  << " does not match requested count " << m_DilationIterations << std::endl;
        return false;
    }
    // The 17 state directions are one function of the authored sun's vertical
    // component. A file baked with another value holds a well-formed table of the
    // WRONG directions, and nothing else in the header can see that.
    if (h.sunStates != kSunStates || std::fabs(h.sunUpCos - (float)kEquinoxSunY) > 1e-6f)
    {
        std::cout << "[BakedGI] Probe cache REJECTED: it holds " << h.sunStates
                  << " sun states derived from sunUpCos " << h.sunUpCos << ", this build needs "
                  << kSunStates << " from " << (float)kEquinoxSunY << ". Delete "
                  << cachePath.string() << " and bake again." << std::endl;
        return false;
    }

    // Payload order: raw response | meta | vis | raw sun. Positive counts append
    // final dilated response | final dilated sun. No scratch table is serialized.
    const uint64_t responseBytes = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s * 16;
    const uint64_t metaBytes     = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    const uint64_t visBytes      = (uint64_t)m_ProbeCount * kVisStride * sizeof(float) * 2;
    const uint64_t sunBytes      = (uint64_t)m_ProbeCount * kSunStates * kResponseFloat4s * 16;
    const uint64_t sunOffset     = responseBytes + metaBytes + visBytes;
    const uint64_t dilatedResponseOffset = sunOffset + sunBytes;
    const uint64_t dilatedSunOffset = dilatedResponseOffset + responseBytes;
    const uint64_t total = dilatedResponseOffset
                         + ((m_DilationIterations != 0u) ? responseBytes + sunBytes : 0u);

    std::vector<char> blob(static_cast<size_t>(total));
    in.read(blob.data(), static_cast<std::streamsize>(total));
    if (in.fail() || in.gcount() != static_cast<std::streamsize>(total))
    {
        std::cerr << "[BakedGI] Probe cache truncated: " << cachePath.string()
                  << " (needed " << total << " B of payload, read " << in.gcount() << " B)"
                  << std::endl;
        return false;
    }

    // The sun buffer must exist before the copy targets it. There is no variant
    // that skips the sun section: the file always carries it and the fold always
    // needs it.
    if (!CreateProbeBuffers())
        return false;
    if (m_DilationIterations != 0u && !CreateDilationBuffers())
        return false;
    m_FinalResponse = h.finalResponse;
    // The cache carries no sky-visibility SH9 section, so the buffer created
    // above holds undefined bytes. Mark it invalid: the fold then zeroes V and
    // the UI reports SH9 as unavailable, instead of reading a NaN field.
    m_SkyVisSHValid = false;
    if (m_DilationIterations != 0u)
    {
        m_FinalDilatedResponse = (m_DilationIterations - 1u) & 1u;
        m_FinalDilatedSunResponse = m_FinalDilatedResponse;
    }

    // One UPLOAD staging buffer for every section — kept alive as a member until
    // the next bake (the recorded list outlives this call).
    D3D12_HEAP_PROPERTIES up = {};
    up.Type = D3D12_HEAP_TYPE_UPLOAD;
    const D3D12_RESOURCE_DESC upDesc = CD3DX12_RESOURCE_DESC::Buffer(total);
    if (FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &upDesc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&m_CacheUpload))))
    {
        std::cerr << "[BakedGI] Failed to create the cache upload buffer" << std::endl;
        return false;
    }
    GraphicsHelper::SetObjectName(m_CacheUpload.Get(), "UP_BakedGICacheLoad");

    {
        uint8_t* mapped = nullptr;
        if (FAILED(m_CacheUpload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
            return false;
        memcpy(mapped, blob.data(), static_cast<size_t>(total));
        m_CacheUpload->Unmap(0, nullptr);
    }

    struct Copy { GPUBuffer* dst; uint64_t offset; uint64_t bytes; };
    std::vector<Copy> copies = {
        { &m_Response[m_FinalResponse], 0,             responseBytes },
        { &m_Meta,                      responseBytes, metaBytes     },
        { &m_Visibility,                responseBytes + metaBytes, visBytes },
        { &m_SunResponse[m_FinalResponse], sunOffset,  sunBytes      },
    };
    if (m_DilationIterations != 0u)
    {
        copies.push_back({ &m_DilatedResponse[m_FinalDilatedResponse], dilatedResponseOffset, responseBytes });
        copies.push_back({ &m_DilatedSunResponse[m_FinalDilatedSunResponse], dilatedSunOffset, sunBytes });
    }
    for (const Copy& c : copies)
    {
        c.dst->Transition(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);
        cmdList->CopyBufferRegion(c.dst->resource.Get(), 0, m_CacheUpload.Get(), c.offset, c.bytes);
        c.dst->Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    // Restore the sun bookkeeping EXACTLY as RecordBake leaves it after a bake.
    // Without m_SunResponseValid the fold reads an unbacked descriptor, and the
    // method-change path also treats the resident table as missing and rebakes.
    m_FinalSunResponse = m_FinalResponse;
    m_SunResponseValid = (m_SunResponse[0].resource != nullptr);
    m_DilatedTransportValid = (m_DilationIterations != 0u);

    m_LoadedFromCache = true;
    m_Valid = true;
    std::cout << "[BakedGI] Probe cache hit: " << m_ProbeCount << " probes ("
              << m_DimX << "x" << m_DimY << "x" << m_DimZ << " @ " << m_Spacing
              << "m) — " << cachePath.string() << std::endl;
    if (m_SunResponseValid)
        LogSunTableHash(reinterpret_cast<const uint8_t*>(blob.data() + sunOffset),
                        m_ProbeCount, "loaded from cache");
    else
        std::cout << "[BakedGI] sun table NOT loaded: the caller did not ask for the sun term"
                  << std::endl;
    return true;
}

bool BakedGI::RecordCacheReadback(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList)
{
    if (!m_Valid || !device || !cmdList)
        return false;

    // A version-3 file always carries the sun section, and a bake that produced
    // no sun table cannot fill it. Refuse rather than write a short file: a short
    // file would fail the truncated-read test on the load side with a message that
    // blames the disk.
    if (!m_SunResponseValid || !m_SunResponse[0].resource)
    {
        std::cout << "[BakedGI] Probe cache NOT written: this bake produced no sun table, "
                     "and the version-" << kProbeCacheVersion << " payload requires one"
                  << std::endl;
        return false;
    }
    if (m_DilationIterations != 0u && !m_DilatedTransportValid)
    {
        std::cout << "[BakedGI] Probe cache NOT written: dilated tables are unavailable" << std::endl;
        return false;
    }

    const uint64_t responseBytes = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s * 16;
    const uint64_t metaBytes     = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    const uint64_t visBytes      = (uint64_t)m_ProbeCount * kVisStride * sizeof(float) * 2;
    const uint64_t sunBytes      = (uint64_t)m_ProbeCount * kSunStates * kResponseFloat4s * 16;
    const uint64_t sunOffset     = responseBytes + metaBytes + visBytes;
    const uint64_t dilatedResponseOffset = sunOffset + sunBytes;
    const uint64_t total = dilatedResponseOffset
                         + ((m_DilationIterations != 0u) ? responseBytes + sunBytes : 0u);

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(total);
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&m_CacheReadback))))
    {
        std::cerr << "[BakedGI] Failed to create the cache readback buffer" << std::endl;
        return false;
    }
    GraphicsHelper::SetObjectName(m_CacheReadback.Get(), "RB_BakedGICache");

    // Raw response | meta | vis | raw sun occupy the base payload. Positive
    // counts append the final dilated response and dilated sun. TryLoadCache and
    // WriteCacheFile use the same offsets. Keep all three places in step.
    struct Copy { GPUBuffer* src; uint64_t dstOffset; uint64_t bytes; };
    std::vector<Copy> copies = {
        { &m_Response[m_FinalResponse],    0,             responseBytes },
        { &m_Meta,                         responseBytes, metaBytes     },
        { &m_Visibility,                   responseBytes + metaBytes, visBytes },
        { &m_SunResponse[m_FinalSunResponse], sunOffset,  sunBytes      },
    };
    if (m_DilationIterations != 0u)
    {
        copies.push_back({ &m_DilatedResponse[m_FinalDilatedResponse], dilatedResponseOffset, responseBytes });
        copies.push_back({ &m_DilatedSunResponse[m_FinalDilatedSunResponse], dilatedResponseOffset + responseBytes, sunBytes });
    }
    for (const Copy& c : copies)
    {
        c.src->Transition(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->CopyBufferRegion(m_CacheReadback.Get(), c.dstOffset, c.src->resource.Get(), 0, c.bytes);
        c.src->Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    return true;
}

void BakedGI::WriteCacheFile(const std::string& scenePath)
{
    if (!m_CacheReadback || scenePath.empty())
        return;

    const std::filesystem::path cachePath = ProbeCachePath(scenePath, m_Spacing);
    if (cachePath.empty())
        return;

    ProbeCacheHeader h = {};
    h.magic = kProbeCacheMagic;
    h.version = kProbeCacheVersion;
    h.dimX = m_DimX; h.dimY = m_DimY; h.dimZ = m_DimZ;
    h.probeCount = m_ProbeCount;
    h.finalResponse = m_FinalResponse;
    h.spacing = m_Spacing;
    h.gridMinX = m_GridMin.x; h.gridMinY = m_GridMin.y; h.gridMinZ = m_GridMin.z;
    // The latitude record. Without it a file baked at another latitude loads as a
    // well-formed table of the wrong directions, and nothing reports it.
    h.sunStates = kSunStates;
    h.sunUpCos  = (float)kEquinoxSunY;
    h.dilationIterations = m_DilationIterations;

    const uint64_t responseBytes = (uint64_t)m_ProbeCount * kDirections * kResponseFloat4s * 16;
    const uint64_t metaBytes     = (uint64_t)m_ProbeCount * sizeof(uint32_t);
    const uint64_t visBytes      = (uint64_t)m_ProbeCount * kVisStride * sizeof(float) * 2;
    const uint64_t sunBytes      = (uint64_t)m_ProbeCount * kSunStates * kResponseFloat4s * 16;
    const uint64_t sunOffset     = responseBytes + metaBytes + visBytes;
    const uint64_t dilatedResponseOffset = sunOffset + sunBytes;
    const uint64_t total = dilatedResponseOffset
                         + ((m_DilationIterations != 0u) ? responseBytes + sunBytes : 0u);

    uint8_t* mapped = nullptr;
    if (FAILED(m_CacheReadback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
    {
        std::cerr << "[BakedGI] Failed to map the cache readback buffer" << std::endl;
        return;
    }

    std::ofstream out(cachePath, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "[BakedGI] Failed to open the probe cache for writing: "
                  << cachePath.string() << std::endl;
        m_CacheReadback->Unmap(0, nullptr);
        return;
    }
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(mapped + 0),
              static_cast<std::streamsize>(responseBytes));
    out.write(reinterpret_cast<const char*>(mapped + responseBytes),
              static_cast<std::streamsize>(metaBytes));
    out.write(reinterpret_cast<const char*>(mapped + responseBytes + metaBytes),
              static_cast<std::streamsize>(visBytes));
    out.write(reinterpret_cast<const char*>(mapped + sunOffset),
              static_cast<std::streamsize>(sunBytes));
    if (m_DilationIterations != 0u)
    {
        out.write(reinterpret_cast<const char*>(mapped + dilatedResponseOffset),
                  static_cast<std::streamsize>(responseBytes));
        out.write(reinterpret_cast<const char*>(mapped + dilatedResponseOffset + responseBytes),
                  static_cast<std::streamsize>(sunBytes));
    }

    // The round-trip hook. Hashed from the SAME bytes that were just written, so
    // a load-side hash that differs proves a transport or ordering fault rather
    // than a bake difference.
    LogSunTableHash(mapped + sunOffset, m_ProbeCount, "written to cache");

    m_CacheReadback->Unmap(0, nullptr);

    std::cout << "[BakedGI] Probe cache written: " << cachePath.string()
              << " (" << (total >> 20) << " MB, " << sunBytes << " B of it sun states)"
              << std::endl;
}

void BakedGI::RecordUpdate(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                           D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                           const DirectX::XMFLOAT3& sunIrradiance, float hour)
{
    if (!m_Valid || !m_UpdatePSO)
        return;

    BakedGIUpdateParams p = {};
    p.sunIrradianceX = sunIrradiance.x;
    p.sunIrradianceY = sunIrradiance.y;
    p.sunIrradianceZ = sunIrradiance.z;
    // The sun input is the hour. The fold picks the two baked states that bracket
    // it, so no live direction reaches the shader.
    p.hour               = hour;
    p.sunResponseSRVIdx  = GetUpdateSunResponseSRVIndex();
    if (p.sunResponseSRVIdx == 0u)
    {
        // No sun table: the fold would read descriptor 0 through the wrong view.
        // Report and leave the lit buffer alone rather than write garbage.
        std::cerr << "[BakedGI] hour-state fold requested with no sun table — skipped"
                  << std::endl;
        return;
    }
    p.responseSRVIdx = GetUpdateResponseSRVIndex();
    p.metaSRVIdx      = (uint32_t)m_Meta.srvIndex;
    p.litUAVIdx       = (uint32_t)m_LitProbes.uavIndex;
    p.litSkyUAVIdx    = (uint32_t)m_LitProbesSky.uavIndex;
    p.litSunUAVIdx    = (uint32_t)m_LitProbesSun.uavIndex;
    p.gridMinX = m_GridMin.x; p.gridMinY = m_GridMin.y; p.gridMinZ = m_GridMin.z;
    p.spacing  = m_Spacing;
    p.dimX = m_DimX; p.dimY = m_DimY; p.dimZ = m_DimZ;
    // Sky-visibility source for the V in the sky term. The EFFECTIVE mode is 0
    // only when the resident bake holds SH data, and 1 for the NoVisibility
    // diagnostic. Mode 1 reads no descriptor, so the SRV index is left at 0
    // there and the shader never touches it. Exactly two modes exist.
    {
        const SkyVisMode skyVis = GetEffectiveSkyVisMode();
        p.skyVisSHSRVIdx = (skyVis == SkyVisMode::SH9) ? GetSkyVisSHSRVIndex() : 0u;
        p.skyVisMode     = (uint32_t)skyVis;
    }

    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BakedGIUpdateParams) / 4, &p, 0);
    cmdList->SetPipelineState(m_UpdatePSO.Get());
    cmdList->Dispatch((m_ProbeCount + 63) / 64, 1, 1);

    D3D12_RESOURCE_BARRIER barriers[3] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_LitProbes.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_LitProbesSky.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_LitProbesSun.resource.Get()),
    };
    cmdList->ResourceBarrier(3, barriers);
}
