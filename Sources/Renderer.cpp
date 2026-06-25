#include "pch.h"

#include "Renderer.h"
#include "Model.h"
#include "Utility.h"
#include <dxcapi.h>
#include <array>
#include <cassert>
#include <cstring>
#include "Rtxdi/GI/ReSTIRGIParameters.h"
#include "Rtxdi/RtxdiUtils.h"
#include "NRI.h"
#include "Extensions/NRIHelper.h"
#include "Extensions/NRIWrapperD3D12.h"
#include "NRD.h"
#include "NRDIntegration.hpp"

namespace
{
    constexpr nrd::Identifier kNrdRelaxDiffuseSpecularIdentifier = 1u;

    nri::AccessLayoutStage ToNriAccessLayoutStage(D3D12_RESOURCE_STATES state)
    {
        if ((state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) != 0)
            return {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};

        if ((state & (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_GENERIC_READ)) != 0)
            return {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};

        return {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
    }

    nrd::Resource MakeNrdResource(GPUTexture& texture)
    {
        nrd::Resource resource = {};
        resource.d3d12.resource = texture.resource.Get();
        resource.d3d12.format = static_cast<DXGIFormat>(texture.format);
        resource.userArg = &texture;
        resource.state = ToNriAccessLayoutStage(texture.state);
        return resource;
    }

    void CopyMatrixToNrd(float dst[16], DirectX::FXMMATRIX matrix)
    {
        DirectX::XMFLOAT4X4 tmp;
        DirectX::XMStoreFloat4x4(&tmp, matrix);
        std::memcpy(dst, &tmp, sizeof(tmp));
    }
}

Renderer::Renderer()
    : m_FrameIndex(0)
    , m_FenceValue(0)
    , m_FenceEvent(nullptr)
{
}

Renderer::~Renderer()
{
    Shutdown();
}

bool Renderer::InitializeNrd()
{
    if (m_NrdInitialized)
        return true;

    nri::QueueFamilyD3D12Desc queueFamily = {};
    ID3D12CommandQueue* queue = m_CommandQueue.Get();
    queueFamily.d3d12Queues = &queue;
    queueFamily.queueNum = 1;
    queueFamily.queueType = nri::QueueType::GRAPHICS;

    nri::DeviceCreationD3D12Desc deviceDesc = {};
    deviceDesc.d3d12Device = m_Device.Get();
    deviceDesc.queueFamilies = &queueFamily;
    deviceDesc.queueFamilyNum = 1;
    deviceDesc.disableD3D12EnhancedBarriers = true;

    const nrd::DenoiserDesc denoisers[] = {
        { kNrdRelaxDiffuseSpecularIdentifier, nrd::Denoiser::RELAX_DIFFUSE_SPECULAR }
    };

    nrd::InstanceCreationDesc instanceDesc = {};
    instanceDesc.denoisers = denoisers;
    instanceDesc.denoisersNum = _countof(denoisers);

    nrd::IntegrationCreationDesc integrationDesc = {};
    strcpy_s(integrationDesc.name, "TortureRedNRD");
    integrationDesc.resourceWidth = static_cast<uint16_t>(m_InternalWidth);
    integrationDesc.resourceHeight = static_cast<uint16_t>(m_InternalHeight);
    integrationDesc.queuedFrameNum = 1;
    integrationDesc.enableWholeLifetimeDescriptorCaching = false;
    integrationDesc.autoWaitForIdle = true;

    m_NrdIntegration = std::make_unique<nrd::Integration>();
    const nrd::Result result = m_NrdIntegration->RecreateD3D12(integrationDesc, instanceDesc, deviceDesc);
    if (result != nrd::Result::SUCCESS)
    {
        std::cerr << "Failed to initialize NRD integration" << std::endl;
        m_NrdIntegration.reset();
        return false;
    }

    m_NrdInitialized = true;
    return true;
}

void Renderer::ShutdownNrd()
{
    if (m_NrdIntegration)
    {
        m_NrdIntegration->Destroy();
        m_NrdIntegration.reset();
    }

    m_NrdInitialized = false;
}

void Renderer::CreateRasterIndirectGIResources()
{
#if 0
    // -----------------------------------------------------------------------
    // Spatial irradiance cache buffers
    // -----------------------------------------------------------------------
    constexpr UINT MAX_ENTRIES  = 32768;
    constexpr UINT TOTAL_CELLS  = 262144;    // 32^3 * 8 cascades

    // Helper lambdas for creating raw (ByteAddressBuffer) and structured UAV buffers

    // Raw (RWByteAddressBuffer) — CreateBuffer default UAV is already RAW
    CreateBuffer(m_IrCacheMetaBuf,     16ULL,               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);
    CreateBuffer(m_IrCacheGridMetaBuf, TOTAL_CELLS * 4ULL,  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);
    CreateBuffer(m_IrCacheLifeBuf,     MAX_ENTRIES * 4ULL,  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);
    CreateBuffer(m_IrCacheTraceArgsBuf,12ULL,               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);

    // Structured (RWStructuredBuffer<T>) — CreateStructuredBuffer now writes correct UAV
    CreateStructuredBuffer(m_IrCachePoolBuf,       sizeof(UINT),      MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_IrCacheEntryCellBuf,  sizeof(UINT),      MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_IrCacheIrradianceBuf, sizeof(Reservoir), MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_IrCacheIndirectionBuf,sizeof(UINT),      MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Position voting buffers
    CreateStructuredBuffer(m_IrCachePosBuf,        sizeof(float) * 4, MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_IrCacheRepropBuf,      sizeof(float) * 4, MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateBuffer(m_IrCacheRepropCountBuf,           MAX_ENTRIES * 4ULL, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);

    // Fill the IrCacheBindlessIndices struct (all UAV indices)
    m_IrCacheIndices.MetaBufIdx             = (UINT)m_IrCacheMetaBuf.uavIndex;
    m_IrCacheIndices.PoolBufIdx             = (UINT)m_IrCachePoolBuf.uavIndex;
    m_IrCacheIndices.GridMetaBufIdx         = (UINT)m_IrCacheGridMetaBuf.uavIndex;
    m_IrCacheIndices.EntryCellBufIdx        = (UINT)m_IrCacheEntryCellBuf.uavIndex;
    m_IrCacheIndices.IrradianceBufIdx       = (UINT)m_IrCacheIrradianceBuf.uavIndex;
    m_IrCacheIndices.LifeBufIdx             = (UINT)m_IrCacheLifeBuf.uavIndex;
    m_IrCacheIndices.IndirectionBufIdx      = (UINT)m_IrCacheIndirectionBuf.uavIndex;
    m_IrCacheIndices.TraceArgsBufIdx        = (UINT)m_IrCacheTraceArgsBuf.uavIndex;
    m_IrCacheIndices.PosBufIdx             = (UINT)m_IrCachePosBuf.uavIndex;
    m_IrCacheIndices.RepropBufIdx          = (UINT)m_IrCacheRepropBuf.uavIndex;
    m_IrCacheIndices.ReproposalCountBufIdx = (UINT)m_IrCacheRepropCountBuf.uavIndex;
#endif

    // ------- SHaRC buffers (~160 MB total) -------
    CreateStructuredBuffer(m_SharcHashEntriesBuf,  8,  SHARC_HASH_ENTRIES_NUM, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_SharcAccumulationBuf, 16, SHARC_HASH_ENTRIES_NUM, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_SharcResolvedBuf,     16, SHARC_HASH_ENTRIES_NUM, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_SharcIndices.HashEntriesBufIdx  = (UINT)m_SharcHashEntriesBuf.uavIndex;
    m_SharcIndices.AccumulationBufIdx = (UINT)m_SharcAccumulationBuf.uavIndex;
    m_SharcIndices.ResolvedBufIdx     = (UINT)m_SharcResolvedBuf.uavIndex;

    // ------- Split Diffuse / Specular ReSTIR buffers -------
    for (int i = 0; i < 2; ++i) {
        CreateStructuredBuffer(m_DiffuseReservoirBuffer[i], sizeof(Reservoir), m_InternalWidth * m_InternalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CreateStructuredBuffer(m_SpecularReservoirBuffer[i], sizeof(Reservoir), m_InternalWidth * m_InternalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    CreateStructuredBuffer(m_DiffuseReservoirIntermediate, sizeof(Reservoir), m_InternalWidth * m_InternalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_SpecularReservoirIntermediate, sizeof(Reservoir), m_InternalWidth * m_InternalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_DiffuseCandidateBuffer, sizeof(DiffuseCandidate), m_InternalWidth * m_InternalHeight, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    
    CreateTexture(m_RasterHdrOutputTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RENDER_TARGET);
    CreateTexture(m_NrdMotionVectorsTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdNormalRoughnessTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R10G10B10A2_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdViewZTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdDenoisedDiffuseTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdDenoisedSpecularTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdValidationTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Universal interchange textures: SSO writes, NrdPackNoise+Lighting read
    CreateTexture(m_FinalDiffuseTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_FinalSpecularTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    InitializeNrd();
}

void Renderer::CreateRasterIndirectGIPipelines()
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
    computeDesc.pRootSignature = m_RootSignature.Get();

    auto CompileAndCreate = [&](const char* file, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso)
    {
        auto cs = GraphicsHelper::CompileShader(file, "main", "cs_6_6");
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&pso));
        }
    };

#if 0
    CompileAndCreate("Shaders/IrCache_Pool_Init.hlsl",      m_IrCachePoolInitPSO);
    CompileAndCreate("Shaders/IrCache_Prepare_Age.hlsl",    m_IrCachePrepareAgePSO);
    CompileAndCreate("Shaders/IrCache_Age.hlsl",            m_IrCacheAgePSO);
    CompileAndCreate("Shaders/IrCache_Prepare_Trace.hlsl",  m_IrCachePrepareTracePSO);
    CompileAndCreate("Shaders/IrCache_Update.hlsl",         m_IrCacheUpdatePSO);
#endif

    // ------- SHaRC PSOs -------
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/SHaRC_Update.hlsl", "main", "cs_6_6",
            {{L"SHARC_UPDATE", L"1"}, {L"SHARC_PROPAGATION_DEPTH", L"4"}, {L"SHARC_UPDATE_DOWNSCALE", L"5"}});
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SharcUpdatePSO));
        }
    }
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/SHaRC_Resolve.hlsl", "main", "cs_6_6", {});
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SharcResolvePSO));
        }
    }
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/SHaRC_Debug.hlsl", "main", "cs_6_6", {});
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SharcDebugPSO));
        }
    }

    auto nrdGuidesCS      = GraphicsHelper::CompileShader("Shaders/NrdPrepareGuides.hlsl",              "main", "cs_6_6");
    auto nrdCompositeCS   = GraphicsHelper::CompileShader("Shaders/NrdCompositeIndirect.hlsl",          "main", "cs_6_6");
    auto giResolveCS      = GraphicsHelper::CompileShader("Shaders/RestirGI_ResolveIntermediates.hlsl", "main", "cs_6_6");
    auto nrdStoreSSO_CS   = GraphicsHelper::CompileShader("Shaders/NrdStoreShadingOutput.hlsl",         "main", "cs_6_6");
    auto nrdPackNoiseCS   = GraphicsHelper::CompileShader("Shaders/NrdPackNoise.hlsl",                  "main", "cs_6_6");

    // ------- Split Diffuse / Specular PSOs -------
    auto diffuseTemporalCS  = GraphicsHelper::CompileShader("Shaders/RestirGI_Diffuse_Temporal.hlsl",  "main", "cs_6_6");
    auto specularTemporalCS = GraphicsHelper::CompileShader("Shaders/RestirGI_Specular_Temporal.hlsl", "main", "cs_6_6");
    auto diffuseSpatialCS   = GraphicsHelper::CompileShader("Shaders/RestirGI_Diffuse_Spatial.hlsl",   "main", "cs_6_6");
    auto specularSpatialCS  = GraphicsHelper::CompileShader("Shaders/RestirGI_Specular_Spatial.hlsl",  "main", "cs_6_6");

    computeDesc.CS = { nrdGuidesCS.data(), nrdGuidesCS.size() };
    m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdPrepareGuidesPSO));

    computeDesc.CS = { nrdCompositeCS.data(), nrdCompositeCS.size() };
    m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdCompositePSO));

    if (!giResolveCS.empty()) {
        computeDesc.CS = { giResolveCS.data(), giResolveCS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_GIResolveIntermediatesPSO));
    }
    if (!nrdStoreSSO_CS.empty()) {
        computeDesc.CS = { nrdStoreSSO_CS.data(), nrdStoreSSO_CS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdStoreShadingOutputPSO));
    }
    if (!nrdPackNoiseCS.empty()) {
        computeDesc.CS = { nrdPackNoiseCS.data(), nrdPackNoiseCS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdPackNoisePSO));
    }

    // ------- Split Diffuse / Specular PSO creation -------
    if (!diffuseTemporalCS.empty()) {
        computeDesc.CS = { diffuseTemporalCS.data(), diffuseTemporalCS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_DiffuseTemporalPSO));
    }
    if (!specularTemporalCS.empty()) {
        computeDesc.CS = { specularTemporalCS.data(), specularTemporalCS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SpecularTemporalPSO));
    }
    if (!diffuseSpatialCS.empty()) {
        computeDesc.CS = { diffuseSpatialCS.data(), diffuseSpatialCS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_DiffuseSpatialPSO));
    }
    if (!specularSpatialCS.empty()) {
        computeDesc.CS = { specularSpatialCS.data(), specularSpatialCS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_SpecularSpatialPSO));
    }
    // Seed file timestamps for hot-reload after all PSOs are initially created.
    CreateRestirDIPipelines();
    SetupShaderTimestamps();

#if 0
    // Command signature for ExecuteIndirect dispatch (used by IrCache_Update indirect pass)
    D3D12_INDIRECT_ARGUMENT_DESC dispatchArg = {};
    dispatchArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    D3D12_COMMAND_SIGNATURE_DESC csSigDesc = {};
    csSigDesc.ByteStride       = sizeof(D3D12_DISPATCH_ARGUMENTS);
    csSigDesc.NumArgumentDescs = 1;
    csSigDesc.pArgumentDescs   = &dispatchArg;
    m_Device->CreateCommandSignature(&csSigDesc, nullptr, IID_PPV_ARGS(&m_DispatchCommandSignature));

    // Probe sphere debug PSO (graphics, depth-read-only, alpha blend)
    {
        auto vs = GraphicsHelper::CompileShader("Shaders/IrCache_DebugSpheres.hlsl", "VSMain", "vs_6_6");
        auto ps = GraphicsHelper::CompileShader("Shaders/IrCache_DebugSpheres.hlsl", "PSMain", "ps_6_6");
        if (!vs.empty() && !ps.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature                          = m_RootSignature.Get();
            desc.VS                                      = { vs.data(), vs.size() };
            desc.PS                                      = { ps.data(), ps.size() };
            desc.RasterizerState                         = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            desc.RasterizerState.FrontCounterClockwise   = TRUE;
            desc.RasterizerState.CullMode                = D3D12_CULL_MODE_BACK;
            // Depth: test against scene geometry, never write
            desc.DepthStencilState                       = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
            desc.DepthStencilState.DepthEnable           = TRUE;
            desc.DepthStencilState.DepthWriteMask        = D3D12_DEPTH_WRITE_MASK_ZERO;
            desc.DepthStencilState.DepthFunc             = D3D12_COMPARISON_FUNC_LESS_EQUAL;
            // Alpha blend
            desc.BlendState                              = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            desc.BlendState.RenderTarget[0].BlendEnable  = TRUE;
            desc.BlendState.RenderTarget[0].SrcBlend     = D3D12_BLEND_SRC_ALPHA;
            desc.BlendState.RenderTarget[0].DestBlend    = D3D12_BLEND_INV_SRC_ALPHA;
            desc.BlendState.RenderTarget[0].BlendOp      = D3D12_BLEND_OP_ADD;
            desc.BlendState.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
            desc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
            desc.BlendState.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            desc.SampleMask                              = UINT_MAX;
            desc.PrimitiveTopologyType                   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets                        = 1;
            desc.RTVFormats[0]                           = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.DSVFormat                               = DXGI_FORMAT_D32_FLOAT;
            desc.SampleDesc.Count                        = 1;
            m_Device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_ProbeSphereDebugPSO));
        }
    }
#endif
}

// =============================================================================
// ReSTIR DI — Direct Illumination via spatiotemporal reservoir resampling
// =============================================================================

void Renderer::CreateRestirDIResources()
{
    const UINT pixelCount = m_InternalWidth * m_InternalHeight;
    for (int i = 0; i < 2; ++i)
        CreateStructuredBuffer(m_DIReservoirBuffer[i], sizeof(DIRreservoir), pixelCount,
                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_DIReservoirIntermediate, sizeof(DIRreservoir), pixelCount,
                           D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Split DI intermediates for SSO bridge path
    CreateTexture(m_DIDiffuseIntermediate, m_InternalWidth, m_InternalHeight,
                  DXGI_FORMAT_R16G16B16A16_FLOAT,
                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_DISpecularIntermediate, m_InternalWidth, m_InternalHeight,
                  DXGI_FORMAT_R16G16B16A16_FLOAT,
                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void Renderer::CreateRestirDIPipelines()
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = m_RootSignature.Get();

    auto CompileAndCreate = [&](const char* file, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso)
    {
        auto cs = GraphicsHelper::CompileShader(file, "main", "cs_6_6");
        if (!cs.empty())
        {
            desc.CS = { cs.data(), cs.size() };
            m_Device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
        }
    };

    CompileAndCreate("Shaders/RestirDI_Temporal.hlsl",        m_RestirDITemporalPSO);
    CompileAndCreate("Shaders/RestirDI_Spatial.hlsl",         m_RestirDISpatialPSO);
    CompileAndCreate("Shaders/RestirDI_SplitShade.hlsl",      m_RestirDISplitShadePSO);
}

void Renderer::DispatchRestirDI(class Model* model, const FrameConstants& frame)
{
    if (!frame.enableRestirDI) return;

    const int curr = m_CurrentDIReservoirIndex;
    const int prev = 1 - curr;

    // Ensure all DI resources are in UAV state
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DIReservoirBuffer[0],    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DIReservoirBuffer[1],    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DIReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DIDiffuseIntermediate,   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DISpecularIntermediate,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_CommandList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());

    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    m_CommandList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));
    m_CommandList->SetComputeRootShaderResourceView(4, m_TLAS.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(10, m_LightsBuffer.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(11, m_LightLUTBuffer.gpuAddress);

    const UINT W = m_InternalWidth, H = m_InternalHeight;
    const UINT gx = (W + 7) / 8, gy = (H + 7) / 8;

    BindlessIndices indices = {};

    // --- Pass 1: Combined Initial Sampling + Temporal Resampling ---
    // InputIdx0 = DIRreservoirBuffer[prev] (previous-frame temporal output)
    // OutputIdx0 = DIRreservoirBuffer[curr]
    indices = {};
    indices.InputIdx0  = m_DIReservoirBuffer[prev].srvIndex;
    indices.OutputIdx0 = m_DIReservoirBuffer[curr].uavIndex;
    indices.OutputIdx1 = m_RestirDebugHeatmap.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->SetPipelineState(m_RestirDITemporalPSO.Get());
    m_CommandList->Dispatch(gx, gy, 1);

    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(m_DIReservoirBuffer[curr].resource.Get());
        m_CommandList->ResourceBarrier(1, &b);
    }

    // --- Pass 2: Spatial Resampling ---
    // InputIdx0 = DIRreservoirBuffer[curr], OutputIdx0 = DIRreservoirIntermediate
    indices = {};
    indices.InputIdx0  = m_DIReservoirBuffer[curr].srvIndex;
    indices.OutputIdx0 = m_DIReservoirIntermediate.uavIndex;
    indices.OutputIdx1 = m_RestirDebugHeatmap.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->SetPipelineState(m_RestirDISpatialPSO.Get());
    m_CommandList->Dispatch(gx, gy, 1);

    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(m_DIReservoirIntermediate.resource.Get());
        m_CommandList->ResourceBarrier(1, &b);
    }

    // --- Pass 3: Split Shade — per-lobe NRD-normalized output ---
    // InputIdx0 = DIRreservoirIntermediate, OutputIdx0 = DIDiffuseIntermediate, OutputIdx1 = DISpecularIntermediate
    if (m_RestirDISplitShadePSO)
    {
        indices = {};
        indices.InputIdx0  = m_DIReservoirIntermediate.srvIndex;
        indices.OutputIdx0 = m_DIDiffuseIntermediate.uavIndex;
        indices.OutputIdx1 = m_DISpecularIntermediate.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        m_CommandList->SetPipelineState(m_RestirDISplitShadePSO.Get());
        m_CommandList->Dispatch(gx, gy, 1);

        D3D12_RESOURCE_BARRIER splitBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_DIDiffuseIntermediate.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_DISpecularIntermediate.resource.Get()),
        };
        m_CommandList->ResourceBarrier(_countof(splitBarriers), splitBarriers);
    }

    // --- Pass 3b: StoreShadingOutput Call 1 (DI base) ---
    // Writes DI intermediates into FinalDiffuseTex / FinalSpecularTex (overwrite).
    // Always dispatched when DI is active (regardless of NRD state).
    if (m_NrdStoreShadingOutputPSO && m_RestirDISplitShadePSO)
    {
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DIDiffuseIntermediate,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DISpecularIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        indices = {};
        indices.InputIdx0  = m_DIDiffuseIntermediate.srvIndex;
        indices.InputIdx1  = m_DISpecularIntermediate.srvIndex;
        indices.OutputIdx0 = m_FinalDiffuseTex.uavIndex;
        indices.OutputIdx1 = m_FinalSpecularTex.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        const UINT isFirstPass = 1u;
        m_CommandList->SetComputeRoot32BitConstants(13, 1, &isFirstPass, 0);
        m_CommandList->SetPipelineState(m_NrdStoreShadingOutputPSO.Get());
        m_CommandList->Dispatch(gx, gy, 1);

        D3D12_RESOURCE_BARRIER ssoBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_FinalDiffuseTex.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_FinalSpecularTex.resource.Get()),
        };
        m_CommandList->ResourceBarrier(_countof(ssoBarriers), ssoBarriers);
    }

    m_CurrentDIReservoirIndex = prev; // Swap for next frame

    // When GI is disabled but NRD is enabled, trigger the NRD denoise pass here.
    // (When GI is enabled, NRDDenoise is called from DispatchRestirGI.)
    if (frame.enableNrdRelax != 0u && !frame.enableRasterIndirectGI)
    {
        NRDDenoise(frame);
        // m_NrdWasActiveLastFrame is set inside NRDDenoise on success.
    }
    else if (frame.enableNrdRelax == 0u && !frame.enableRasterIndirectGI)
    {
        // NRD disabled and GI disabled: transition Final* to SRV for Lighting.hlsl
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
}


// =============================================================================
// Shader Hot-Reload Implementation
// =============================================================================

// ---------------------------------------------------------------------------
// SetupShaderTimestamps
// Scans Sources/Shaders/ recursively and caches the last-write-time of every
// .hlsl file.  Called once after all PSOs are first created.
// ---------------------------------------------------------------------------
void Renderer::SetupShaderTimestamps()
{
    namespace fs = std::filesystem;
    m_ShaderTimestamps.clear();

    fs::path shaderDir = fs::path(SHADER_SOURCE_DIR);

    std::error_code ec;
    for (const auto& dirEntry : fs::recursive_directory_iterator(shaderDir, ec))
    {
        if (!ec && dirEntry.is_regular_file() && dirEntry.path().extension() == ".hlsl")
        {
            std::error_code ec2;
            auto canonical = fs::weakly_canonical(dirEntry.path(), ec2).string();
            if (!ec2)
            {
                std::error_code ec3;
                auto t = fs::last_write_time(dirEntry.path(), ec3);
                if (!ec3) m_ShaderTimestamps[canonical] = t;
            }
        }
    }
    if (ec) {
        std::cout << "[HotReload] Error scanning " << shaderDir.string()
                  << ": " << ec.message() << std::endl;
    }
    std::cout << "[HotReload] Watching " << m_ShaderTimestamps.size()
              << " shader files under " << shaderDir.string() << std::endl;
}

// ---------------------------------------------------------------------------
// CheckAndReloadShaders
// Call once per frame.  If any watched .hlsl changed, GPU-syncs once then
// rebuilds ALL PSOs (graphics + compute).  This handles include changes too.
// ---------------------------------------------------------------------------
void Renderer::CheckAndReloadShaders()
{
    namespace fs = std::filesystem;

    bool anyChanged = false;
    for (const auto& [path, prevTime] : m_ShaderTimestamps)
    {
        std::error_code ec;
        auto curTime = fs::last_write_time(path, ec);
        if (!ec && curTime != prevTime)
        {
            anyChanged = true;
            break;
        }
    }

    if (!anyChanged) return;

    WaitForPreviousFrame();
    GraphicsHelper::InvalidateShaderCache();

    std::cout << "[HotReload] Shader change detected - rebuilding all PSOs..." << std::endl;

    // CreatePipelineState rebuilds all graphics PSOs; it also calls
    // CreateRayTracingPipeline() internally when RT is supported.
    CreatePipelineState();

    // CreateRasterIndirectGIPipelines rebuilds all compute GI/SHaRC/ReSTIR PSOs
    // and calls SetupShaderTimestamps() at the end to refresh the timestamp map.
    CreateRasterIndirectGIPipelines();

    // Rebuild TAA PSOs
    CreateTaaPipelines();

    std::cout << "[HotReload] Done." << std::endl;
}


bool Renderer::Initialize(HWND hwnd)
{
    UINT dxgiFactoryFlags = 0;

    // Enable the debug layer (requires the Graphics Tools "optional feature").
    {
        Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }

    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    CHECK_HR(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2 failed");

    // Create device
    Microsoft::WRL::ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    CHECK_HR(D3D12CreateDevice(
        hardwareAdapter.Get(),
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&m_Device)
    ), "D3D12CreateDevice failed");

    // Check for Ray Tracing support
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    if (SUCCEEDED(m_Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5))))
    {
        m_RayTracingSupported = (options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
    }
    std::cout << "Ray Tracing Supported: " << (m_RayTracingSupported ? "Yes" : "No") << std::endl;

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    CHECK_HR(m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_CommandQueue)), "CreateCommandQueue failed");

    // Create swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = 2;
    swapChainDesc.Width = WINDOW_WIDTH;
    swapChainDesc.Height = WINDOW_HEIGHT;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain;
    CHECK_HR(factory->CreateSwapChainForHwnd(
        m_CommandQueue.Get(),
        hwnd,
        &swapChainDesc,
        nullptr,
        nullptr,
        &swapChain
    ), "CreateSwapChainForHwnd failed");

    CHECK_HR(swapChain.As(&m_SwapChain), "SwapChain As failed");

    m_FrameIndex = m_SwapChain->GetCurrentBackBufferIndex();

    GraphicsHelper::Initialize(m_Device.Get());

    // Create render target view for each frame
    for (UINT n = 0; n < 2; n++)
    {
        CHECK_HR(m_SwapChain->GetBuffer(n, IID_PPV_ARGS(&m_RenderTargets[n])), "GetBuffer failed");
        m_Device->CreateRenderTargetView(m_RenderTargets[n].Get(), nullptr, GraphicsHelper::GetRTVCPUHandle(GraphicsHelper::AllocateRTV()));
    }

    // Create constant buffers
    if (!CreateBuffer(m_FrameCB, (sizeof(FrameConstants) + 255) & ~255, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
    {
        std::cerr << "Failed to create frame constant buffer" << std::endl;
        return false;
    }

    // Create GBuffer — initially at WINDOW_WIDTH x WINDOW_HEIGHT.
    // Will be recreated at internal resolution by CreateInternalResolutionResources().
    CreateGBuffer(WINDOW_WIDTH, WINDOW_HEIGHT);

    // Create Shadow Map
    const UINT shadowMapSize = 2048;
    if (!CreateTexture(m_ShadowMap, shadowMapSize, shadowMapSize, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr))
    {
        std::cerr << "Failed to create shadow map texture" << std::endl;
        return false;
    }

    // Create Path Tracer Output
    if (m_RayTracingSupported)
    {
        if (!CreateTexture(m_AccumulationBuffer, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr))
        {
            std::cerr << "Failed to create accumulation buffer" << std::endl;
            return false;
        }

        if (!CreateTexture(m_PathTracerOutput, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr))
        {
            std::cerr << "Failed to create path tracer output texture" << std::endl;
            return false;
        }

        if (!CreateTexture(m_PathTracerPresentOutput, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr))
        {
            std::cerr << "Failed to create path tracer present texture" << std::endl;
            return false;
        }

        if (!CreateTexture(m_RestirDebugHeatmap, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr))
        {
            std::cerr << "Failed to create ReSTIR debug heatmap texture" << std::endl;
            return false;
        }

        // Create ReSTIR Reservoirs — initially at WINDOW_WIDTH x WINDOW_HEIGHT.
        // Will be recreated at internal resolution by CreateInternalResolutionResources().
        for (int i = 0; i < 2; ++i)
        {
            if (!CreateStructuredBuffer(m_ReservoirBuffer[i], sizeof(Reservoir), WINDOW_WIDTH * WINDOW_HEIGHT, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
            {
                std::cerr << "Failed to create ReSTIR reservoir buffer" << std::endl;
                return false;
            }
        }

        if (!CreateStructuredBuffer(m_ReservoirIntermediate, sizeof(Reservoir), WINDOW_WIDTH * WINDOW_HEIGHT, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
        {
            std::cerr << "Failed to create ReSTIR intermediate reservoir buffer" << std::endl;
            return false;
        }

        // Create Path Visualization Line Buffer (small: MAX_PATH_VIZ_LINES * sizeof(PathVizLine))
        if (!CreateStructuredBuffer(m_PathVizLineBuffer, sizeof(PathVizLine), MAX_PATH_VIZ_LINES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
        {
            std::cerr << "Failed to create PathViz line buffer" << std::endl;
            return false;
        }

        // Create RTXDI Reservoirs
        // See RtxdiUtils.cpp: CalculateReservoirBufferParameters
        uint32_t renderWidthBlocks = (m_InternalWidth + 15) / 16;
        uint32_t renderHeightBlocks = (m_InternalHeight + 15) / 16;
        uint32_t reservoirArrayPitch = renderWidthBlocks * 256 * renderHeightBlocks;

        for (int i = 0; i < 2; ++i)
        {
            if (!CreateStructuredBuffer(m_RtxdiReservoirBuffer[i], sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
            {
                std::cerr << "Failed to create RTXDI reservoir buffer " << i << std::endl;
                return false;
            }
        }

        if (!CreateStructuredBuffer(m_RtxdiReservoirIntermediate, sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
        {
            std::cerr << "Failed to create RTXDI intermediate reservoir buffer" << std::endl;
            return false;
        }

        // Create Neighbor Offsets Buffer
        const uint32_t neighborOffsetCount = 8192;
        if (!CreateBuffer(m_RtxdiNeighborOffsetsBuffer, neighborOffsetCount * 2, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false))
        {
            std::cerr << "Failed to create RTXDI neighbor offset buffer" << std::endl;
            return false;
        }
        
        std::vector<uint8_t> offsets(neighborOffsetCount * 2);
        rtxdi::FillNeighborOffsetBuffer(offsets.data(), neighborOffsetCount);
        memcpy(m_RtxdiNeighborOffsetsBuffer.cpuPtr, offsets.data(), offsets.size());

        // Create Typed SRV for Neighbor Offsets
        m_RtxdiNeighborOffsetsBuffer.srvIndex = GraphicsHelper::AllocateSRV();
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R8G8_SNORM;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = neighborOffsetCount;
        srvDesc.Buffer.StructureByteStride = 0;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

        m_Device->CreateShaderResourceView(m_RtxdiNeighborOffsetsBuffer.resource.Get(), &srvDesc, GraphicsHelper::GetSRVCPUHandle(m_RtxdiNeighborOffsetsBuffer.srvIndex));
    }

    // Create command allocator
    CHECK_HR(m_Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_CommandAllocator)), "CreateCommandAllocator failed");

    // Create command list
    CHECK_HR(m_Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_CommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_CommandList)), "CreateCommandList failed");
    CHECK_HR(m_CommandList->Close(), "CommandList Close failed");

    // Create fence
    CHECK_HR(m_Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_Fence)), "CreateFence failed");
    m_FenceValue = 1;

    m_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (m_FenceEvent == nullptr)
    {
        std::cerr << "CreateEvent failed" << std::endl;
        return false;
    }

    // Create root signature and pipeline state
    CreateRootSignature();
    CreatePipelineState();

    // Check for SM 6.8 support
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_8 };
    if (FAILED(m_Device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))))
    {
        shaderModel.HighestShaderModel = (D3D_SHADER_MODEL)0; // Unknown
    }
    printf("Max supported shader model: %u.%u\n", (shaderModel.HighestShaderModel >> 4) & 0xF, shaderModel.HighestShaderModel & 0xF);

    if (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_8)
    {
        printf("Shader Model 6.8 is NOT supported. Please ensure Agility SDK is loaded correctly.\n");
    }
    else
    {
        printf("Shader Model 6.8 is confirmed supported!\n");
    }

    std::cout << "Renderer initialized successfully!" << std::endl;
    return true;
}

void Renderer::Shutdown()
{
    // Wait for the GPU to be done with all resources
    WaitForPreviousFrame();

    ShutdownNrd();

    GraphicsHelper::Shutdown();

    // Cleanup constant buffers
    if (m_FrameCB.resource && m_FrameCB.cpuPtr)
    {
        m_FrameCB.resource->Unmap(0, nullptr);
        m_FrameCB.cpuPtr = nullptr;
    }

    if (m_FenceEvent)
    {
        CloseHandle(m_FenceEvent);
        m_FenceEvent = nullptr;
    }

    m_BlasPool.clear();
}

void Renderer::Resize(uint32_t width, uint32_t height)
{
    // Implementation for window resize - would need to recreate swap chain and depth buffer
    // For now, just a placeholder
}

void Renderer::BeginFrame()
{
    // Record commands
    CHECK_HR(m_CommandAllocator->Reset(), "CommandAllocator Reset failed");
    CHECK_HR(m_CommandList->Reset(m_CommandAllocator.Get(), nullptr), "CommandList Reset failed");

    // Set descriptor heaps
    ID3D12DescriptorHeap* heaps[] = { GraphicsHelper::GetSRVHeap() };
    m_CommandList->SetDescriptorHeaps(_countof(heaps), heaps);

    // Set necessary state
    m_CommandList->SetGraphicsRootSignature(m_RootSignature.Get());

    // Bind the global descriptor table (bindless)
    m_CommandList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    // Set Frame constant buffer (viewProj)
    m_CommandList->SetGraphicsRootConstantBufferView(0, m_FrameCB.gpuAddress);

    // Bind TLAS for ray-traced shadows in pixel shader
    m_CommandList->SetGraphicsRootShaderResourceView(4, m_TLAS.gpuAddress);

    // Bind Lights Buffer (t0, space2) - root parameter 10
    m_CommandList->SetGraphicsRootShaderResourceView(10, m_LightsBuffer.gpuAddress);
    m_CommandList->SetGraphicsRootShaderResourceView(11, m_LightLUTBuffer.gpuAddress); // Light LUT (t1, space2)

    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(m_InternalWidth), static_cast<float>(m_InternalHeight));
    D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, m_InternalWidth, m_InternalHeight);
    m_CommandList->RSSetViewports(1, &viewport);
    m_CommandList->RSSetScissorRects(1, &scissorRect);

    m_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

void Renderer::EndFrame()
{
    // Transition back buffer to present state
    GraphicsHelper::TransitionResource(m_CommandList.Get(), (m_RenderTargets[m_FrameIndex].Get()), m_BackBufferStates[m_FrameIndex], D3D12_RESOURCE_STATE_PRESENT);

    CHECK_HR(m_CommandList->Close(), "CommandList Close failed");

    // Execute the command list
    ID3D12CommandList* ppCommandLists[] = { m_CommandList.Get() };
    m_CommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present the frame
    CHECK_HR(m_SwapChain->Present(1, 0), "Present failed");

    // Wait for the GPU to finish
    WaitForPreviousFrame();
}

void Renderer::Present()
{
    // This is now handled in EndFrame
}

void Renderer::ExecuteCommandList()
{
    CHECK_HR(m_CommandList->Close(), "CommandList Close failed");
    ID3D12CommandList* cmds[] = { m_CommandList.Get() };
    m_CommandQueue->ExecuteCommandLists(_countof(cmds), cmds);
    WaitForPreviousFrame();
}

D3D12_CPU_DESCRIPTOR_HANDLE Renderer::GetCurrentBackBufferRTV() const
{
    return GraphicsHelper::GetRTVCPUHandle(m_FrameIndex);
}

ID3D12Resource* Renderer::GetCurrentBackBuffer() const
{
    return m_RenderTargets[m_FrameIndex].Get();
}

void Renderer::CreateRootSignature()
{
    CD3DX12_DESCRIPTOR_RANGE srvRanges[2];
    srvRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4096, 0, 0); // t0 space0: Bindless textures

    CD3DX12_DESCRIPTOR_RANGE uavRange0;
    uavRange0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0); // u0 space0: Restir RTXDI ping pong buffer

    CD3DX12_DESCRIPTOR_RANGE uavRange1;
    uavRange1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0); // u1 space0: Restir RTXDI ping pong buffer

    CD3DX12_DESCRIPTOR_RANGE srvRangeRtxdiOffsets;
    srvRangeRtxdiOffsets.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5, 1); // t5 space1: RTXDI Neighbor Offsets

    CD3DX12_DESCRIPTOR_RANGE srvRangeSpace3;
    srvRangeSpace3.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 3); // t0 space3: ReSTIR GI SRVs

    CD3DX12_DESCRIPTOR_RANGE srvRangeSpace3_2;
    srvRangeSpace3_2.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 3); // t1 space3: IrCache Debug

    CD3DX12_ROOT_PARAMETER rootParameters[14];
    rootParameters[0].InitAsConstantBufferView(0); // b0: FrameConstants
    rootParameters[1].InitAsShaderResourceView(0, 1); // t0 space1: Material Data
    rootParameters[2].InitAsShaderResourceView(1, 1); // t1 space1: Draw Node Data
    rootParameters[3].InitAsDescriptorTable(1, &srvRanges[0]); // t0 space0: Bindless textures
    rootParameters[4].InitAsShaderResourceView(2, 1); // t2 space1: TLAS
    rootParameters[5].InitAsShaderResourceView(3, 1); // t3 space1: Indices
    rootParameters[6].InitAsShaderResourceView(4, 1); // t4 space1: Vertices
    rootParameters[7].InitAsDescriptorTable(1, &uavRange0); // u0 : Restir RTXDI ping pong buffer
    rootParameters[8].InitAsDescriptorTable(1, &uavRange1); // u1 : Restir RTXDI ping pong buffer
    rootParameters[9].InitAsDescriptorTable(1, &srvRangeRtxdiOffsets); // t5 space1
    rootParameters[10].InitAsShaderResourceView(0, 2); // t0 space2: Lights Buffer
    rootParameters[11].InitAsShaderResourceView(1, 2); // t1 space2: Light LUT Buffer
    rootParameters[12].InitAsConstants(sizeof(BindlessIndices) / 4, 1, 0); // b1: Bindless indices
    rootParameters[13].InitAsConstants(sizeof(IrCacheBindlessIndices) / 4, 2, 0); // b2: IrCache bindless indices

    CD3DX12_STATIC_SAMPLER_DESC samplers[2];
    samplers[0].Init(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);
    samplers[1].Init(1, D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR, 
        D3D12_TEXTURE_ADDRESS_MODE_BORDER, D3D12_TEXTURE_ADDRESS_MODE_BORDER, D3D12_TEXTURE_ADDRESS_MODE_BORDER);
    samplers[1].ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    samplers[1].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init(_countof(rootParameters), rootParameters, _countof(samplers), samplers, 
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (FAILED(hr))
    {
        if (error)
        {
            std::cerr << "D3D12SerializeRootSignature failed: " << (char*)error->GetBufferPointer() << std::endl;
        }
        CHECK_HR(hr, "D3D12SerializeRootSignature failed");
    }

    CHECK_HR(m_Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_RootSignature)), "CreateRootSignature failed");

    // Create command signature for ExecuteIndirect
    D3D12_INDIRECT_ARGUMENT_DESC drawArg = {};
    drawArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;

    D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc = {};
    commandSignatureDesc.ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
    commandSignatureDesc.NumArgumentDescs = 1;
    commandSignatureDesc.pArgumentDescs = &drawArg;

    CHECK_HR(m_Device->CreateCommandSignature(&commandSignatureDesc, nullptr, IID_PPV_ARGS(&m_CommandSignature)), "CreateCommandSignature failed");
}

void Renderer::DrawProbeSpheresDebug()
{
#if 0
    auto* cmdList = m_CommandList.Get();
    if (!m_ProbeSphereDebugPSO)
        return;
    cmdList->SetPipelineState(m_ProbeSphereDebugPSO.Get());
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    // 8 stacks × 8 slices × 6 verts/quad = 384 verts per probe
    // 32768 instances — VS culls unoccupied entries by emitting a degenerate clip position
    cmdList->DrawInstanced(384, 32768, 0, 0);
#endif
}

void Renderer::CreatePipelineState()
{
    auto GetDefaultPsoDesc = [&]() {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = m_RootSignature.Get();
        desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        desc.RasterizerState.FrontCounterClockwise = TRUE;
        desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        desc.SampleMask = UINT_MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.SampleDesc.Count = 1;
        return desc;
    };

    // 1. Depth Pre-Pass PSO
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.NumRenderTargets = 0;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_DepthPrePassPSO));
    }

    // 2. G-Buffer PSO
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Gbuffer.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Gbuffer.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.NumRenderTargets = 3;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        psoDesc.RTVFormats[2] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        
        // No depth write, using pre-pass
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_GBufferPSO));

        // Create a version of G-Buffer PSO that writes to depth (for when pre-pass is disabled)
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_GBufferWritePSO));
    }

    // 3. Lighting PSO (LDR — renders to R8G8B8A8_UNORM back buffer with tonemapping)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_LightingPSO));
    }

    // 3.1 Lighting HDR PSO (renders to R16G16B16A16_FLOAT for TAA input — no tonemapping in shader)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Lighting.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_LightingHdrPSO));
    }

    // 3.5 Debug PSO
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/DebugShadow.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/DebugShadow.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_DebugPSO));
    }

    // 4. Shadow PSO
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.RasterizerState.DepthBias = 1000;
        psoDesc.RasterizerState.SlopeScaledDepthBias = 1.5f;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.NumRenderTargets = 0;

        CHECK_HR(m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_ShadowPSO)), "CreateGraphicsPipelineState for Shadow PSO failed");
    }

    // 5. Transparent PSO (Forward)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/Forward.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/Forward.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        
        // Enable Alpha Blending
        D3D12_RENDER_TARGET_BLEND_DESC blendDesc = {};
        blendDesc.BlendEnable = TRUE;
        blendDesc.LogicOpEnable = FALSE;
        blendDesc.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        blendDesc.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        blendDesc.BlendOp = D3D12_BLEND_OP_ADD;
        blendDesc.SrcBlendAlpha = D3D12_BLEND_ONE;
        blendDesc.DestBlendAlpha = D3D12_BLEND_ZERO;
        blendDesc.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        blendDesc.LogicOp = D3D12_LOGIC_OP_NOOP;
        blendDesc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        psoDesc.BlendState.RenderTarget[0] = blendDesc;

        // Double sided 
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        
        // Read-only depth
        psoDesc.DepthStencilState.DepthEnable = TRUE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;

        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM; // Backbuffer format
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

        CHECK_HR(m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_TransparentPSO)), "CreateGraphicsPipelineState for Transparent PSO failed");
    }

    if (m_RayTracingSupported)
    {
        CreateRayTracingPipeline();
    }

    std::cout << "Pipeline states created successfully" << std::endl;
}

void Renderer::CreateRayTracingPipeline()
{
    // Load Path Tracer shader
    auto pathTracerCode = GraphicsHelper::CompileShader("Shaders/PathTracer.hlsl", "CSMain", "cs_6_6");
    if (!pathTracerCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { pathTracerCode.data(), pathTracerCode.size() };
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_PathTracerPSO)), "Failed to create Path Tracer Compute PSO");
    }

    auto pathTracerPresentCode = GraphicsHelper::CompileShader("Shaders/PathTracerPresent.hlsl", "CSMain", "cs_6_6");
    if (!pathTracerPresentCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { pathTracerPresentCode.data(), pathTracerPresentCode.size() };
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_PathTracerPresentPSO)), "Failed to create Path Tracer Present PSO");
    }

    // Load ReSTIR Multi-pass shaders
    auto restirTemporalCode = GraphicsHelper::CompileShader("Shaders/RestirGI_Temporal.hlsl", "CSMain", "cs_6_6");
    if (!restirTemporalCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { restirTemporalCode.data(), restirTemporalCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirTemporalPSO)), "Failed to create ReSTIR Temporal PSO");
    }

    auto restirSpatialCode = GraphicsHelper::CompileShader("Shaders/RestirGI_Spatial.hlsl", "CSMain", "cs_6_6");
    if (!restirSpatialCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { restirSpatialCode.data(), restirSpatialCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirSpatialPSO)), "Failed to create ReSTIR Spatial PSO");
    }

    auto restirResolveCode = GraphicsHelper::CompileShader("Shaders/RestirGI_Resolve.hlsl", "CSMain", "cs_6_6");
    if (!restirResolveCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { restirResolveCode.data(), restirResolveCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirResolvePSO)), "Failed to create ReSTIR Resolve PSO");
    }

    auto restirDebugCode = GraphicsHelper::CompileShader("Shaders/RestirGI_ReservoirDebug.hlsl", "main", "cs_6_6");
    if (!restirDebugCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { restirDebugCode.data(), restirDebugCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirReservoirDebugPSO)), "Failed to create ReSTIR Reservoir Debug PSO");
    }

    // RTXDI PSOs
    auto rtxdiTemporalCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Temporal.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiTemporalCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { rtxdiTemporalCode.data(), rtxdiTemporalCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirTemporalPSO)), "Failed to create RTXDI Temporal PSO");
    }

    auto rtxdiSpatialCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Spatial.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiSpatialCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { rtxdiSpatialCode.data(), rtxdiSpatialCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirSpatialPSO)), "Failed to create RTXDI Spatial PSO");
    }

    auto rtxdiResolveCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Resolve.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiResolveCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { rtxdiResolveCode.data(), rtxdiResolveCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirResolvePSO)), "Failed to create RTXDI Resolve PSO");
    }

    auto rtxdiDebugCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Debug.hlsl", "main", "cs_6_6");
    if (!rtxdiDebugCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { rtxdiDebugCode.data(), rtxdiDebugCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirReservoirDebugPSO)), "Failed to create RTXDI Reservoir Debug PSO");
    }

    // Path Visualization Lines PSO (graphics pipeline, line list, no depth)
    {
        auto vsCode = GraphicsHelper::CompileShader("Shaders/PathVizLines.hlsl", "VSMain", "vs_6_6");
        auto psCode = GraphicsHelper::CompileShader("Shaders/PathVizLines.hlsl", "PSMain", "ps_6_6");
        if (!vsCode.empty() && !psCode.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
            psoDesc.pRootSignature = m_RootSignature.Get();
            psoDesc.VS = { vsCode.data(), vsCode.size() };
            psoDesc.PS = { psCode.data(), psCode.size() };
            psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            psoDesc.NumRenderTargets = 1;
            psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
            psoDesc.SampleDesc.Count = 1;
            psoDesc.SampleMask = UINT_MAX;
            psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            psoDesc.DepthStencilState.DepthEnable = FALSE;
            psoDesc.DepthStencilState.StencilEnable = FALSE;
            CHECK_HR(m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PathVizLinePSO)), "Failed to create PathViz Lines PSO");
        }
    }
}

void Renderer::DispatchRays(Model* model, const FrameConstants& frame, const LightConstants& light)
{
    if (!model) return;

    const bool useCustomRestirHeatmap = frame.enableRestir && !frame.useRTXDI &&
        frame.restirReservoirDebugMode >= RESTIR_RESERVOIR_DEBUG_SOURCE_PDF;

    // Update constant buffers
    memcpy(m_FrameCB.cpuPtr, &frame, sizeof(FrameConstants));

    // Transition UAVs
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_AccumulationBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_PathTracerOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_PathTracerPresentOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_ReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_ReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_ReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RtxdiReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RtxdiReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RtxdiReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_PathVizLineBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (useCustomRestirHeatmap)
    {
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RestirDebugHeatmap, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    D3D12_RESOURCE_BARRIER uavBarriers[7];
    uavBarriers[0] = CD3DX12_RESOURCE_BARRIER::UAV(m_AccumulationBuffer.resource.Get());
    uavBarriers[1] = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
    uavBarriers[2] = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerPresentOutput.resource.Get());
    uavBarriers[3] = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirIntermediate.resource.Get());
    uavBarriers[4] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[0].resource.Get());
    uavBarriers[5] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[1].resource.Get());
    uavBarriers[6] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirIntermediate.resource.Get());
    m_CommandList->ResourceBarrier(7, uavBarriers);

    m_CommandList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());

    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    m_CommandList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0)); // Bindless
    m_CommandList->SetComputeRootShaderResourceView(4, m_TLAS.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(10, m_LightsBuffer.gpuAddress); // Lights Buffer
    m_CommandList->SetComputeRootShaderResourceView(11, m_LightLUTBuffer.gpuAddress); // Light LUT Buffer

    BindlessIndices indices = {};

    int currentReservoir = m_CurrentReservoirIndex;
    int previousReservoir = 1 - currentReservoir;

    if (frame.useRTXDI)
    {
        // NVIDIA RTXDI Path
        // Bind common RTXDI resources
        m_CommandList->SetComputeRootDescriptorTable(9, GraphicsHelper::GetSRVGPUHandle(m_RtxdiNeighborOffsetsBuffer.srvIndex));

        // Pass 1: Temporal Resampling
        m_CommandList->SetPipelineState(m_RtxdiRestirTemporalPSO.Get());
        m_CommandList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirBuffer[currentReservoir].uavIndex));
        m_CommandList->SetComputeRootDescriptorTable(8, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirBuffer[previousReservoir].uavIndex));
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier1 = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[currentReservoir].resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier1);

        // Pass 2: Spatial Resampling
        m_CommandList->SetPipelineState(m_RtxdiRestirSpatialPSO.Get());
        m_CommandList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirIntermediate.uavIndex));
        m_CommandList->SetComputeRootDescriptorTable(8, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirBuffer[currentReservoir].uavIndex));
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirIntermediate.resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier2);

        // Pass 3: Resolve
        m_CommandList->SetPipelineState(m_RtxdiRestirResolvePSO.Get());
        m_CommandList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirIntermediate.uavIndex));
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices        
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        if (frame.restirReservoirDebugMode != RESTIR_RESERVOIR_DEBUG_OFF && m_RtxdiRestirReservoirDebugPSO)
        {
            D3D12_RESOURCE_BARRIER debugBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
            m_CommandList->ResourceBarrier(1, &debugBarrier);

            m_CommandList->SetPipelineState(m_RtxdiRestirReservoirDebugPSO.Get());
            m_CommandList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirIntermediate.uavIndex));
            indices.OutputIdx0 = m_PathTracerOutput.uavIndex;
            m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
            m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);
        }
    }
    else if (frame.enableRestir)
    {
        // Torture ReSTIR (Manual Implementation)
        // Pass 1: Temporal — writes to ReservoirBuffer[current], reads history from ReservoirBuffer[previous]
        m_CommandList->SetPipelineState(m_RestirTemporalPSO.Get());
        indices.InputIdx0 = m_ReservoirBuffer[previousReservoir].srvIndex;
        indices.OutputIdx0 = m_ReservoirBuffer[currentReservoir].uavIndex;
        indices.OutputIdx1 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.uavIndex : UINT(-1);
        indices.PathVizLineBufferIdx = (uint32_t)m_PathVizLineBuffer.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barriers1[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirBuffer[currentReservoir].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_PathVizLineBuffer.resource.Get())
        };
        m_CommandList->ResourceBarrier(2, barriers1);

        // Pass 2: Spatial — writes to Intermediate, reads temporal from ReservoirBuffer[current]
        m_CommandList->SetPipelineState(m_RestirSpatialPSO.Get());
        indices.InputIdx0 = m_ReservoirBuffer[currentReservoir].srvIndex;
        indices.OutputIdx0 = m_ReservoirIntermediate.uavIndex;
        indices.OutputIdx1 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.uavIndex : UINT(-1);
        indices.PathVizLineBufferIdx = (uint32_t)m_PathVizLineBuffer.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirIntermediate.resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier2);

        // Pass 3: Resolve — reads spatial output from Intermediate
        m_CommandList->SetPipelineState(m_RestirResolvePSO.Get());
        indices.InputIdx0 = m_ReservoirIntermediate.srvIndex;
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        if (frame.restirReservoirDebugMode != RESTIR_RESERVOIR_DEBUG_OFF && m_RestirReservoirDebugPSO)
        {
            D3D12_RESOURCE_BARRIER debugBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
            m_CommandList->ResourceBarrier(1, &debugBarrier);

            if (useCustomRestirHeatmap)
            {
                GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RestirDebugHeatmap, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }

            m_CommandList->SetPipelineState(m_RestirReservoirDebugPSO.Get());
            indices.InputIdx0 = m_ReservoirIntermediate.srvIndex;
            indices.InputIdx1 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.srvIndex : UINT(-1);
            indices.OutputIdx0 = m_PathTracerOutput.uavIndex;
            m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
            m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);
        }
    }
    else
    {
        // Old Path Trace
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        m_CommandList->SetPipelineState(m_PathTracerPSO.Get());
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);
    }

    m_CurrentReservoirIndex = previousReservoir; // Swap for next frame

    if (m_PathTracerPresentPSO)
    {
        D3D12_RESOURCE_BARRIER presentBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
        m_CommandList->ResourceBarrier(1, &presentBarrier);

        m_CommandList->SetPipelineState(m_PathTracerPresentPSO.Get());
        indices.InputIdx0 = m_PathTracerOutput.srvIndex;
        indices.OutputIdx0 = m_PathTracerPresentOutput.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);
    }

    // Transition for blitting/Imgui
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_PathTracerPresentOutput, D3D12_RESOURCE_STATE_COPY_SOURCE);
}

void Renderer::DispatchRestirGI(class Model* model, const FrameConstants& frame)
{
    if (!frame.enableRasterIndirectGI)
    {
        // Reset the flag: NRD did not run from the GI path this frame.
        // If DI is also active with NRD, DispatchRestirDI will set it back to true.
        m_NrdWasActiveLastFrame = false;
        return;
    }
        

    const bool useCustomRestirHeatmap = frame.restirReservoirDebugMode >= RESTIR_RESERVOIR_DEBUG_SOURCE_PDF;

    // Transition G-Buffer targets to SRV state for compute
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GBuffer.albedo, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GBuffer.normal, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GBuffer.material, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GBuffer.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    //GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RasterIndirectLightingTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Split diffuse/specular buffers
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DiffuseReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DiffuseReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_SpecularReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_SpecularReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DiffuseReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_SpecularReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DiffuseCandidateBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (useCustomRestirHeatmap)
    {
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RestirDebugHeatmap, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    m_CommandList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());

    // Bind common resources
    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    m_CommandList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0)); // Bindless
    m_CommandList->SetComputeRootShaderResourceView(4, m_TLAS.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(10, m_LightsBuffer.gpuAddress); // Lights Buffer
    m_CommandList->SetComputeRootShaderResourceView(11, m_LightLUTBuffer.gpuAddress); // Light LUT Buffer

    BindlessIndices indices = {};

    int currentReservoir = m_CurrentReservoirIndex;
    int previousReservoir = 1 - currentReservoir;

#if 0
    // IrCache TODO:
    // 1. Cascade scrollling one cell at time, only cell at edge reallocated
    // 2. Repositioning probes toward nearest open space using ray (avoid probe inside wall)
    // 3. Lazy trace - trace if probe recently allocated, or light changed

    // -----------------------------------------------------------------------
    // Spatial IrCache Pipeline
    // -----------------------------------------------------------------------

    // One-shot pool initialisation on the very first frame
    if (!m_IrCacheInitialized)
    {
        m_IrCacheInitialized = true;

        // Pool Init clears grid meta, pool, life, and meta counters in one pass.
        // Dispatch over TOTAL_CELLS = 262144 (dominant sweep).
        m_CommandList->SetPipelineState(m_IrCachePoolInitPSO.Get());
        m_CommandList->SetComputeRoot32BitConstants(13, sizeof(IrCacheBindlessIndices) / 4, &m_IrCacheIndices, 0);
        m_CommandList->Dispatch((262144 + 63) / 64, 1, 1);

        D3D12_RESOURCE_BARRIER initBarriers[7] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheGridMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCachePoolBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheLifeBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCachePosBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheRepropBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheRepropCountBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(7, initBarriers);
    }

    // Bind IrCache constants once for all IrCache passes
    m_CommandList->SetComputeRoot32BitConstants(13, sizeof(IrCacheBindlessIndices) / 4, &m_IrCacheIndices, 0);

    // --- Pass 1: Prepare Age (reset compact write idx) ---
    m_CommandList->SetPipelineState(m_IrCachePrepareAgePSO.Get());
    m_CommandList->Dispatch(1, 1, 1);

    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheMetaBuf.resource.Get());
        m_CommandList->ResourceBarrier(1, &b);
    }

    // --- Pass 2: Age (expire old entries, build indirection, apply+clear position votes) ---
    m_CommandList->SetPipelineState(m_IrCacheAgePSO.Get());
    m_CommandList->Dispatch((32768 + 63) / 64, 1, 1);

    {
        D3D12_RESOURCE_BARRIER barriers[6] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheLifeBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheIndirectionBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheGridMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCachePosBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheRepropCountBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(6, barriers);
    }

    // --- Pass 3: Prepare Trace (snapshot live count → TraceArgs) ---
    m_CommandList->SetPipelineState(m_IrCachePrepareTracePSO.Get());
    m_CommandList->Dispatch(1, 1, 1);

    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheTraceArgsBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(2, barriers);
    }

    // --- Pass 4: Update probes (indirect, one group per probe) ---
    m_CommandList->SetPipelineState(m_IrCacheUpdatePSO.Get());
    m_CommandList->ExecuteIndirect(m_DispatchCommandSignature.Get(), 1,
        m_IrCacheTraceArgsBuf.resource.Get(), 0, nullptr, 0);

    {
        D3D12_RESOURCE_BARRIER barriers[6] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheIrradianceBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheLifeBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheGridMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCachePoolBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheRepropBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheRepropCountBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(6, barriers);
    }
#endif

    // -----------------------------------------------------------------------
    // SHaRC (Spatial Hash Radiance Cache) Pipeline
    // -----------------------------------------------------------------------

    // Bind SHaRC indices; slot 13 (b2) is read by SHaRC_Update, SHaRC_Resolve,
    // and RestirGI_Raster_Temporal (query pass) — set once, persists for all three.
    m_CommandList->SetComputeRoot32BitConstants(13, sizeof(SharcBindlessIndices) / 4, &m_SharcIndices, 0);

    // --- Pass 1: SHaRC Update — trace secondary rays, deposit samples into hash table ---
    // Downscale by 5 (matching RTXGI default): each thread updates one rotating
    // full-resolution pixel inside a 5x5 tile, giving 25x fewer deposits per frame
    // while maintaining whole-screen coverage over time.
    static constexpr UINT SHARC_UPDATE_DOWNSCALE = 5;
    const UINT sharcUpdateW = (m_InternalWidth  + SHARC_UPDATE_DOWNSCALE - 1) / SHARC_UPDATE_DOWNSCALE;
    const UINT sharcUpdateH = (m_InternalHeight + SHARC_UPDATE_DOWNSCALE - 1) / SHARC_UPDATE_DOWNSCALE;
    m_CommandList->SetPipelineState(m_SharcUpdatePSO.Get());
    m_CommandList->Dispatch((sharcUpdateW + 7) / 8, (sharcUpdateH + 7) / 8, 1);

    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcHashEntriesBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcAccumulationBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(2, barriers);
    }

    // --- Pass 2: SHaRC Resolve — EMA blend accumulation→resolved, clears accumulation ---
    m_CommandList->SetPipelineState(m_SharcResolvePSO.Get());
    m_CommandList->Dispatch((SHARC_HASH_ENTRIES_NUM + 255) / 256, 1, 1);

    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcHashEntriesBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_SharcResolvedBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(2, barriers);
    }

    // -----------------------------------------------------------------------
    // Split Diffuse / Specular ReSTIR passes
    // (1) RTDGI Temporal → (2) RTR Temporal → (3) Diffuse Spatial →
    // (4) Specular Spatial → (5) Split Resolve
    // -----------------------------------------------------------------------

    // --- Pass 1: Diffuse Temporal (RTDGI) ---
    // InputIdx0 = prev diffuse reservoirs, OutputIdx0 = curr diffuse reservoirs,
    // OutputIdx1 = diffuse candidate buffer, OutputIdx2 = debug heatmap
    m_CommandList->SetPipelineState(m_DiffuseTemporalPSO.Get());
    indices.InputIdx0  = m_DiffuseReservoirBuffer[previousReservoir].srvIndex;
    indices.OutputIdx0 = m_DiffuseReservoirBuffer[currentReservoir].uavIndex;
    indices.OutputIdx1 = m_DiffuseCandidateBuffer.uavIndex;
    indices.OutputIdx2 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.uavIndex : UINT(-1);
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    {
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_DiffuseReservoirBuffer[currentReservoir].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_DiffuseCandidateBuffer.resource.Get()),
        };
        m_CommandList->ResourceBarrier(2, barriers);
    }

    // --- Pass 2: Specular Temporal (RTR) ---
    // InputIdx0 = prev specular reservoirs, InputIdx1 = diffuse candidate buffer (SRV),
    // OutputIdx0 = curr specular reservoirs, OutputIdx1 = debug heatmap
    m_CommandList->SetPipelineState(m_SpecularTemporalPSO.Get());
    indices.InputIdx0  = m_SpecularReservoirBuffer[previousReservoir].srvIndex;
    indices.InputIdx1  = m_DiffuseCandidateBuffer.srvIndex;
    indices.OutputIdx0 = m_SpecularReservoirBuffer[currentReservoir].uavIndex;
    indices.OutputIdx1 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.uavIndex : UINT(-1);
    indices.OutputIdx2 = UINT(-1);
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_SpecularReservoirBuffer[currentReservoir].resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier);
    }

    // --- Pass 3: Diffuse Spatial ---
    // InputIdx0 = curr diffuse reservoirs, OutputIdx0 = diffuse intermediate
    m_CommandList->SetPipelineState(m_DiffuseSpatialPSO.Get());
    indices.InputIdx0  = m_DiffuseReservoirBuffer[currentReservoir].srvIndex;
    indices.InputIdx1  = UINT(-1);
    indices.OutputIdx0 = m_DiffuseReservoirIntermediate.uavIndex;
    indices.OutputIdx1 = UINT(-1);
    indices.OutputIdx2 = UINT(-1);
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_DiffuseReservoirIntermediate.resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier);
    }

    // --- Pass 4: Specular Spatial ---
    // InputIdx0 = curr specular reservoirs, OutputIdx0 = specular intermediate
    m_CommandList->SetPipelineState(m_SpecularSpatialPSO.Get());
    indices.InputIdx0  = m_SpecularReservoirBuffer[currentReservoir].srvIndex;
    indices.OutputIdx0 = m_SpecularReservoirIntermediate.uavIndex;
    indices.OutputIdx1 = UINT(-1);
    indices.OutputIdx2 = UINT(-1);
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_SpecularReservoirIntermediate.resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier);
    }

    const bool useNrd = frame.enableNrdRelax != 0
        && frame.restirReservoirDebugMode == RESTIR_RESERVOIR_DEBUG_OFF
        && frame.sharcDebug == 0;

    // --- Pass 4b: GI Resolve Intermediates ---
    // Converts GI reservoir StructuredBuffers → raw float4 intermediates (BRDF eval + NRD normalize).
    // Always dispatched when GI is active; SSO always needs the intermediates to bridge to Final*.
    if (m_GIResolveIntermediatesPSO)
    {
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_DiffuseReservoirIntermediate,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_SpecularReservoirIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GIDiffuseIntermediate,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GISpecularIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        indices = {};
        indices.InputIdx0  = m_DiffuseReservoirIntermediate.srvIndex;
        indices.InputIdx1  = m_SpecularReservoirIntermediate.srvIndex;
        indices.OutputIdx0 = m_GIDiffuseIntermediate.uavIndex;
        indices.OutputIdx1 = m_GISpecularIntermediate.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        m_CommandList->SetPipelineState(m_GIResolveIntermediatesPSO.Get());
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER giResolveBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_GIDiffuseIntermediate.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_GISpecularIntermediate.resource.Get()),
        };
        m_CommandList->ResourceBarrier(_countof(giResolveBarriers), giResolveBarriers);
    }

    // --- Pass 4c: StoreShadingOutput Call 2 (GI contribution) ---
    // Reads GI intermediates and bridges into FinalDiffuseTex / FinalSpecularTex.
    // Always dispatched when GI is active (regardless of NRD state).
    // isFirstPass = 0 if DI ran this frame (additive blend), 1 if DI was off (overwrite).
    if (m_NrdStoreShadingOutputPSO && m_GIResolveIntermediatesPSO)
    {
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GIDiffuseIntermediate,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GISpecularIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        indices = {};
        indices.InputIdx0  = m_GIDiffuseIntermediate.srvIndex;
        indices.InputIdx1  = m_GISpecularIntermediate.srvIndex;
        indices.OutputIdx0 = m_FinalDiffuseTex.uavIndex;
        indices.OutputIdx1 = m_FinalSpecularTex.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        // isFirstPass=0 if DI ran this frame (additive blend), 1 if DI was off (overwrite)
        const UINT isFirstPass = (frame.enableRestirDI != 0u) ? 0u : 1u;
        m_CommandList->SetComputeRoot32BitConstants(13, 1, &isFirstPass, 0);
        m_CommandList->SetPipelineState(m_NrdStoreShadingOutputPSO.Get());
        m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER ssoBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_FinalDiffuseTex.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_FinalSpecularTex.resource.Get()),
        };
        m_CommandList->ResourceBarrier(_countof(ssoBarriers), ssoBarriers);
    }

    if (useNrd && NRDDenoise(frame))
    {
        m_CurrentReservoirIndex = previousReservoir;
        return;
    }

    // NRD path was not taken this frame — record so next activation can RESTART.
    m_NrdWasActiveLastFrame = false;

    // NRD disabled: transition Final* to SRV for Lighting.hlsl
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    m_CurrentReservoirIndex = previousReservoir; // Swap for next frame
}

bool Renderer::NRDDenoise(const FrameConstants& frame)
{
    if (!m_NrdPrepareGuidesPSO || !m_NrdCompositePSO || !InitializeNrd())
        return false;

    if (!m_NrdPackNoisePSO)
        return false;

    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdMotionVectorsTex,    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdNormalRoughnessTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdViewZTex,            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdDenoisedDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdDenoisedSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdValidationTex,       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // ---- Step 1: NRD Prepare Guides (unchanged) ----
    m_CommandList->SetPipelineState(m_NrdPrepareGuidesPSO.Get());
    BindlessIndices indices = {};
    indices.OutputIdx0 = m_NrdMotionVectorsTex.uavIndex;
    indices.OutputIdx1 = m_NrdNormalRoughnessTex.uavIndex;
    indices.OutputIdx2 = m_NrdViewZTex.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER guideBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdMotionVectorsTex.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdNormalRoughnessTex.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdViewZTex.resource.Get())
    };
    m_CommandList->ResourceBarrier(_countof(guideBarriers), guideBarriers);

    // ---- Step 2: NrdPackNoise — Final* → RELAX format ----
    // FinalDiffuse/FinalSpecular already contain the merged DI+GI signal (written by SSO calls).
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdRelaxDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdRelaxSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_CommandList->SetPipelineState(m_NrdPackNoisePSO.Get());
    indices = {};
    indices.InputIdx0  = m_FinalDiffuseTex.srvIndex;
    indices.InputIdx1  = m_FinalSpecularTex.srvIndex;
    indices.OutputIdx0 = m_NrdRelaxDiffuseTex.uavIndex;
    indices.OutputIdx1 = m_NrdRelaxSpecularTex.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER relaxBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdRelaxDiffuseTex.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdRelaxSpecularTex.resource.Get()),
    };
    m_CommandList->ResourceBarrier(_countof(relaxBarriers), relaxBarriers);

    DirectX::XMMATRIX projectionInverse = DirectX::XMLoadFloat4x4(&frame.projectionInverse);
    DirectX::XMMATRIX projection = DirectX::XMMatrixInverse(nullptr, projectionInverse);
    DirectX::XMMATRIX view = DirectX::XMMatrixInverse(nullptr, DirectX::XMLoadFloat4x4(&frame.viewInverse));
    DirectX::XMMATRIX viewPrev = DirectX::XMMatrixInverse(nullptr, DirectX::XMLoadFloat4x4(&frame.viewInversePrevious));

    nrd::CommonSettings commonSettings = {};
    CopyMatrixToNrd(commonSettings.viewToClipMatrix, projection);
    CopyMatrixToNrd(commonSettings.viewToClipMatrixPrev, projection);
    CopyMatrixToNrd(commonSettings.worldToViewMatrix, view);
    CopyMatrixToNrd(commonSettings.worldToViewMatrixPrev, viewPrev);
    commonSettings.motionVectorScale[0] = 1.0f;
    commonSettings.motionVectorScale[1] = 1.0f;
    commonSettings.motionVectorScale[2] = 0.0f;
    commonSettings.resourceSize[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.resourceSize[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.resourceSizePrev[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.resourceSizePrev[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.rectSize[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.rectSize[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.rectSizePrev[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.rectSizePrev[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.denoisingRange = 1000.0f;
    commonSettings.disocclusionThreshold = 0.01f;
    commonSettings.frameIndex = frame.frameIndex;
    // Force RESTART when NRD was inactive on the previous frame (e.g. raster
    // indirect GI was toggled off then back on).  This resets NRD's internal
    // m_PrevFrameIndexFromSettings so the +1-per-frame assertion won't fire.
    const bool needRestart = (frame.frameIndex <= 1) || !m_NrdWasActiveLastFrame;
    commonSettings.accumulationMode = needRestart ? nrd::AccumulationMode::RESTART : nrd::AccumulationMode::CONTINUE;
    commonSettings.enableValidation = frame.enableNrdValidation != 0;

    nrd::RelaxSettings relaxSettings = {};
    relaxSettings.diffuseMaxAccumulatedFrameNum = 12;
    relaxSettings.specularMaxAccumulatedFrameNum = 8;
    relaxSettings.diffuseMaxFastAccumulatedFrameNum = 4;
    relaxSettings.specularMaxFastAccumulatedFrameNum = 3;
    relaxSettings.diffusePrepassBlurRadius = 20.0f;
    relaxSettings.specularPrepassBlurRadius = 40.0f;
    relaxSettings.minHitDistanceWeight = 0.05f;
    relaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
    relaxSettings.enableAntiFirefly = true;

    m_NrdIntegration->NewFrame();
    if (m_NrdIntegration->SetCommonSettings(commonSettings) != nrd::Result::SUCCESS)
        return false;
    if (m_NrdIntegration->SetDenoiserSettings(kNrdRelaxDiffuseSpecularIdentifier, &relaxSettings) != nrd::Result::SUCCESS)
        return false;

    nrd::ResourceSnapshot resourceSnapshot = {};
    resourceSnapshot.restoreInitialState = true;
    resourceSnapshot.SetResource(nrd::ResourceType::IN_MV, MakeNrdResource(m_NrdMotionVectorsTex));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, MakeNrdResource(m_NrdNormalRoughnessTex));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, MakeNrdResource(m_NrdViewZTex));
    // NRD reads RELAX-packed textures from NrdPackNoise output.
    resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, MakeNrdResource(m_NrdRelaxDiffuseTex));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, MakeNrdResource(m_NrdRelaxSpecularTex));
    resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, MakeNrdResource(m_NrdDenoisedDiffuseTex));
    resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, MakeNrdResource(m_NrdDenoisedSpecularTex));
    if (frame.enableNrdValidation != 0)
    {
        resourceSnapshot.SetResource(nrd::ResourceType::OUT_VALIDATION, MakeNrdResource(m_NrdValidationTex));
    }

    nri::CommandBufferD3D12Desc commandBufferDesc = {};
    commandBufferDesc.d3d12CommandList = m_CommandList.Get();
    commandBufferDesc.d3d12CommandAllocator = m_CommandAllocator.Get();

    const nrd::Identifier denoisers[] = { kNrdRelaxDiffuseSpecularIdentifier };
    m_NrdIntegration->DenoiseD3D12(denoisers, _countof(denoisers), commandBufferDesc, resourceSnapshot);

    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdDenoisedDiffuseTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdDenoisedSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (frame.enableNrdValidation != 0)
    {
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdValidationTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    //GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdUnpackedDiffuseTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    //GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdUnpackedSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* heaps[] = { GraphicsHelper::GetSRVHeap() };
    m_CommandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());
    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    // The composite pass reads NRD output (SRV) and writes denoised radiance back to Final* (UAV).
    // This is a circular write-back: Final* was read by NrdPackNoise, now overwritten with denoised data.
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_CommandList->SetPipelineState(m_NrdCompositePSO.Get());
    indices = {};
    indices.InputIdx0 = m_NrdDenoisedDiffuseTex.srvIndex;
    indices.InputIdx1 = m_NrdDenoisedSpecularTex.srvIndex;
    indices.InputIdx2 = frame.enableNrdValidation != 0 ? m_NrdValidationTex.srvIndex : UINT(-1);
    indices.OutputIdx0 = m_FinalDiffuseTex.uavIndex;
    indices.OutputIdx1 = m_FinalSpecularTex.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    // Transition Final* to pixel-shader-readable SRV for Lighting.hlsl
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    m_NrdWasActiveLastFrame = true;
    return true;
}

void Renderer::DrawPathVizLines(const FrameConstants& frame)
{
    if (!m_PathVizLinePSO || !m_PathVizLineBuffer.resource) return;

    // Transition buffer from UAV (written by compute) to SRV-readable for VS
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_PathVizLineBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = GetCurrentBackBufferRTV();
    m_CommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
    D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, (LONG)WINDOW_WIDTH, (LONG)WINDOW_HEIGHT);
    m_CommandList->RSSetViewports(1, &viewport);
    m_CommandList->RSSetScissorRects(1, &scissorRect);

    m_CommandList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    m_CommandList->SetGraphicsRootSignature(m_RootSignature.Get());
    m_CommandList->SetGraphicsRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0)); // Bindless

    BindlessIndices vizIndices = {};
    vizIndices.PathVizLineBufferIdx = (uint32_t)m_PathVizLineBuffer.srvIndex;
    m_CommandList->SetGraphicsRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &vizIndices, 0);

    m_CommandList->SetPipelineState(m_PathVizLinePSO.Get());
    m_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
    m_CommandList->DrawInstanced(MAX_PATH_VIZ_LINES * 2, 1, 0, 0);
}

void Renderer::CopyTextureToBackBuffer(const GPUTexture& texture)
{
    // Ensure source texture is in COPY_SOURCE
    GraphicsHelper::TransitionResource(m_CommandList.Get(), const_cast<GPUTexture&>(texture), D3D12_RESOURCE_STATE_COPY_SOURCE);

    // Transition backbuffer to COPY_DEST
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RenderTargets[m_FrameIndex].Get(), m_BackBufferStates[m_FrameIndex], D3D12_RESOURCE_STATE_COPY_DEST);

    // Check if source and destination have the same dimensions
    D3D12_RESOURCE_DESC srcDesc = texture.resource->GetDesc();
    D3D12_RESOURCE_DESC dstDesc = m_RenderTargets[m_FrameIndex]->GetDesc();

    if (srcDesc.Width == dstDesc.Width && srcDesc.Height == dstDesc.Height)
    {
        m_CommandList->CopyResource(m_RenderTargets[m_FrameIndex].Get(), texture.resource.Get());
    }
    else
    {
        // Copy the smaller region (internal resolution) to the top-left of the back buffer
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = m_RenderTargets[m_FrameIndex].Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = texture.resource.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;

        D3D12_BOX srcBox = {};
        srcBox.left = 0;
        srcBox.top = 0;
        srcBox.front = 0;
        srcBox.right = (UINT)std::min(srcDesc.Width, dstDesc.Width);
        srcBox.bottom = std::min(srcDesc.Height, dstDesc.Height);
        srcBox.back = 1;

        m_CommandList->CopyTextureRegion(&dst, 0, 0, 0, &src, &srcBox);
    }

    // Transition backbuffer to RTV for ImGui
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RenderTargets[m_FrameIndex].Get(), m_BackBufferStates[m_FrameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET);
}

void Renderer::BuildAccelerationStructures(Model* model)
{
    if (!m_RayTracingSupported || !model)
        return;

    // Keep temporary buffers alive until ExecuteCommandList finishes
    GPUBuffer scratchBuffer;
    GPUBuffer tlasScratch;
    GPUBuffer instanceDescBuffer;

    // Reset command list for AS build
    m_CommandAllocator->Reset();
    m_CommandList->Reset(m_CommandAllocator.Get(), nullptr);

    Microsoft::WRL::ComPtr<ID3D12Device5> device5;
    CHECK_HR(m_Device.As(&device5), "Failed to get ID3D12Device5");

    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmdList4;
    CHECK_HR(m_CommandList.As(&cmdList4), "Failed to get ID3D12GraphicsCommandList4");

    // 1. Identify all unique primitives and build BLAS for each
    std::vector<const GLTFPrimitive*> modelPrims;
    model->GetAllPrimitives(modelPrims);

    struct BLASBuildInfo {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs;
        D3D12_RAYTRACING_GEOMETRY_DESC geom;
    };
    std::vector<BLASBuildInfo> buildInfos;
    std::vector<GLTFPrimitive*> primsToBuild;
    UINT64 maxScratchSize = 0;

    for (const auto* cp : modelPrims)
    {
        GLTFPrimitive* prim = const_cast<GLTFPrimitive*>(cp);
        BLASBuildInfo info = {};
        info.geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        info.geom.Triangles.VertexBuffer.StartAddress = model->GetGlobalVertexBufferAddress() + (prim->globalVertexOffset * sizeof(GLTFVertex));
        info.geom.Triangles.VertexBuffer.StrideInBytes = sizeof(GLTFVertex);
        info.geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        info.geom.Triangles.VertexCount = static_cast<UINT>(prim->vertices.size());
        info.geom.Triangles.IndexBuffer = model->GetGlobalIndexBufferAddress() + (prim->globalIndexOffset * sizeof(uint32_t));
        info.geom.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        info.geom.Triangles.IndexCount = static_cast<UINT>(prim->indices.size());
        
        if (prim->alphaMode == AlphaMode::Mask || prim->alphaMode == AlphaMode::Blend)
        {
            info.geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        }
        else
        {
            info.geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        }

        info.inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        info.inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        info.inputs.NumDescs = 1;
        info.inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        info.inputs.pGeometryDescs = &info.geom;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
        device5->GetRaytracingAccelerationStructurePrebuildInfo(&info.inputs, &prebuildInfo);

        maxScratchSize = (maxScratchSize > prebuildInfo.ScratchDataSizeInBytes) ? maxScratchSize : prebuildInfo.ScratchDataSizeInBytes;

        GPUBuffer blasBuffer;
        if (CreateBuffer(blasBuffer, prebuildInfo.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE))
        {
            m_BlasPool[prim] = std::move(blasBuffer);
            buildInfos.push_back(info);
            primsToBuild.push_back(prim);
        }
    }

    if (!primsToBuild.empty())
    {
        CreateBuffer(scratchBuffer, maxScratchSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        for (size_t i = 0; i < primsToBuild.size(); ++i)
        {
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
            buildDesc.Inputs = buildInfos[i].inputs;
            buildDesc.Inputs.pGeometryDescs = &buildInfos[i].geom; // Use pointer to internal geom
            buildDesc.ScratchAccelerationStructureData = scratchBuffer.gpuAddress;
            buildDesc.DestAccelerationStructureData = m_BlasPool[primsToBuild[i]].gpuAddress;

            cmdList4->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
            D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(scratchBuffer.resource.Get());
            m_CommandList->ResourceBarrier(1, &uavBarrier);
        }
    }

    // 2. Build TLAS for all draw node instances
    const auto& nodeData = model->GetDrawNodeData();
    std::vector<const GLTFPrimitive*> nodePrims;
    model->GetDrawNodePrimitives(nodePrims);

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDescs;
    for (size_t i = 0; i < nodeData.size(); ++i)
    {
        D3D12_RAYTRACING_INSTANCE_DESC inst = {};
        const auto& world = nodeData[i].world;
        inst.Transform[0][0] = world._11; inst.Transform[0][1] = world._21; inst.Transform[0][2] = world._31; inst.Transform[0][3] = world._41;
        inst.Transform[1][0] = world._12; inst.Transform[1][1] = world._22; inst.Transform[1][2] = world._32; inst.Transform[1][3] = world._42;
        inst.Transform[2][0] = world._13; inst.Transform[2][1] = world._23; inst.Transform[2][2] = world._33; inst.Transform[2][3] = world._43;

        inst.InstanceID = static_cast<UINT>(i);
        inst.InstanceMask = 0xFF;
        inst.InstanceContributionToHitGroupIndex = 0;
        inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE;
        inst.AccelerationStructure = m_BlasPool[nodePrims[i]].gpuAddress;
        instanceDescs.push_back(inst);
    }

    if (!instanceDescs.empty())
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
        tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        tlasInputs.NumDescs = static_cast<UINT>(instanceDescs.size());
        tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuildInfo = {};
        device5->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuildInfo);

        CreateBuffer(m_TLAS, tlasPrebuildInfo.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);

        CreateBuffer(tlasScratch, tlasPrebuildInfo.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        CreateBuffer(instanceDescBuffer, instanceDescs.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        memcpy(instanceDescBuffer.cpuPtr, instanceDescs.data(), instanceDescs.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
        tlasBuildDesc.Inputs = tlasInputs;
        tlasBuildDesc.Inputs.InstanceDescs = instanceDescBuffer.gpuAddress;
        tlasBuildDesc.ScratchAccelerationStructureData = tlasScratch.gpuAddress;
        tlasBuildDesc.DestAccelerationStructureData = m_TLAS.gpuAddress;

        cmdList4->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_TLAS.resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier);
    }

    ExecuteCommandList();
    std::cout << "Built acceleration structures for " << instanceDescs.size() << " instances." << std::endl;
}

void Renderer::CreateLightsBuffer()
{
    // Create Structured Buffer directly on Upload Heap for easy per-frame updates without command list
    CHECK_BOOL(CreateStructuredBuffer(m_LightsBuffer, sizeof(LightConstants), m_MaxLights, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ), "CreateLightsBuffer failed");
}

void Renderer::UpdateLightsBuffer(const std::vector<LightConstants>& lights)
{
    if (lights.empty()) return;
    UINT numLights = std::min((UINT)lights.size(), m_MaxLights);
    
    // Calculate selection PDFs before copying to GPU
    std::vector<LightConstants> lightsWithPDF = lights;
    float totalWeight = 0.0f;
    std::vector<float> weights(numLights);

    // Light 0 is treated as the main directional light and is not part of the local-light LUT domain.
    lightsWithPDF[0].selectionPDF = 0.0f;

    uint32_t localLightCount = 0;
    for (UINT i = 1; i < numLights; ++i) {
        float w = 0.0f;
        if (lights[i].position.w > 0.5f) {
            float luminance = 0.2126f * lights[i].color.x + 0.7152f * lights[i].color.y + 0.0722f * lights[i].color.z;
            w = lights[i].intensity * luminance;
            localLightCount++;
        }
        weights[i] = w;
        totalWeight += w;
    }

    for (UINT i = 1; i < numLights; ++i) {
        if (lights[i].position.w <= 0.5f) {
            lightsWithPDF[i].selectionPDF = 0.0f;
        } else if (totalWeight > 0.0f) {
            lightsWithPDF[i].selectionPDF = weights[i] / totalWeight;
        } else if (localLightCount > 0) {
            lightsWithPDF[i].selectionPDF = 1.0f / float(localLightCount);
        } else {
            lightsWithPDF[i].selectionPDF = 0.0f;
        }
    }

    memcpy(m_LightsBuffer.cpuPtr, lightsWithPDF.data(), numLights * sizeof(LightConstants));
    
    // Also update the LUT buffer for importance sampling
    UpdateLightLUTBuffer(lightsWithPDF);
}

D3D12_GPU_VIRTUAL_ADDRESS Renderer::GetLightsBufferGPUAddress() const
{
    return m_LightsBuffer.gpuAddress;
}

void Renderer::CreateLightLUTBuffer()
{
    // LUT buffer: 256 uint entries for O(1) light sampling
    // Each entry maps a CDF bucket to a light index
    CHECK_BOOL(CreateBuffer(m_LightLUTBuffer, sizeof(uint32_t) * LIGHT_LUT_RESOLUTION, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, true), "CreateLightLUTBuffer failed");
}

void Renderer::UpdateLightLUTBuffer(const std::vector<LightConstants>& lights)
{
    if (lights.empty() || m_LightLUTBuffer.cpuPtr == nullptr) return;
    
    UINT numLights = std::min((UINT)lights.size(), m_MaxLights);
    std::vector<uint32_t> lut(LIGHT_LUT_RESOLUTION);
    std::vector<float> weights(numLights);
    std::vector<uint32_t> localLightIndices;
    localLightIndices.reserve(numLights > 0 ? numLights - 1 : 0);
    
    // Build a local-light-only distribution. Light 0 remains deterministic and bypasses this LUT.
    float totalWeight = 0.0f;
    for (UINT i = 1; i < numLights; ++i) {
        float w = 0.0f;
        if (lights[i].position.w > 0.5f) {
            float luminance = 0.2126f * lights[i].color.x + 0.7152f * lights[i].color.y + 0.0722f * lights[i].color.z;
            w = lights[i].intensity * luminance;
            localLightIndices.push_back(i);
        }
        weights[i] = w;
        totalWeight += w;
    }

    if (localLightIndices.empty()) {
        std::fill(lut.begin(), lut.end(), 0u);
        memcpy(m_LightLUTBuffer.cpuPtr, lut.data(), LIGHT_LUT_RESOLUTION * sizeof(uint32_t));
        return;
    }

    if (totalWeight <= 0.0f) {
        for (UINT i = 0; i < LIGHT_LUT_RESOLUTION; ++i) {
            lut[i] = localLightIndices[i % localLightIndices.size()];
        }
        memcpy(m_LightLUTBuffer.cpuPtr, lut.data(), LIGHT_LUT_RESOLUTION * sizeof(uint32_t));
        return;
    }
    
    // Build CDF
    std::vector<float> cdf(numLights);
    float cumulative = 0.0f;
    for (UINT i = 1; i < numLights; ++i) {
        cumulative += weights[i] / totalWeight;
        cdf[i] = cumulative;
    }
    cdf[numLights - 1] = 1.0f;
    
    // Build LUT: for each LUT entry, find which light index to sample
    for (UINT i = 0; i < LIGHT_LUT_RESOLUTION; ++i) {
        float u = (i + 0.5f) / float(LIGHT_LUT_RESOLUTION); // Center of bin
        
        // Binary search in CDF to find light index
        uint32_t lightIdx = localLightIndices.back();
        for (UINT j = 1; j < numLights; ++j) {
            if (weights[j] > 0.0f && u <= cdf[j]) {
                lightIdx = j;
                break;
            }
        }
        lut[i] = lightIdx;
    }
    
    // Copy to GPU
    memcpy(m_LightLUTBuffer.cpuPtr, lut.data(), LIGHT_LUT_RESOLUTION * sizeof(uint32_t));
}

void Renderer::CreateGBuffer(uint32_t w, uint32_t h)
{
    float blackClear[] = { 0, 0, 0, 0 };
    CreateTexture(m_GBuffer.albedo, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.normal, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.material, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.depth, w, h, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE);
}

std::vector<char> Renderer::LoadShader(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Failed to open shader file: " << filename << std::endl;
        return std::vector<char>();
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    return buffer;
}

void Renderer::UpdateFrameCB(const FrameConstants& frameConstants)
{
    memcpy(m_FrameCB.cpuPtr, &frameConstants, sizeof(FrameConstants));
}

void Renderer::GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter)
{
    *ppAdapter = nullptr;
    for (UINT adapterIndex = 0; ; ++adapterIndex)
    {
        IDXGIAdapter1* pAdapter = nullptr;
        if (DXGI_ERROR_NOT_FOUND == pFactory->EnumAdapters1(adapterIndex, &pAdapter))
        {
            break;
        }

        DXGI_ADAPTER_DESC1 desc;
        pAdapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        {
            pAdapter->Release();
            continue;
        }

        if (SUCCEEDED(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
        {
            *ppAdapter = pAdapter;
            return;
        }

        pAdapter->Release();
    }
}

void Renderer::WaitForPreviousFrame()
{
    // Signal and increment the fence value
    const UINT64 fence = m_FenceValue;
    CHECK_HR(m_CommandQueue->Signal(m_Fence.Get(), fence), "CommandQueue Signal failed");
    m_FenceValue++;

    // Wait until the previous frame is finished
    if (m_Fence->GetCompletedValue() < fence)
    {
        CHECK_HR(m_Fence->SetEventOnCompletion(fence, m_FenceEvent), "SetEventOnCompletion failed");
        WaitForSingleObject(m_FenceEvent, INFINITE);
    }

    m_FrameIndex = m_SwapChain->GetCurrentBackBufferIndex();
}

void Renderer::TransitionBackBuffer(D3D12_RESOURCE_STATES newState)
{
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_RenderTargets[m_FrameIndex].Get(), m_BackBufferStates[m_FrameIndex], newState);
}

// =============================================================================
// Internal Resolution Resource Management
// =============================================================================

void Renderer::CreateInternalResolutionResources(uint32_t w, uint32_t h)
{
    // GPU-idle before releasing and reallocating resources
    WaitForPreviousFrame();

    m_InternalWidth = w;
    m_InternalHeight = h;

    std::cout << "Recreating internal-resolution resources at " << w << "x" << h << std::endl;

    // ---- GBuffer ----
    CreateGBuffer(w, h);

    // ---- Path Tracer Output Textures ----
    if (m_RayTracingSupported)
    {
        CreateTexture(m_AccumulationBuffer, w, h, DXGI_FORMAT_R32G32B32A32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
        CreateTexture(m_PathTracerOutput, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
        CreateTexture(m_PathTracerPresentOutput, w, h, DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
        CreateTexture(m_RestirDebugHeatmap, w, h, DXGI_FORMAT_R16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);

        // ReSTIR-DI Reservoirs
        for (int i = 0; i < 2; ++i)
            CreateStructuredBuffer(m_ReservoirBuffer[i], sizeof(Reservoir), w * h,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CreateStructuredBuffer(m_ReservoirIntermediate, sizeof(Reservoir), w * h,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // RTXDI Reservoirs
        uint32_t renderWidthBlocks = (w + 15) / 16;
        uint32_t renderHeightBlocks = (h + 15) / 16;
        uint32_t reservoirArrayPitch = renderWidthBlocks * 256 * renderHeightBlocks;
        for (int i = 0; i < 2; ++i)
            CreateStructuredBuffer(m_RtxdiReservoirBuffer[i], sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CreateStructuredBuffer(m_RtxdiReservoirIntermediate, sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    // ---- Split Diffuse / Specular ReSTIR buffers ----
    for (int i = 0; i < 2; ++i) {
        CreateStructuredBuffer(m_DiffuseReservoirBuffer[i], sizeof(Reservoir), w * h,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CreateStructuredBuffer(m_SpecularReservoirBuffer[i], sizeof(Reservoir), w * h,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    CreateStructuredBuffer(m_DiffuseReservoirIntermediate, sizeof(Reservoir), w * h,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_SpecularReservoirIntermediate, sizeof(Reservoir), w * h,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_DiffuseCandidateBuffer, sizeof(DiffuseCandidate), w * h,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // ---- ReSTIR DI buffers ----
    for (int i = 0; i < 2; ++i)
        CreateStructuredBuffer(m_DIReservoirBuffer[i], sizeof(DIRreservoir), w * h,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_DIReservoirIntermediate, sizeof(DIRreservoir), w * h,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Split DI intermediates for SSO bridge path
    CreateTexture(m_DIDiffuseIntermediate, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_DISpecularIntermediate, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // ---- Raster HDR + NRD Textures ----
    // HDR render target for rasterizer lighting when TAA is active
    CreateTexture(m_RasterHdrOutputTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RENDER_TARGET);
    CreateTexture(m_NrdMotionVectorsTex, w, h, DXGI_FORMAT_R16G16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdNormalRoughnessTex, w, h, DXGI_FORMAT_R10G10B10A2_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdViewZTex, w, h, DXGI_FORMAT_R16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // GI resolved intermediates (raw float4: NRD-normalized radiance + hitT)
    CreateTexture(m_GIDiffuseIntermediate, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_GISpecularIntermediate, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // RELAX-packed inputs for NRD denoiser (NrdPackNoise output)
    CreateTexture(m_NrdRelaxDiffuseTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdRelaxSpecularTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdDenoisedDiffuseTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdDenoisedSpecularTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_NrdValidationTex, w, h, DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Universal interchange textures: SSO writes, NrdPackNoise+Lighting read
    CreateTexture(m_FinalDiffuseTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateTexture(m_FinalSpecularTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // ---- Re-initialize NRD at new resolution ----
    if (m_NrdInitialized)
    {
        ShutdownNrd();
        InitializeNrd();
    }

    std::cout << "Internal-resolution resources recreated: " << w << "x" << h << std::endl;
}

// =============================================================================
// TAA / Temporal Super-Resolution
// =============================================================================

void Renderer::CreateTaaResources(uint32_t outputW, uint32_t outputH, uint32_t internalW, uint32_t internalH)
{
    // Output-resolution textures (shared by both modes)
    for (int i = 0; i < 2; ++i)
    {
        CreateTexture(m_TaaHistoryTex[i], outputW, outputH,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    CreateTexture(m_TaaReprojectedHistoryTex, outputW, outputH,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    CreateTexture(m_TaaClosestVelocityTex, outputW, outputH,
        DXGI_FORMAT_R16G16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    CreateTexture(m_TaaOutputTex, outputW, outputH,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_TaaEnabled = true;
    m_TaaHistoryIndex = 0;

    std::cout << "TAA resources created: output=" << outputW << "x" << outputH
              << " internal=" << internalW << "x" << internalH << std::endl;
}

void Renderer::CreateTaaPipelines()
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
    computeDesc.pRootSignature = m_RootSignature.Get();

    // Naive TSR PSOs
    auto reprojectCS = GraphicsHelper::CompileShader("Shaders/NaiveTsr_Reproject.hlsl", "main", "cs_6_6");
    if (!reprojectCS.empty())
    {
        computeDesc.CS = { reprojectCS.data(), reprojectCS.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NaiveTsrReprojectPSO)),
            "Failed to create NaiveTsr Reproject PSO");
    }

    auto resolveCS = GraphicsHelper::CompileShader("Shaders/NaiveTsr_Resolve.hlsl", "main", "cs_6_6");
    if (!resolveCS.empty())
    {
        computeDesc.CS = { resolveCS.data(), resolveCS.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NaiveTsrResolvePSO)),
            "Failed to create NaiveTsr Resolve PSO");
    }

    // Motion vector generation PSO
    auto motionVecCS = GraphicsHelper::CompileShader("Shaders/MotionVectors.hlsl", "main", "cs_6_6");
    if (!motionVecCS.empty())
    {
        computeDesc.CS = { motionVecCS.data(), motionVecCS.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_MotionVectorsPSO)),
            "Failed to create Motion Vectors PSO");
    }

    std::cout << "TAA pipelines created" << std::endl;
}

void Renderer::GenerateMotionVectors(const FrameConstants& frame)
{
    if (!m_MotionVectorsPSO) return;

    // Transition depth to SRV, motion vectors to UAV
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GBuffer.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdMotionVectorsTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());
    m_CommandList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    BindlessIndices indices = {};
    indices.OutputIdx0 = m_NrdMotionVectorsTex.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);

    m_CommandList->SetPipelineState(m_MotionVectorsPSO.Get());
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_NrdMotionVectorsTex.resource.Get());
    m_CommandList->ResourceBarrier(1, &barrier);
}

void Renderer::DispatchNaiveTsr(const FrameConstants& frame, const GPUTexture& inputColor)
{
    if (!m_NaiveTsrReprojectPSO || !m_NaiveTsrResolvePSO) return;

    const uint32_t outputW = frame.outputWidth;
    const uint32_t outputH = frame.outputHeight;

    int currentHistory = m_TaaHistoryIndex;
    int previousHistory = 1 - currentHistory;

    // ---- Pass 1: Reproject History ----
    {
        // Transition inputs to SRV
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_TaaHistoryTex[previousHistory], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_NrdMotionVectorsTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_GBuffer.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Transition outputs to UAV
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_TaaReprojectedHistoryTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_TaaClosestVelocityTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        m_CommandList->SetComputeRootSignature(m_RootSignature.Get());
        m_CommandList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
        m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
        m_CommandList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

        BindlessIndices indices = {};
        indices.InputIdx0 = m_TaaHistoryTex[previousHistory].srvIndex;
        indices.InputIdx1 = m_NrdMotionVectorsTex.srvIndex;
        indices.InputIdx2 = m_GBuffer.depth.srvIndex;
        indices.OutputIdx0 = m_TaaReprojectedHistoryTex.uavIndex;
        indices.OutputIdx1 = m_TaaClosestVelocityTex.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);

        m_CommandList->SetPipelineState(m_NaiveTsrReprojectPSO.Get());
        m_CommandList->Dispatch((outputW + 7) / 8, (outputH + 7) / 8, 1);

        // UAV barrier
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaReprojectedHistoryTex.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaClosestVelocityTex.resource.Get()),
        };
        m_CommandList->ResourceBarrier(2, barriers);
    }

    // ---- Pass 2: TAA Resolve ----
    {
        // Transition inputs to SRV
        GraphicsHelper::TransitionResource(m_CommandList.Get(), const_cast<GPUTexture&>(inputColor), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_TaaReprojectedHistoryTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_TaaClosestVelocityTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Transition outputs to UAV
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_TaaHistoryTex[currentHistory], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_TaaOutputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        BindlessIndices indices = {};
        indices.InputIdx0 = inputColor.srvIndex;
        indices.InputIdx1 = m_TaaReprojectedHistoryTex.srvIndex;
        indices.InputIdx2 = m_TaaClosestVelocityTex.srvIndex;
        indices.OutputIdx0 = m_TaaHistoryTex[currentHistory].uavIndex;
        indices.OutputIdx1 = m_TaaOutputTex.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);

        m_CommandList->SetPipelineState(m_NaiveTsrResolvePSO.Get());
        m_CommandList->Dispatch((outputW + 7) / 8, (outputH + 7) / 8, 1);

        // UAV barrier
        D3D12_RESOURCE_BARRIER barriers[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaHistoryTex[currentHistory].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_TaaOutputTex.resource.Get()),
        };
        m_CommandList->ResourceBarrier(2, barriers);
    }

    // Swap history index for next frame
    m_TaaHistoryIndex = previousHistory;
}