#include "pch.h"

#include "Renderer.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include <dxcapi.h>
#include <array>
#include <cassert>
#include <cstring>

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
    return m_Denoise.Initialize(m_Device.Get(), m_CommandQueue.Get(), m_InternalWidth, m_InternalHeight);
}

void Renderer::ShutdownNrd()
{
    m_Denoise.Shutdown();
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

    // ------- SHaRC + split diffuse/specular buffers -------
    m_RestirGI.CreateResources(m_InternalWidth, m_InternalHeight);

    CreateTexture(m_RasterHdrOutputTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, 1, 1, "Tex_RasterHdrOutput");
    // NRD/Denoise resources
    m_Denoise.CreateResources(m_InternalWidth, m_InternalHeight);
    // Universal interchange textures: SSO writes, NrdPackNoise+Lighting read
    CreateTexture(m_FinalDiffuseTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_FinalDiffuse");
    CreateTexture(m_FinalSpecularTex, m_InternalWidth, m_InternalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_FinalSpecular");

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

    // ------- SHaRC + split diffuse/specular PSOs -------
    m_RestirGI.CreatePipelines(m_Device.Get(), m_RootSignature.Get());

    auto nrdStoreSSO_CS   = GraphicsHelper::CompileShader("Shaders/NrdStoreShadingOutput.hlsl",         "main", "cs_6_6");

    // NRD PSOs
    m_Denoise.CreatePipelines(m_Device.Get(), m_RootSignature.Get());

    if (!nrdStoreSSO_CS.empty()) {
        computeDesc.CS = { nrdStoreSSO_CS.data(), nrdStoreSSO_CS.size() };
        m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdStoreShadingOutputPSO));
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
            desc.DepthStencilState.DepthFunc             = D3D12_COMPARISON_FUNC_GREATER_EQUAL; // Reverse-Z
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
    m_RestirDI.CreateResources(m_InternalWidth, m_InternalHeight);
}

void Renderer::CreateRestirDIPipelines()
{
    m_RestirDI.CreatePipelines(m_Device.Get(), m_RootSignature.Get());
}

void Renderer::DispatchRestirDI(class Model* model, const FrameConstants& frame)
{
    if (!frame.enableRestirDI) return;

    m_RestirDI.Execute(m_CommandList.Get(), m_RootSignature.Get(), model, frame,
                        m_FrameCB.gpuAddress, m_AccelStructure.GetTLASGPUAddress(),
                        m_DeferredLighting.GetLightsBufferGPUAddress(), m_DeferredLighting.GetLightLUTBufferGPUAddress(),
                        m_FullScreenDebugTex, m_FinalDiffuseTex, m_FinalSpecularTex,
                        m_NrdStoreShadingOutputPSO.Get(),
                        m_InternalWidth, m_InternalHeight);

    const bool diDebugActive = frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF;

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

    // DI debug: FullScreenDebugTex UAV → SRV (only when GI is off;
    // when GI is on, DispatchRestirGI handles the SRV transition).
    if (diDebugActive && !frame.enableRasterIndirectGI)
    {
        D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(m_FullScreenDebugTex.resource.Get());
        m_CommandList->ResourceBarrier(1, &b);
        GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FullScreenDebugTex,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
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

    CreateMeshletPipelines();

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

    // Pin GPU clocks for stable profiling/debugging timestamps. Requires Windows
    // Developer Mode or an elevated process; harmless warning otherwise.
    // Call once, right after device creation (per Microsoft docs).
    {
        HRESULT hrStable = m_Device->SetStablePowerState(TRUE);
        if (FAILED(hrStable))
            std::cerr << "[Renderer] SetStablePowerState failed (0x" << std::hex << hrStable << std::dec
                      << ") — enable Developer Mode or run elevated; GPU clocks will boost normally" << std::endl;
    }

    // QI to ID3D12Device2 for CreatePipelineState (pipeline state streams, required for Mesh Shader PSOs)
    CHECK_HR(m_Device->QueryInterface(IID_PPV_ARGS(&m_Device2)), "QueryInterface ID3D12Device2 failed");

    // Check for Ray Tracing support
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    if (SUCCEEDED(m_Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5))))
    {
        m_RayTracingSupported = (options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
    }
    std::cout << "Ray Tracing Supported: " << (m_RayTracingSupported ? "Yes" : "No") << std::endl;

    // Check for Mesh Shader support
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = {};
        if (SUCCEEDED(m_Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7))))
        {
            m_MeshShaderSupported = (options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
        }
    }
    std::cout << "Mesh Shader Supported: " << (m_MeshShaderSupported ? "Yes" : "No") << std::endl;

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    CHECK_HR(m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_CommandQueue)), "CreateCommandQueue failed");

    // Create copy queue
    D3D12_COMMAND_QUEUE_DESC copyQueueDesc = {};
    copyQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    copyQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    CHECK_HR(m_Device->CreateCommandQueue(&copyQueueDesc, IID_PPV_ARGS(&m_CopyQueue)), "CreateCommandQueue (copy) failed");

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
    if (!CreateBuffer(m_FrameCB, (sizeof(FrameConstants) + 255) & ~255, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_FrameConstants"))
    {
        std::cerr << "Failed to create frame constant buffer" << std::endl;
        return false;
    }

    // Frozen snapshot bound to cull dispatches while freeze culling is enabled
    if (!CreateBuffer(m_CullFrameCB, (sizeof(FrameConstants) + 255) & ~255, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "CB_CullFrameConstants"))
    {
        std::cerr << "Failed to create cull frame constant buffer" << std::endl;
        return false;
    }

    // Create GBuffer — initially at WINDOW_WIDTH x WINDOW_HEIGHT.
    // Will be recreated at internal resolution by CreateInternalResolutionResources().
    CreateGBuffer(WINDOW_WIDTH, WINDOW_HEIGHT);

    // Create Shadow Map
    if (!m_Shadow.CreateResources())
    {
        return false;
    }

    // Create Path Tracer / ReSTIR GI / RTXDI resources
    if (m_RayTracingSupported)
    {
        if (!m_PathTracing.CreateResources(m_Device.Get(), m_RayTracingSupported, m_InternalWidth, m_InternalHeight))
        {
            std::cerr << "Failed to create path tracing resources" << std::endl;
            return false;
        }

        if (!CreateTexture(m_FullScreenDebugTex, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_FullScreenDebug"))
        {
            std::cerr << "Failed to create full-screen debug texture" << std::endl;
            return false;
        }
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

    // Meshlet pipeline resources and PSOs
    CreateMeshletResources();
    CreateMeshletPipelines();

    // GPU on-screen debug text renderer
    if (!m_DebugTextRenderer.Initialize(m_Device.Get(), m_CommandQueue.Get(), m_RootSignature.Get()))
    {
        std::cerr << "Failed to initialize debug text renderer" << std::endl;
        return false;
    }

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

    m_AccelStructure.Reset();
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

    // Bind the command list to the microprofile GPU context so MICROPROFILE_SCOPEGPUI works
    MICROPROFILE_GPU_SET_CONTEXT(m_CommandList.Get(), MicroProfileGetGlobalGpuThreadLog());

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
    m_CommandList->SetGraphicsRootShaderResourceView(4, m_AccelStructure.GetTLASGPUAddress());

    // Bind Lights Buffer (t0, space2) - root parameter 10
    m_CommandList->SetGraphicsRootShaderResourceView(10, m_DeferredLighting.GetLightsBufferGPUAddress());
    m_CommandList->SetGraphicsRootShaderResourceView(11, m_DeferredLighting.GetLightLUTBufferGPUAddress()); // Light LUT (t1, space2)

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

    CD3DX12_DESCRIPTOR_RANGE srvRangeMeshletSpace3;
    srvRangeMeshletSpace3.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 16, 0, 3); // t0-t15 space3: Meshlet stream buffers

    CD3DX12_ROOT_PARAMETER rootParameters[15];
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
    rootParameters[12].InitAsConstants(sizeof(BinningParams) / 4, 1, 0); // b1: Bindless/Binning/Raster indices (max of all pass params)
    rootParameters[13].InitAsConstants(sizeof(IrCacheBindlessIndices) / 4, 2, 0); // b2: IrCache bindless indices
    rootParameters[14].InitAsDescriptorTable(1, &srvRangeMeshletSpace3); // t0-t15 space3: Meshlet streams

    CD3DX12_STATIC_SAMPLER_DESC samplers[2];
    samplers[0].Init(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);
    samplers[1].Init(1, D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR, 
        D3D12_TEXTURE_ADDRESS_MODE_BORDER, D3D12_TEXTURE_ADDRESS_MODE_BORDER, D3D12_TEXTURE_ADDRESS_MODE_BORDER);
    samplers[1].ComparisonFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL; // Reverse-Z: closer = larger depth
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

    // Depth PrePass + GBuffer
    m_GBufferPass.CreatePipelines(m_Device.Get(), m_RootSignature.Get());
    
    // Transparency
    m_Transparency.CreatePipelines(m_Device.Get(), m_RootSignature.Get());

    // Shadow
    m_Shadow.CreatePipelines(m_Device.Get(), m_RootSignature.Get());

    // Lighting
    m_DeferredLighting.CreatePipelines(m_Device.Get(), m_RootSignature.Get());

    // 3.15 FullScreenDebug PSO (LDR — renders to R8G8B8A8_UNORM back buffer with tonemapping)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/FullScreenDebug.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/FullScreenDebug.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_FullScreenDebugPSO));
    }

    // 3.16 FullScreenDebug HDR PSO (renders to R16G16B16A16_FLOAT for TAA input — no tonemapping)
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/FullScreenDebug.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/FullScreenDebug.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_FullScreenDebugHdrPSO));
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

    if (m_RayTracingSupported)
    {
        CreateRayTracingPipeline();
    }

    std::cout << "Pipeline states created successfully" << std::endl;
}

void Renderer::CreateRayTracingPipeline()
{
    m_PathTracing.CreatePipelines(m_Device.Get(), m_RootSignature.Get());
}

void Renderer::DispatchRays(Model* model, const FrameConstants& frame, const LightConstants& light)
{
    if (!model) return;

    // Update constant buffers
    memcpy(m_FrameCB.cpuPtr, &frame, sizeof(FrameConstants));

    m_PathTracing.DispatchRays(m_CommandList.Get(), m_RootSignature.Get(), model, frame,
                                m_FrameCB.gpuAddress, m_AccelStructure.GetTLASGPUAddress(),
                                m_DeferredLighting.GetLightsBufferGPUAddress(), m_DeferredLighting.GetLightLUTBufferGPUAddress(),
                                m_InternalWidth, m_InternalHeight);
}

void Renderer::DispatchRestirGI(class Model* model, const FrameConstants& frame)
{
    if (!frame.enableRasterIndirectGI)
    {
        // Reset the flag: NRD did not run from the GI path this frame.
        // If DI is also active with NRD, DispatchRestirDI will set it back to true.
        m_Denoise.SetWasActiveLastFrame(false);
        return;
    }

    int currentReservoir = m_PathTracing.GetCurrentReservoirIndex();
    int previousReservoir = 1 - currentReservoir;

    const bool sharcDebugRan = m_RestirGI.Execute(
        m_CommandList.Get(), m_RootSignature.Get(), model, frame,
        m_FrameCB.gpuAddress, m_AccelStructure.GetTLASGPUAddress(),
        m_DeferredLighting.GetLightsBufferGPUAddress(), m_DeferredLighting.GetLightLUTBufferGPUAddress(),
        m_GBufferPass.GetGBuffer(), m_FullScreenDebugTex, m_FinalDiffuseTex, m_FinalSpecularTex,
        m_NrdStoreShadingOutputPSO.Get(),
        currentReservoir, previousReservoir,
        m_InternalWidth, m_InternalHeight);

    if (sharcDebugRan)
    {
        m_Denoise.SetWasActiveLastFrame(false);
        m_PathTracing.SetCurrentReservoirIndex(previousReservoir);
        return;
    }

    const bool useNrd = frame.enableNrdRelax != 0
        && frame.restirReservoirDebugMode == RESTIR_RESERVOIR_DEBUG_OFF
        && frame.sharcDebug == 0;

    if (useNrd && NRDDenoise(frame))
    {
        m_PathTracing.SetCurrentReservoirIndex(previousReservoir);
        return;
    }

    // NRD path was not taken this frame — record so next activation can RESTART.
    m_Denoise.SetWasActiveLastFrame(false);

    // NRD disabled: transition Final* to SRV for Lighting.hlsl
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    m_PathTracing.SetCurrentReservoirIndex(previousReservoir); // Swap for next frame
}

bool Renderer::NRDDenoise(const FrameConstants& frame)
{
    if (!InitializeNrd())
        return false;

    return m_Denoise.Execute(m_CommandList.Get(), m_CommandAllocator.Get(), m_RootSignature.Get(),
                              m_FrameCB.gpuAddress, frame, m_FinalDiffuseTex, m_FinalSpecularTex,
                              m_InternalWidth, m_InternalHeight);
}

void Renderer::DrawPathVizLines(const FrameConstants& frame)
{
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = GetCurrentBackBufferRTV();
    m_PathTracing.DrawPathVizLines(m_CommandList.Get(), m_RootSignature.Get(), m_FrameCB.gpuAddress, rtvHandle);
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

    // Reset command list for AS build
    m_CommandAllocator->Reset();
    m_CommandList->Reset(m_CommandAllocator.Get(), nullptr);

    // Build() closes/submits/waits internally (via the callback) before its
    // stack-local scratch buffers go out of scope.
    size_t instanceCount = m_AccelStructure.Build(m_Device.Get(), m_CommandList.Get(), model, this);

    std::cout << "Built acceleration structures for " << instanceCount << " instances." << std::endl;
}

void Renderer::CreateGBuffer(uint32_t w, uint32_t h)
{
    m_GBufferPass.CreateResources(w, h);
}

void Renderer::ExecuteGBufferPass(Model* model, const DirectX::BoundingFrustum& frustum, bool enableDepthPrePass)
{
    m_GBufferPass.Execute(m_CommandList.Get(), model, this, frustum, enableDepthPrePass);
}

void Renderer::ExecuteLightingPass(Model* model, const FrameConstants& frame, bool rasterTaaActive,
                                    bool debugActive, bool debugShadowMap, uint32_t outputWidth, uint32_t outputHeight)
{
    m_DeferredLighting.Execute(m_CommandList.Get(), this, model, frame, rasterTaaActive, debugActive,
                                debugShadowMap, outputWidth, outputHeight);
}

void Renderer::ExecuteTransparencyPass(Model* model, const DirectX::BoundingFrustum& frustum,
                                        bool rasterTaaActive, uint32_t outputWidth, uint32_t outputHeight)
{
    m_Transparency.Execute(m_CommandList.Get(), model, this, frustum, rasterTaaActive, outputWidth, outputHeight);
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

void Renderer::UpdateCullFrameCB(const FrameConstants& frameConstants)
{
    memcpy(m_CullFrameCB.cpuPtr, &frameConstants, sizeof(FrameConstants));
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

    // ---- Path Tracer Output Textures (Rendering/PathTracing.h/.cpp) ----
    if (m_RayTracingSupported)
    {
        m_PathTracing.OnResolutionChanged(w, h);

        CreateTexture(m_FullScreenDebugTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_FullScreenDebug");
    }

    // ---- SHaRC + split diffuse/specular ReSTIR resources (Rendering/RestirGI.h/.cpp) ----
    m_RestirGI.CreateResources(w, h);

    // ---- ReSTIR DI buffers/textures (Rendering/RestirDI.h/.cpp) ----
    m_RestirDI.CreateResources(w, h);

    // ---- Raster HDR + NRD Textures ----
    // HDR render target for rasterizer lighting when TAA is active
    CreateTexture(m_RasterHdrOutputTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, 1, 1, "Tex_RasterHdrOutput");
    // NRD/Denoise resolution-dependent resources (Rendering/Denoise.h/.cpp)
    m_Denoise.OnResolutionChanged(w, h);
    // Universal interchange textures: SSO writes, NrdPackNoise+Lighting read
    CreateTexture(m_FinalDiffuseTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_FinalDiffuse");
    CreateTexture(m_FinalSpecularTex, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_FinalSpecular");

    // ---- Visibility buffer (meshlet debug overlay R32_UINT) + HZB ----
    m_Meshlet.RecreateVisibilityBuffer(w, h);
    m_GPUCulling.RecreateHZB(w, h);

    // ---- Re-initialize NRD at new resolution ----
    if (m_Denoise.IsInitialized())
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
    m_TAA.CreateResources(outputW, outputH, internalW, internalH);
}

void Renderer::CreateTaaPipelines()
{
    m_TAA.CreatePipelines(m_Device.Get(), m_RootSignature.Get());
}

void Renderer::GenerateMotionVectors(const FrameConstants& frame)
{
    m_TAA.GenerateMotionVectors(m_CommandList.Get(), m_RootSignature.Get(), m_FrameCB.gpuAddress,
                                 m_GBufferPass.GetGBuffer(), m_Denoise.GetMotionVectorsTex(), m_InternalWidth, m_InternalHeight);
}

void Renderer::DispatchNaiveTsr(const FrameConstants& frame, const GPUTexture& inputColor)
{
    m_TAA.Execute(m_CommandList.Get(), m_RootSignature.Get(), m_FrameCB.gpuAddress, frame,
                  inputColor, m_GBufferPass.GetGBuffer(), m_Denoise.GetMotionVectorsTex());
}

// =============================================================================
// Meshlet Pipeline — thin delegation to GPUCulling + Meshlet
// =============================================================================

void Renderer::CreateMeshletResources()
{
    m_Meshlet.CreateResources(m_InternalWidth, m_InternalHeight);
    m_GPUCulling.CreateResources(m_InternalWidth, m_InternalHeight);
}

void Renderer::CreateMeshletPipelines()
{
    m_GPUCulling.CreatePipelines(m_Device.Get(), m_RootSignature.Get());
    m_Meshlet.CreatePipelines(m_Device.Get(), m_Device2.Get(), m_RootSignature.Get(), m_MeshShaderSupported);
}

void Renderer::DispatchMeshletTwoPassCull(Model* model, const FrameConstants& frame,
                                           bool occlusionEnabled, int phase, bool freezeCulling)
{
    // Freeze culling: cull dispatches read the frozen snapshot (m_CullFrameCB)
    // while binning/rasterize keep the live m_FrameCB.
    D3D12_GPU_VIRTUAL_ADDRESS cullFrameAddress = freezeCulling ? m_CullFrameCB.gpuAddress : m_FrameCB.gpuAddress;
    m_GPUCulling.CullTwoPass(m_CommandList.Get(), cullFrameAddress, model, occlusionEnabled, phase);
}

void Renderer::DispatchMeshletBinning()
{
    m_Meshlet.Binning(m_CommandList.Get(), m_RootSignature.Get(), m_FrameCB.gpuAddress,
                       m_GPUCulling.GetVisibleMeshletsSRVIndex(),
                       m_GPUCulling.GetVisibleMeshletsCounterSRVIndex(),
                       m_GPUCulling.GetVisibleMeshletsCounterUAVIndex());
}

void Renderer::DispatchMeshletRasterize(Model* model)
{
    m_Meshlet.Rasterize(m_CommandList.Get(), m_RootSignature.Get(), m_FrameCB.gpuAddress, model,
                         m_GPUCulling.GetVisibleMeshletsSRVIndex());
}

