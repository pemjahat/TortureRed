#include "pch.h"

#include "Renderer.h"
#include "Model.h"
#include "Utility.h"
#include <dxcapi.h>
#include <cassert>
#include "Rtxdi/GI/ReSTIRGIParameters.h"
#include "Rtxdi/RtxdiUtils.h"

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

void Renderer::CreateRasterIndirectGIResources()
{
    // -----------------------------------------------------------------------
    // Spatial irradiance cache buffers
    // -----------------------------------------------------------------------
    constexpr UINT MAX_ENTRIES  = 32768;
    constexpr UINT TOTAL_CELLS  = 262144;    // 32^3 * 8 cascades

    // Helper lambdas for creating raw (ByteAddressBuffer) and structured UAV buffers

    // Raw (RWByteAddressBuffer) — CreateBuffer default UAV is already RAW
    CreateBuffer(m_IrCacheMetaBuf,     16ULL,               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);
    CreateBuffer(m_IrCacheGridMetaBuf, TOTAL_CELLS * 8ULL,  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);
    CreateBuffer(m_IrCacheLifeBuf,     MAX_ENTRIES * 4ULL,  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);
    CreateBuffer(m_IrCacheTraceArgsBuf,12ULL,               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, true);

    // Structured (RWStructuredBuffer<T>) — CreateStructuredBuffer now writes correct UAV
    CreateStructuredBuffer(m_IrCachePoolBuf,       sizeof(UINT),      MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_IrCacheEntryCellBuf,  sizeof(UINT),      MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_IrCacheIrradianceBuf, sizeof(float) * 4, MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_IrCacheIndirectionBuf,sizeof(UINT),      MAX_ENTRIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Fill the IrCacheBindlessIndices struct (all UAV indices)
    m_IrCacheIndices.MetaBufIdx        = (UINT)m_IrCacheMetaBuf.uavIndex;
    m_IrCacheIndices.PoolBufIdx        = (UINT)m_IrCachePoolBuf.uavIndex;
    m_IrCacheIndices.GridMetaBufIdx    = (UINT)m_IrCacheGridMetaBuf.uavIndex;
    m_IrCacheIndices.EntryCellBufIdx   = (UINT)m_IrCacheEntryCellBuf.uavIndex;
    m_IrCacheIndices.IrradianceBufIdx  = (UINT)m_IrCacheIrradianceBuf.uavIndex;
    m_IrCacheIndices.LifeBufIdx        = (UINT)m_IrCacheLifeBuf.uavIndex;
    m_IrCacheIndices.IndirectionBufIdx = (UINT)m_IrCacheIndirectionBuf.uavIndex;
    m_IrCacheIndices.TraceArgsBufIdx   = (UINT)m_IrCacheTraceArgsBuf.uavIndex;

    // ReSTIR reservoir buffers (unchanged)
    CreateStructuredBuffer(m_RasterReservoirs[0], sizeof(Reservoir), WINDOW_WIDTH * WINDOW_HEIGHT, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_RasterReservoirs[1], sizeof(Reservoir), WINDOW_WIDTH * WINDOW_HEIGHT, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CreateStructuredBuffer(m_RasterReservoirIntermediate, sizeof(Reservoir), WINDOW_WIDTH * WINDOW_HEIGHT, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    
    CreateTexture(m_RasterIndirectLightingTex, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void Renderer::CreateRasterIndirectGIPipelines()
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
    computeDesc.pRootSignature = m_RootSignature.Get();

    auto CompileAndCreate = [&](const char* file, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso)
    {
        auto cs = CompileShader(file, "main", "cs_6_6");
        if (!cs.empty())
        {
            computeDesc.CS = { cs.data(), cs.size() };
            m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&pso));
        }
    };

    CompileAndCreate("Shaders/IrCache_Pool_Init.hlsl",      m_IrCachePoolInitPSO);
    CompileAndCreate("Shaders/IrCache_Prepare_Age.hlsl",    m_IrCachePrepareAgePSO);
    CompileAndCreate("Shaders/IrCache_Age.hlsl",            m_IrCacheAgePSO);
    CompileAndCreate("Shaders/IrCache_Prepare_Trace.hlsl",  m_IrCachePrepareTracePSO);
    CompileAndCreate("Shaders/IrCache_Update.hlsl",         m_IrCacheUpdatePSO);

    auto restirTemporalCS = CompileShader("Shaders/RestirGI_Raster_Temporal.hlsl", "main", "cs_6_6");
    auto restirSpatialCS  = CompileShader("Shaders/RestirGI_Raster_Spatial.hlsl",  "main", "cs_6_6");
    auto restirResolveCS  = CompileShader("Shaders/RestirGI_Raster_Resolve.hlsl",  "main", "cs_6_6");

    computeDesc.CS = { restirTemporalCS.data(), restirTemporalCS.size() };
    m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_RestirGIRasterTemporalPSO));

    computeDesc.CS = { restirSpatialCS.data(), restirSpatialCS.size() };
    m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_RestirGIRasterSpatialPSO));

    computeDesc.CS = { restirResolveCS.data(), restirResolveCS.size() };
    m_Device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_RestirGIRasterResolvePSO));

    // Command signature for ExecuteIndirect dispatch (used by IrCache_Update indirect pass)
    D3D12_INDIRECT_ARGUMENT_DESC dispatchArg = {};
    dispatchArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    D3D12_COMMAND_SIGNATURE_DESC csSigDesc = {};
    csSigDesc.ByteStride       = sizeof(D3D12_DISPATCH_ARGUMENTS);
    csSigDesc.NumArgumentDescs = 1;
    csSigDesc.pArgumentDescs   = &dispatchArg;
    m_Device->CreateCommandSignature(&csSigDesc, nullptr, IID_PPV_ARGS(&m_DispatchCommandSignature));
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

    // Create descriptor heap for render target views
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = 16;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    CHECK_HR(m_Device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_RTVHeap)), "CreateDescriptorHeap for RTV failed");

    UINT rtvDescriptorSize = m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Create render target view for each frame
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_RTVHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < 2; n++)
    {
        CHECK_HR(m_SwapChain->GetBuffer(n, IID_PPV_ARGS(&m_RenderTargets[n])), "GetBuffer failed");
        m_Device->CreateRenderTargetView(m_RenderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.ptr += rtvDescriptorSize;
    }

    // Create DSV descriptor heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
        dsvHeapDesc.NumDescriptors = 4;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        CHECK_HR(m_Device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_DSVHeap)), "CreateDescriptorHeap for DSV failed");
    }

    // Create constant buffers
    if (!CreateBuffer(m_FrameCB, (sizeof(FrameConstants) + 255) & ~255, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
    {
        std::cerr << "Failed to create frame constant buffer" << std::endl;
        return false;
    }

    // Create SRV descriptor heap for textures
    {
        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
        srvHeapDesc.NumDescriptors = 4096; // Increased from 1024
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        CHECK_HR(m_Device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_SRVHeap)), "CreateDescriptorHeap for SRV failed");
    }

    // Create GBuffer
    CreateGBuffer();

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

        if (!CreateTexture(m_PathTracerOutput, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr))
        {
            std::cerr << "Failed to create path tracer output texture" << std::endl;
            return false;
        }

        // Create ReSTIR Reservoirs
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

        // Create RTXDI Reservoirs
        // See RtxdiUtils.cpp: CalculateReservoirBufferParameters
        uint32_t renderWidthBlocks = (WINDOW_WIDTH + 15) / 16;
        uint32_t renderHeightBlocks = (WINDOW_HEIGHT + 15) / 16;
        uint32_t reservoirArrayPitch = renderWidthBlocks * 256 * renderHeightBlocks;

        for (int i = 0; i < 2; ++i)
        {
            if (!CreateStructuredBuffer(m_RtxdiReservoirBuffer[i], sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
            {
                std::cerr << "Failed to create RTXDI reservoir buffer " << i << std::endl;
                return false;
            }
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
        m_RtxdiNeighborOffsetsBuffer.srvIndex = AllocateDescriptor();
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R8G8_SNORM;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = neighborOffsetCount;
        srvDesc.Buffer.StructureByteStride = 0;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

        m_Device->CreateShaderResourceView(m_RtxdiNeighborOffsetsBuffer.resource.Get(), &srvDesc, GetCPUDescriptorHandle(m_RtxdiNeighborOffsetsBuffer.srvIndex));
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
    ID3D12DescriptorHeap* heaps[] = { m_SRVHeap.Get() };
    m_CommandList->SetDescriptorHeaps(_countof(heaps), heaps);

    // Set necessary state
    m_CommandList->SetGraphicsRootSignature(m_RootSignature.Get());

    // Bind the global descriptor table (bindless)
    m_CommandList->SetGraphicsRootDescriptorTable(3, m_SRVHeap->GetGPUDescriptorHandleForHeapStart());

    // Set Frame constant buffer (viewProj)
    m_CommandList->SetGraphicsRootConstantBufferView(0, m_FrameCB.gpuAddress);

    // Bind TLAS for ray-traced shadows in pixel shader
    m_CommandList->SetGraphicsRootShaderResourceView(4, m_TLAS.gpuAddress);

    // Bind Lights Buffer (t0, space2) - root parameter 10
    m_CommandList->SetGraphicsRootShaderResourceView(10, m_LightsBuffer.gpuAddress);
    m_CommandList->SetGraphicsRootShaderResourceView(11, m_LightLUTBuffer.gpuAddress); // Light LUT (t1, space2)

    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(WINDOW_WIDTH), static_cast<float>(WINDOW_HEIGHT));
    D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    m_CommandList->RSSetViewports(1, &viewport);
    m_CommandList->RSSetScissorRects(1, &scissorRect);

    m_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

void Renderer::EndFrame()
{
    // Transition back buffer to present state
    TransitionResource(m_RenderTargets[m_FrameIndex].Get(), m_BackBufferStates[m_FrameIndex], D3D12_RESOURCE_STATE_PRESENT);

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
    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_RTVHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += (UINT64)m_FrameIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    return handle;
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
        std::vector<char> vs = CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.NumRenderTargets = 0;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_DepthPrePassPSO));
    }

    // 2. G-Buffer PSO
    {
        std::vector<char> vs = CompileShader("Shaders/Gbuffer.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = CompileShader("Shaders/Gbuffer.hlsl", "PSMain", "ps_6_8");
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

    // 3. Lighting PSO
    {
        std::vector<char> vs = CompileShader("Shaders/Lighting.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = CompileShader("Shaders/Lighting.hlsl", "PSMain", "ps_6_8");
        auto psoDesc = GetDefaultPsoDesc();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_LightingPSO));
    }

    // 3.5 Debug PSO
    {
        std::vector<char> vs = CompileShader("Shaders/DebugShadow.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = CompileShader("Shaders/DebugShadow.hlsl", "PSMain", "ps_6_8");
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
        std::vector<char> vs = CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");
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
        std::vector<char> vs = CompileShader("Shaders/Forward.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = CompileShader("Shaders/Forward.hlsl", "PSMain", "ps_6_8");
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
    auto pathTracerCode = CompileShader("Shaders/PathTracer.hlsl", "CSMain", "cs_6_6");
    if (!pathTracerCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { pathTracerCode.data(), pathTracerCode.size() };
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_PathTracerPSO)), "Failed to create Path Tracer Compute PSO");
    }

    // Load ReSTIR Multi-pass shaders
    auto restirTemporalCode = CompileShader("Shaders/RestirGI_Temporal.hlsl", "CSMain", "cs_6_6");
    if (!restirTemporalCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { restirTemporalCode.data(), restirTemporalCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirTemporalPSO)), "Failed to create ReSTIR Temporal PSO");
    }

    auto restirSpatialCode = CompileShader("Shaders/RestirGI_Spatial.hlsl", "CSMain", "cs_6_6");
    if (!restirSpatialCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { restirSpatialCode.data(), restirSpatialCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirSpatialPSO)), "Failed to create ReSTIR Spatial PSO");
    }

    auto restirResolveCode = CompileShader("Shaders/RestirGI_Resolve.hlsl", "CSMain", "cs_6_6");
    if (!restirResolveCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { restirResolveCode.data(), restirResolveCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirResolvePSO)), "Failed to create ReSTIR Resolve PSO");
    }

    // RTXDI PSOs
    auto rtxdiTemporalCode = CompileShader("Shaders/RestirGI_RTXDI_Temporal.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiTemporalCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { rtxdiTemporalCode.data(), rtxdiTemporalCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirTemporalPSO)), "Failed to create RTXDI Temporal PSO");
    }

    auto rtxdiSpatialCode = CompileShader("Shaders/RestirGI_RTXDI_Spatial.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiSpatialCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { rtxdiSpatialCode.data(), rtxdiSpatialCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirSpatialPSO)), "Failed to create RTXDI Spatial PSO");
    }

    auto rtxdiResolveCode = CompileShader("Shaders/RestirGI_RTXDI_Resolve.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiResolveCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.CS = { rtxdiResolveCode.data(), rtxdiResolveCode.size() };
        CHECK_HR(m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirResolvePSO)), "Failed to create RTXDI Resolve PSO");
    }
}

void Renderer::DispatchRays(Model* model, const FrameConstants& frame, const LightConstants& light)
{
    if (!model) return;

    // Update constant buffers
    memcpy(m_FrameCB.cpuPtr, &frame, sizeof(FrameConstants));

    // Transition UAVs
    TransitionResource(m_AccumulationBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_PathTracerOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_ReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_ReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_ReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_RtxdiReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_RtxdiReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    D3D12_RESOURCE_BARRIER uavBarriers[5];
    uavBarriers[0] = CD3DX12_RESOURCE_BARRIER::UAV(m_AccumulationBuffer.resource.Get());
    uavBarriers[1] = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
    uavBarriers[2] = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirIntermediate.resource.Get());
    uavBarriers[3] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[0].resource.Get());
    uavBarriers[4] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[1].resource.Get());
    m_CommandList->ResourceBarrier(5, uavBarriers);

    m_CommandList->SetDescriptorHeaps(1, m_SRVHeap.GetAddressOf());
    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());

    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    m_CommandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(0)); // Bindless
    m_CommandList->SetComputeRootShaderResourceView(4, m_TLAS.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(10, m_LightsBuffer.gpuAddress); // Lights Buffer
    m_CommandList->SetComputeRootShaderResourceView(11, m_LightLUTBuffer.gpuAddress); // Light LUT Buffer

    BindlessIndices indices;

    int currentReservoir = m_CurrentReservoirIndex;
    int previousReservoir = 1 - currentReservoir;

    if (frame.useRTXDI)
    {
        // NVIDIA RTXDI Path
        // Bind common RTXDI resources
        m_CommandList->SetComputeRootDescriptorTable(9, GetGPUDescriptorHandle(m_RtxdiNeighborOffsetsBuffer.srvIndex));

        // Pass 1: Temporal Resampling
        m_CommandList->SetPipelineState(m_RtxdiRestirTemporalPSO.Get());
        m_CommandList->SetComputeRootDescriptorTable(7, GetGPUDescriptorHandle(m_RtxdiReservoirBuffer[currentReservoir].uavIndex));
        m_CommandList->SetComputeRootDescriptorTable(8, GetGPUDescriptorHandle(m_RtxdiReservoirBuffer[previousReservoir].uavIndex));
        m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier1 = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[currentReservoir].resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier1);

        // Pass 2: Spatial Resampling
        m_CommandList->SetPipelineState(m_RtxdiRestirSpatialPSO.Get());
        m_CommandList->SetComputeRootDescriptorTable(7, GetGPUDescriptorHandle(m_ReservoirIntermediate.uavIndex));
        m_CommandList->SetComputeRootDescriptorTable(8, GetGPUDescriptorHandle(m_RtxdiReservoirBuffer[currentReservoir].uavIndex));
        m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirIntermediate.resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier2);

        // Pass 3: Resolve
        m_CommandList->SetPipelineState(m_RtxdiRestirResolvePSO.Get());
        m_CommandList->SetComputeRootDescriptorTable(7, GetGPUDescriptorHandle(m_ReservoirIntermediate.uavIndex));
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices        
        m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);
    }
    else if (frame.enableRestir)
    {
        // Torture ReSTIR (Manual Implementation)
        // Pass 1: Temporal — writes to ReservoirBuffer[current], reads history from ReservoirBuffer[previous]
        m_CommandList->SetPipelineState(m_RestirTemporalPSO.Get());
        indices.InputIdx0 = m_ReservoirBuffer[previousReservoir].srvIndex;
        indices.OutputIdx0 = m_ReservoirBuffer[currentReservoir].uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier1 = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirBuffer[currentReservoir].resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier1);

        // Pass 2: Spatial — writes to Intermediate, reads temporal from ReservoirBuffer[current]
        m_CommandList->SetPipelineState(m_RestirSpatialPSO.Get());
        indices.InputIdx0 = m_ReservoirBuffer[currentReservoir].srvIndex;
        indices.OutputIdx0 = m_ReservoirIntermediate.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirIntermediate.resource.Get());
        m_CommandList->ResourceBarrier(1, &barrier2);

        // Pass 3: Resolve — reads spatial output from Intermediate
        m_CommandList->SetPipelineState(m_RestirResolvePSO.Get());
        indices.InputIdx0 = m_ReservoirIntermediate.srvIndex;
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);
    }
    else
    {
        // Old Path Trace
        m_CommandList->SetPipelineState(m_PathTracerPSO.Get());
        m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);
    }

    m_CurrentReservoirIndex = previousReservoir; // Swap for next frame

    // Transition for blitting/Imgui
    TransitionResource(m_PathTracerOutput, D3D12_RESOURCE_STATE_COPY_SOURCE);
}

void Renderer::DispatchRasterIndirectGI(class Model* model, const FrameConstants& frame)
{
    if (!frame.enableRasterIndirectGI)
        return;

    // Transition G-Buffer targets to SRV state for compute
    TransitionResource(m_GBuffer.albedo, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    TransitionResource(m_GBuffer.normal, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    TransitionResource(m_GBuffer.material, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    TransitionResource(m_GBuffer.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    TransitionResource(m_RasterIndirectLightingTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_RasterReservoirs[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_RasterReservoirs[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_RasterReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    m_CommandList->SetDescriptorHeaps(1, m_SRVHeap.GetAddressOf());
    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());

    // Bind common resources
    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    m_CommandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(0)); // Bindless
    m_CommandList->SetComputeRootShaderResourceView(4, m_TLAS.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(10, m_LightsBuffer.gpuAddress); // Lights Buffer
    m_CommandList->SetComputeRootShaderResourceView(11, m_LightLUTBuffer.gpuAddress); // Light LUT Buffer

    BindlessIndices indices;

    int currentReservoir = m_CurrentReservoirIndex;
    int previousReservoir = 1 - currentReservoir;

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

        D3D12_RESOURCE_BARRIER initBarriers[4] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheGridMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCachePoolBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheLifeBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(4, initBarriers);
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

    // --- Pass 2: Age (expire old entries, build indirection) ---
    m_CommandList->SetPipelineState(m_IrCacheAgePSO.Get());
    m_CommandList->Dispatch((32768 + 63) / 64, 1, 1);

    {
        D3D12_RESOURCE_BARRIER barriers[4] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheLifeBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheIndirectionBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheGridMetaBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(4, barriers);
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
        D3D12_RESOURCE_BARRIER barriers[4] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheIrradianceBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheLifeBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCacheGridMetaBuf.resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_IrCachePoolBuf.resource.Get()),
        };
        m_CommandList->ResourceBarrier(4, barriers);
    }

    // -----------------------------------------------------------------------
    // ReSTIR passes  (temporal → spatial → resolve)
    // -----------------------------------------------------------------------
    // Restir Temporal — InputIdx0 no longer needed for IrCache; IrCache is at b2
    m_CommandList->SetPipelineState(m_RestirGIRasterTemporalPSO.Get());
    indices.InputIdx0  = m_RasterReservoirs[previousReservoir].srvIndex;
    indices.OutputIdx0 = m_RasterReservoirs[currentReservoir].uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER barrier1 = CD3DX12_RESOURCE_BARRIER::UAV(m_RasterReservoirs[currentReservoir].resource.Get());
    m_CommandList->ResourceBarrier(1, &barrier1);

    // Restir Spatial
    // writes to Intermediate, reads temporal from ReservoirBuffer[current]
    m_CommandList->SetPipelineState(m_RestirGIRasterSpatialPSO.Get());
    indices.InputIdx0 = m_RasterReservoirs[currentReservoir].srvIndex;
    indices.OutputIdx0 = m_RasterReservoirIntermediate.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
    m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(m_RasterReservoirIntermediate.resource.Get());
    m_CommandList->ResourceBarrier(1, &barrier2);

    // Restir Resolve
    // reads spatial output from Intermediate
    TransitionResource(m_RasterIndirectLightingTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_CommandList->SetPipelineState(m_RestirGIRasterResolvePSO.Get());
    indices.InputIdx0 = m_RasterReservoirIntermediate.srvIndex;
    indices.OutputIdx0 = m_RasterIndirectLightingTex.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
    m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

    m_CurrentReservoirIndex = previousReservoir; // Swap for next frame
}

void Renderer::CopyTextureToBackBuffer(const GPUTexture& texture)
{
    // Ensure source texture is in COPY_SOURCE
    TransitionResource(const_cast<GPUTexture&>(texture), D3D12_RESOURCE_STATE_COPY_SOURCE);

    // Transition backbuffer to COPY_DEST
    TransitionResource(m_RenderTargets[m_FrameIndex].Get(), m_BackBufferStates[m_FrameIndex], D3D12_RESOURCE_STATE_COPY_DEST);

    m_CommandList->CopyResource(m_RenderTargets[m_FrameIndex].Get(), texture.resource.Get());

    // Transition backbuffer to RTV for ImGui
    TransitionResource(m_RenderTargets[m_FrameIndex].Get(), m_BackBufferStates[m_FrameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET);
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

UINT Renderer::AllocateDescriptor()
{
    return m_SrvHeapIndex++;
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

    for (UINT i = 0; i < numLights; ++i) {
        float luminance = 0.2126f * lights[i].color.x + 0.7152f * lights[i].color.y + 0.0722f * lights[i].color.z;
        float w = lights[i].intensity * luminance;
        weights[i] = w;
        totalWeight += w;
    }

    for (UINT i = 0; i < numLights; ++i) {
        if (totalWeight > 0.0f) {
            lightsWithPDF[i].selectionPDF = weights[i] / totalWeight;
        } else {
            lightsWithPDF[i].selectionPDF = 1.0f / float(numLights);
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
    
    // Compute importance weights for each light
    float totalWeight = 0.0f;
    for (UINT i = 0; i < numLights; ++i) {
        // Importance = intensity * luminance(color)
        // Skip directional lights (position.w == 0) - they get 0 weight in stochastic sampling
        float w = 0.0f;
        if (lights[i].position.w > 0.5f) {
            float luminance = 0.2126f * lights[i].color.x + 0.7152f * lights[i].color.y + 0.0722f * lights[i].color.z;
            w = lights[i].intensity * luminance;
        }
        weights[i] = w;
        totalWeight += w;
    }
    
    // Build CDF
    std::vector<float> cdf(numLights);
    float cumulative = 0.0f;
    for (UINT i = 0; i < numLights; ++i) {
        if (totalWeight > 0.0f) {
            cumulative += weights[i] / totalWeight;
        }
        cdf[i] = cumulative;
    }
    if (numLights > 0) cdf[numLights - 1] = 1.0f;
    
    // Build LUT: for each LUT entry, find which light index to sample
    for (UINT i = 0; i < LIGHT_LUT_RESOLUTION; ++i) {
        float u = (i + 0.5f) / float(LIGHT_LUT_RESOLUTION); // Center of bin
        
        // Binary search in CDF to find light index
        uint32_t lightIdx = 0;
        if (numLights > 0 && totalWeight > 0.0f) {
            // Find first CDF entry >= u
            for (UINT j = 0; j < numLights; ++j) {
                if (u <= cdf[j]) {
                    lightIdx = j;
                    break;
                }
            }
        }
        lut[i] = lightIdx;
    }
    
    // Copy to GPU
    memcpy(m_LightLUTBuffer.cpuPtr, lut.data(), LIGHT_LUT_RESOLUTION * sizeof(uint32_t));
}

bool Renderer::CreateBuffer(GPUBuffer& buffer, UINT64 size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, bool createSRV, bool createUAV)
{
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(heapType);
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size);

    if (initialState & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE))
    {
        desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    CHECK_HR(m_Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&buffer.resource)), "CreateCommittedResource for Buffer failed");

    buffer.size = size;
    buffer.state = initialState;
    buffer.gpuAddress = buffer.resource->GetGPUVirtualAddress();

    if (heapType == D3D12_HEAP_TYPE_UPLOAD)
    {
        buffer.resource->Map(0, nullptr, &buffer.cpuPtr);
    }

    if (createSRV)
    {
        buffer.srvIndex = AllocateDescriptor();
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GetCPUDescriptorHandle(buffer.srvIndex);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = (UINT)(size / 4);
        srvDesc.Buffer.StructureByteStride = 0;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;

        m_Device->CreateShaderResourceView(buffer.resource.Get(), &srvDesc, srvHandle);
    }

    if (createUAV && (initialState & D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
    {
        buffer.uavIndex = AllocateDescriptor();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = GetCPUDescriptorHandle(buffer.uavIndex);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = (UINT)(size / 4);
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

        m_Device->CreateUnorderedAccessView(buffer.resource.Get(), nullptr, &uavDesc, uavHandle);
    }

    return true;
}

bool Renderer::CreateStructuredBuffer(GPUBuffer& buffer, UINT64 elementSize, UINT64 elementCount, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState)
{
    UINT64 size = elementSize * elementCount;
    if (!CreateBuffer(buffer, size, heapType, initialState, false, false)) return false;

    buffer.srvIndex = AllocateDescriptor();
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GetCPUDescriptorHandle(buffer.srvIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = (UINT)elementCount;
    srvDesc.Buffer.StructureByteStride = (UINT)elementSize;
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    m_Device->CreateShaderResourceView(buffer.resource.Get(), &srvDesc, srvHandle);

    if (initialState & D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    {
        buffer.uavIndex = AllocateDescriptor();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = GetCPUDescriptorHandle(buffer.uavIndex);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = (UINT)elementCount; // Assuming Reservoir buffers for now, could be more generic
        uavDesc.Buffer.StructureByteStride = (UINT)elementSize;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

        m_Device->CreateUnorderedAccessView(buffer.resource.Get(), nullptr, &uavDesc, uavHandle);
    }
    return true;
}

bool Renderer::CreateTexture(GPUTexture& texture, UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, const FLOAT* clearColor, UINT mipLevels, UINT arraySize)
{
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = static_cast<UINT16>(arraySize);
    desc.MipLevels = static_cast<UINT16>(mipLevels);
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;

    D3D12_CLEAR_VALUE clearVal = {};
    clearVal.Format = format;
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL && format == DXGI_FORMAT_R32_TYPELESS)
    {
        clearVal.Format = DXGI_FORMAT_D32_FLOAT;
    }
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
    {
        if (clearColor) memcpy(clearVal.Color, clearColor, sizeof(float) * 4);
    }
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        clearVal.DepthStencil.Depth = 1.0f;
    }

    CHECK_HR(m_Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        (flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) ? &clearVal : nullptr,
        IID_PPV_ARGS(&texture.resource)), "CreateCommittedResource for Texture failed");

    texture.state = initialState;
    texture.format = format;

    // Create SRV
    if (!(flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
    {
        texture.srvIndex = AllocateDescriptor();
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_SRVHeap->GetCPUDescriptorHandleForHeapStart();
        srvHandle.ptr += (UINT64)texture.srvIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS)
            srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
        else
            srvDesc.Format = format;
        
        if (arraySize > 1)
        {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srvDesc.Texture2DArray.MipLevels = mipLevels;
            srvDesc.Texture2DArray.ArraySize = arraySize;
            srvDesc.Texture2DArray.FirstArraySlice = 0;
            srvDesc.Texture2DArray.MostDetailedMip = 0;
        }
        else
        {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MipLevels = mipLevels;
        }

        m_Device->CreateShaderResourceView(texture.resource.Get(), &srvDesc, srvHandle);
    }

    // Create UAV
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        texture.uavIndex = AllocateDescriptor();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = m_SRVHeap->GetCPUDescriptorHandleForHeapStart();
        uavHandle.ptr += (UINT64)texture.uavIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = (format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS) ? DXGI_FORMAT_R32_FLOAT : format;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = 0;
        uavDesc.Texture2D.PlaneSlice = 0;

        m_Device->CreateUnorderedAccessView(texture.resource.Get(), nullptr, &uavDesc, uavHandle);
    }

    // Create RTV or DSV
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
    {
        static UINT rtvCount = 2; // Start after swap chain RTVs
        texture.rtvHandle = m_RTVHeap->GetCPUDescriptorHandleForHeapStart();
        texture.rtvHandle.ptr += (UINT64)rtvCount++ * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        m_Device->CreateRenderTargetView(texture.resource.Get(), nullptr, texture.rtvHandle);
    }
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        static UINT dsvCount = 0; // Unified allocation starting from 0
        texture.dsvHandle = m_DSVHeap->GetCPUDescriptorHandleForHeapStart();
        texture.dsvHandle.ptr += (UINT64)dsvCount++ * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        
        D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
        dsvDesc.Format = (format == DXGI_FORMAT_R32_TYPELESS) ? DXGI_FORMAT_D32_FLOAT : format;
        dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        
        m_Device->CreateDepthStencilView(texture.resource.Get(), &dsvDesc, texture.dsvHandle);
    }

    return true;
}

void Renderer::TransitionResource(GPUTexture& texture, D3D12_RESOURCE_STATES newState)
{
    if (texture.state == newState) return;

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        texture.resource.Get(),
        texture.state,
        newState
    );
    m_CommandList->ResourceBarrier(1, &barrier);
    texture.state = newState;
}

void Renderer::TransitionResource(GPUBuffer& buffer, D3D12_RESOURCE_STATES newState)
{
    if (buffer.state == newState) return;

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        buffer.resource.Get(),
        buffer.state,
        newState
    );
    m_CommandList->ResourceBarrier(1, &barrier);
    buffer.state = newState;
}

bool Renderer::CreateTexture3D(GPUTexture& texture, UINT width, UINT height, UINT depth, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, UINT mipLevels)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = static_cast<UINT16>(depth);
    desc.MipLevels = static_cast<UINT16>(mipLevels);
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;

    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    HRESULT hr = m_Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&texture.resource)
    );

    if (FAILED(hr)) return false;

    texture.state = initialState;
    texture.format = format;

    // Create SRV if not a depth stencil
    if (!(flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        srvDesc.Texture3D.MipLevels = mipLevels;
        srvDesc.Texture3D.MostDetailedMip = 0;

        texture.srvIndex = AllocateDescriptor();
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_SRVHeap->GetCPUDescriptorHandleForHeapStart();
        srvHandle.ptr += texture.srvIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_Device->CreateShaderResourceView(texture.resource.Get(), &srvDesc, srvHandle);
    }

    // Create UAV if requested
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = format;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uavDesc.Texture3D.MipSlice = 0;
        uavDesc.Texture3D.FirstWSlice = 0;
        uavDesc.Texture3D.WSize = depth;

        texture.uavIndex = AllocateDescriptor();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = m_SRVHeap->GetCPUDescriptorHandleForHeapStart();
        uavHandle.ptr += texture.uavIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_Device->CreateUnorderedAccessView(texture.resource.Get(), nullptr, &uavDesc, uavHandle);
    }

    return true;
}

void Renderer::TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES& currentState, D3D12_RESOURCE_STATES newState)
{
    if (currentState == newState) return;
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, currentState, newState);
    m_CommandList->ResourceBarrier(1, &barrier);
    currentState = newState;
}

void Renderer::TransitionBackBuffer(D3D12_RESOURCE_STATES newState)
{
    TransitionResource(m_RenderTargets[m_FrameIndex].Get(), m_BackBufferStates[m_FrameIndex], newState);
}

void Renderer::CreateGBuffer()
{
    float blackClear[] = { 0, 0, 0, 0 };
    CreateTexture(m_GBuffer.albedo, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.normal, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.material, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.depth, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE);
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

std::vector<char> Renderer::CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target)
{
    // Load HLSL source
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open HLSL file: " << filename << std::endl;
        return std::vector<char>();
    }

    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Create DXC compiler and utils
    Microsoft::WRL::ComPtr<IDxcUtils> dxcUtils;
    Microsoft::WRL::ComPtr<IDxcCompiler3> dxcCompiler;

    CHECK_HR(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils)), "DxcCreateInstance for DxcUtils failed");
    CHECK_HR(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler)), "DxcCreateInstance for DxcCompiler failed");

    // Create blob from source
    Microsoft::WRL::ComPtr<IDxcBlobEncoding> sourceBlob;
    CHECK_HR(dxcUtils->CreateBlob(source.c_str(), static_cast<UINT32>(source.size()), CP_UTF8, &sourceBlob), "CreateBlob failed");

    // Compile shader
    std::wstring entryPointW(entryPoint.begin(), entryPoint.end());
    std::wstring targetW(target.begin(), target.end());

    std::vector<LPCWSTR> arguments;
    arguments.push_back(L"-E");
    arguments.push_back(entryPointW.c_str());
    arguments.push_back(L"-T");
    arguments.push_back(targetW.c_str());
    arguments.push_back(L"-HV");
    arguments.push_back(L"2021");
    arguments.push_back(L"-I");
    arguments.push_back(L"Shaders");

#ifdef RTXDI_INCLUDE_DIR
    std::string rtxdiInclude = RTXDI_INCLUDE_DIR;
    std::wstring rtxdiIncludeW(rtxdiInclude.begin(), rtxdiInclude.end());
    arguments.push_back(L"-I");
    arguments.push_back(rtxdiIncludeW.c_str());
#endif

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = CP_UTF8;

    Microsoft::WRL::ComPtr<IDxcIncludeHandler> includeHandler;
    dxcUtils->CreateDefaultIncludeHandler(&includeHandler);

    Microsoft::WRL::ComPtr<IDxcResult> result;
    CHECK_HR(dxcCompiler->Compile(&sourceBuffer, arguments.data(), (UINT32)arguments.size(), includeHandler.Get(), IID_PPV_ARGS(&result)), "Compile failed");

    // Check compilation result
    HRESULT statusHr;
    CHECK_HR(result->GetStatus(&statusHr), "GetStatus failed");

    if (!SUCCEEDED(statusHr))
    {
        // Get error messages
        Microsoft::WRL::ComPtr<IDxcBlobUtf8> errors;
        if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr)) && errors && errors->GetStringLength() > 0)
        {
            std::cerr << "DXC Shader Compilation Errors for " << filename << " (" << entryPoint << " -> " << target << "):" << std::endl;
            std::cerr << errors->GetStringPointer() << std::endl;
        }
        else
        {
            std::cerr << "Shader compilation failed for " << filename << " (" << entryPoint << " -> " << target << ") but no error details available." << std::endl;
        }
        return std::vector<char>();
    }

    // Get compiled shader
    Microsoft::WRL::ComPtr<IDxcBlob> shaderBlob;
    CHECK_HR(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr), "GetOutput failed");

    std::vector<char> compiledShader(shaderBlob->GetBufferSize());
    memcpy(compiledShader.data(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());

    return compiledShader;
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