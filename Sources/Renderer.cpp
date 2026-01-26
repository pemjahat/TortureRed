#include "Renderer.h"
#include "Model.h"
#include "Utility.h"
#include <iostream>
#include <fstream>
#include <dxcapi.h>
#include <cassert>

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
    HRESULT hr = CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory));
    if (FAILED(hr))
    {
        std::cerr << "CreateDXGIFactory2 failed" << std::endl;
        return false;
    }

    // Create device
    Microsoft::WRL::ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    hr = D3D12CreateDevice(
        hardwareAdapter.Get(),
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&m_Device)
    );
    if (FAILED(hr))
    {
        std::cerr << "D3D12CreateDevice failed" << std::endl;
        return false;
    }

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

    hr = m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_CommandQueue));
    if (FAILED(hr))
    {
        std::cerr << "CreateCommandQueue failed" << std::endl;
        return false;
    }

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
    hr = factory->CreateSwapChainForHwnd(
        m_CommandQueue.Get(),
        hwnd,
        &swapChainDesc,
        nullptr,
        nullptr,
        &swapChain
    );
    if (FAILED(hr))
    {
        std::cerr << "CreateSwapChainForHwnd failed" << std::endl;
        return false;
    }

    hr = swapChain.As(&m_SwapChain);
    if (FAILED(hr))
    {
        std::cerr << "SwapChain As failed" << std::endl;
        return false;
    }

    m_FrameIndex = m_SwapChain->GetCurrentBackBufferIndex();

    // Create descriptor heap for render target views
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = 16;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    hr = m_Device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_RTVHeap));
    if (FAILED(hr))
    {
        std::cerr << "CreateDescriptorHeap for RTV failed" << std::endl;
        return false;
    }

    UINT rtvDescriptorSize = m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Create render target view for each frame
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_RTVHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < 2; n++)
    {
        hr = m_SwapChain->GetBuffer(n, IID_PPV_ARGS(&m_RenderTargets[n]));
        if (FAILED(hr))
        {
            std::cerr << "GetBuffer failed" << std::endl;
            return false;
        }
        m_Device->CreateRenderTargetView(m_RenderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.ptr += rtvDescriptorSize;
    }

    // Create DSV descriptor heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
        dsvHeapDesc.NumDescriptors = 4;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        hr = m_Device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_DSVHeap));
        if (FAILED(hr))
        {
            std::cerr << "CreateDescriptorHeap for DSV failed" << std::endl;
            return false;
        }
    }

    // Create constant buffers
    if (!CreateBuffer(m_FrameCB, (sizeof(FrameConstants) + 255) & ~255, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
    {
        std::cerr << "Failed to create frame constant buffer" << std::endl;
        return false;
    }

    if (!CreateBuffer(m_LightCB, (sizeof(LightConstants) + 255) & ~255, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
    {
        std::cerr << "Failed to create light constant buffer" << std::endl;
        return false;
    }

    // Create SRV descriptor heap for textures
    {
        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
        srvHeapDesc.NumDescriptors = 4096; // Increased from 1024
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        hr = m_Device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_SRVHeap));
        if (FAILED(hr))
        {
            std::cerr << "CreateDescriptorHeap for SRV failed" << std::endl;
            return false;
        }
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
    }

    // Create command allocator
    hr = m_Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_CommandAllocator));
    if (FAILED(hr))
    {
        std::cerr << "CreateCommandAllocator failed" << std::endl;
        return false;
    }

    // Create command list
    hr = m_Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_CommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_CommandList));
    if (FAILED(hr))
    {
        std::cerr << "CreateCommandList failed" << std::endl;
        return false;
    }
    hr = m_CommandList->Close();
    if (FAILED(hr))
    {
        std::cerr << "CommandList Close failed" << std::endl;
        return false;
    }

    // Create fence
    hr = m_Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_Fence));
    if (FAILED(hr))
    {
        std::cerr << "CreateFence failed" << std::endl;
        return false;
    }
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

    if (m_LightCB.resource && m_LightCB.cpuPtr)
    {
        m_LightCB.resource->Unmap(0, nullptr);
        m_LightCB.cpuPtr = nullptr;
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
    HRESULT hr = m_CommandAllocator->Reset();
    if (FAILED(hr))
    {
        std::cerr << "CommandAllocator Reset failed" << std::endl;
        return;
    }

    hr = m_CommandList->Reset(m_CommandAllocator.Get(), nullptr);
    if (FAILED(hr))
    {
        std::cerr << "CommandList Reset failed" << std::endl;
        return;
    }

    // Set necessary state
    m_CommandList->SetGraphicsRootSignature(m_RootSignature.Get());

    // Set descriptor heaps
    ID3D12DescriptorHeap* heaps[] = { m_SRVHeap.Get() };
    m_CommandList->SetDescriptorHeaps(_countof(heaps), heaps);

    // Bind the global descriptor table (bindless)
    m_CommandList->SetGraphicsRootDescriptorTable(4, m_SRVHeap->GetGPUDescriptorHandleForHeapStart());

    // Set Frame constant buffer (viewProj)
    m_CommandList->SetGraphicsRootConstantBufferView(0, m_FrameCB.gpuAddress);

    // Set Light constant buffer
    m_CommandList->SetGraphicsRootConstantBufferView(1, m_LightCB.gpuAddress);

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

    HRESULT hr = m_CommandList->Close();
    if (FAILED(hr))
    {
        std::cerr << "CommandList Close failed" << std::endl;
        return;
    }

    // Execute the command list
    ID3D12CommandList* ppCommandLists[] = { m_CommandList.Get() };
    m_CommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present the frame
    hr = m_SwapChain->Present(1, 0);
    if (FAILED(hr))
    {
        std::cerr << "Present failed" << std::endl;
    }

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
    D3D12_DESCRIPTOR_RANGE srvRanges[2] = {};
    // t0 space0: Bindless textures
    srvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRanges[0].NumDescriptors = 4096;
    srvRanges[0].BaseShaderRegister = 0;
    srvRanges[0].RegisterSpace = 0;
    srvRanges[0].OffsetInDescriptorsFromTableStart = 0;

    // t0 space2: Bindless buffers (unused now)
    srvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRanges[1].NumDescriptors = 4096;
    srvRanges[1].BaseShaderRegister = 0;
    srvRanges[1].RegisterSpace = 2;
    srvRanges[1].OffsetInDescriptorsFromTableStart = 0;

    D3D12_DESCRIPTOR_RANGE uavRange0 = {};
    uavRange0.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange0.NumDescriptors = 1;
    uavRange0.BaseShaderRegister = 0;
    uavRange0.RegisterSpace = 0;
    uavRange0.OffsetInDescriptorsFromTableStart = 0;

    D3D12_DESCRIPTOR_RANGE uavRange1 = {};
    uavRange1.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange1.NumDescriptors = 1;
    uavRange1.BaseShaderRegister = 1;
    uavRange1.RegisterSpace = 0;
    uavRange1.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER rootParameters[10] = {};

    // 0: b0 FrameConstants
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameters[0].Descriptor.ShaderRegister = 0;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // 1: b1 Light constants
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameters[1].Descriptor.ShaderRegister = 1;
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // 2: t0 (space1) Material Data SRV
    rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rootParameters[2].Descriptor.ShaderRegister = 0;
    rootParameters[2].Descriptor.RegisterSpace = 1;
    rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // 3: t1 (space1) Draw Node Data SRV
    rootParameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rootParameters[3].Descriptor.ShaderRegister = 1;
    rootParameters[3].Descriptor.RegisterSpace = 1;
    rootParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // 4: t0 (space0/space2) Bindless descriptor table
    rootParameters[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[4].DescriptorTable.NumDescriptorRanges = 2;
    rootParameters[4].DescriptorTable.pDescriptorRanges = srvRanges;
    rootParameters[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // 5: t2 (space1) TLAS
    rootParameters[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rootParameters[5].Descriptor.ShaderRegister = 2;
    rootParameters[5].Descriptor.RegisterSpace = 1;

    // 6: t3 (space1) Primitive Data SRV - indices
    rootParameters[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rootParameters[6].Descriptor.ShaderRegister = 3;
    rootParameters[6].Descriptor.RegisterSpace = 1;
    rootParameters[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // 7: t4 (space1) Global Vertex Buffer SRV
    rootParameters[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rootParameters[7].Descriptor.ShaderRegister = 4;
    rootParameters[7].Descriptor.RegisterSpace = 1;
    rootParameters[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // 8: u0 (space0) Accumulation Buffer UAV
    rootParameters[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[8].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[8].DescriptorTable.pDescriptorRanges = &uavRange0;

    // 9: u1 (space0) Output Buffer UAV
    rootParameters[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[9].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[9].DescriptorTable.pDescriptorRanges = &uavRange1;

    // Static samplers
    D3D12_STATIC_SAMPLER_DESC samplers[2] = {};
    
    // Default sampler
    samplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].MipLODBias = 0.0f;
    samplers[0].MaxAnisotropy = 1;
    samplers[0].ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    samplers[0].MinLOD = 0.0f;
    samplers[0].MaxLOD = D3D12_FLOAT32_MAX;
    samplers[0].ShaderRegister = 0;  // s0
    samplers[0].RegisterSpace = 0;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Shadow sampler
    samplers[1].Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
    samplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[1].MipLODBias = 0.0f;
    samplers[1].MaxAnisotropy = 1;
    samplers[1].ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    samplers[1].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
    samplers[1].MinLOD = 0.0f;
    samplers[1].MaxLOD = D3D12_FLOAT32_MAX;
    samplers[1].ShaderRegister = 1;  // s1
    samplers[1].RegisterSpace = 0;
    samplers[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = _countof(rootParameters);
    rootSignatureDesc.pParameters = rootParameters;
    rootSignatureDesc.NumStaticSamplers = _countof(samplers);
    rootSignatureDesc.pStaticSamplers = samplers;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (FAILED(hr))
    {
        std::cerr << "D3D12SerializeRootSignature failed" << std::endl;
        return;
    }

    hr = m_Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_RootSignature));
    if (FAILED(hr))
    {
        std::cerr << "CreateRootSignature failed" << std::endl;
        return;
    }

    // Create command signature for ExecuteIndirect
    D3D12_INDIRECT_ARGUMENT_DESC drawArg = {};
    drawArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;

    D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc = {};
    commandSignatureDesc.ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
    commandSignatureDesc.NumArgumentDescs = 1;
    commandSignatureDesc.pArgumentDescs = &drawArg;

    hr = m_Device->CreateCommandSignature(&commandSignatureDesc, nullptr, IID_PPV_ARGS(&m_CommandSignature));
    if (FAILED(hr))
    {
        std::cerr << "CreateCommandSignature failed" << std::endl;
    }
}

void Renderer::CreatePipelineState()
{
    // 1. Depth Pre-Pass PSO
    {
        std::vector<char> vs = CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { nullptr, 0 };
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 0;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.SampleDesc.Count = 1;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_DepthPrePassPSO));
    }

    // 2. G-Buffer PSO
    {
        std::vector<char> vs = CompileShader("Shaders/Gbuffer.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = CompileShader("Shaders/Gbuffer.hlsl", "PSMain", "ps_6_8");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { nullptr, 0 };
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);        
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO; // No depth write, using pre-pass
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 3;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        psoDesc.RTVFormats[2] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.SampleDesc.Count = 1;
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
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_LightingPSO));
    }

    // 3.5 Debug PSO
    {
        std::vector<char> vs = CompileShader("Shaders/DebugShadow.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = CompileShader("Shaders/DebugShadow.hlsl", "PSMain", "ps_6_8");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;
        m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_DebugPSO));
    }

    // 4. Shadow PSO
    {
        std::vector<char> vs = CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_8");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { nullptr, 0 };
        psoDesc.pRootSignature = m_RootSignature.Get();
        psoDesc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        psoDesc.RasterizerState.DepthBias = 1000;
        psoDesc.RasterizerState.DepthBiasClamp = 0.0f;
        psoDesc.RasterizerState.SlopeScaledDepthBias = 1.5f;
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 0;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.SampleDesc.Count = 1;

        HRESULT hr = m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_ShadowPSO));
        if (FAILED(hr))
        {
            std::cerr << "CreateGraphicsPipelineState for Shadow PSO failed" << std::endl;
        }
    }

    if (m_RayTracingSupported)
    {
        CreateRayTracingPipeline();
    }

    std::cout << "Pipeline states created successfully" << std::endl;
}

void Renderer::CreateRayTracingPipeline()
{
    // Load Compute Shader
    auto shaderCode = CompileShader("Shaders/PathTracer.hlsl", "CSMain", "cs_6_5");
    if (shaderCode.empty())
    {
        std::cerr << "Path Tracer shader compilation failed!" << std::endl;
        return;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_RootSignature.Get();
    psoDesc.CS = { shaderCode.data(), shaderCode.size() };
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    HRESULT hr = m_Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_PathTracerPSO));
    if (FAILED(hr))
    {
        std::cerr << "Failed to create Path Tracer Compute PSO" << std::endl;
    }
}

void Renderer::DispatchRays(Model* model, const FrameConstants& frame, const LightConstants& light)
{
    if (!m_PathTracerPSO || !model) return;

    // Update constant buffers
    memcpy(m_FrameCB.cpuPtr, &frame, sizeof(FrameConstants));
    memcpy(m_LightCB.cpuPtr, &light, sizeof(LightConstants));

    // Transition UAVs
    TransitionResource(m_AccumulationBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    TransitionResource(m_PathTracerOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    D3D12_RESOURCE_BARRIER uavBarriers[2];
    uavBarriers[0] = CD3DX12_RESOURCE_BARRIER::UAV(m_AccumulationBuffer.resource.Get());
    uavBarriers[1] = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
    m_CommandList->ResourceBarrier(2, uavBarriers);

    m_CommandList->SetComputeRootSignature(m_RootSignature.Get());
    m_CommandList->SetPipelineState(m_PathTracerPSO.Get());
    m_CommandList->SetDescriptorHeaps(1, m_SRVHeap.GetAddressOf());

    m_CommandList->SetComputeRootConstantBufferView(0, m_FrameCB.gpuAddress);
    m_CommandList->SetComputeRootConstantBufferView(1, m_LightCB.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(2, model->GetMaterialBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(3, model->GetDrawNodeBufferAddress());
    m_CommandList->SetComputeRootDescriptorTable(4, GetGPUDescriptorHandle(0)); // Bindless
    m_CommandList->SetComputeRootShaderResourceView(5, m_TLAS.gpuAddress);
    m_CommandList->SetComputeRootShaderResourceView(6, model->GetGlobalIndexBufferAddress());
    m_CommandList->SetComputeRootShaderResourceView(7, model->GetGlobalVertexBufferAddress());
    m_CommandList->SetComputeRootDescriptorTable(8, GetGPUDescriptorHandle(m_AccumulationBuffer.uavIndex));
    m_CommandList->SetComputeRootDescriptorTable(9, GetGPUDescriptorHandle(m_PathTracerOutput.uavIndex));

    m_CommandList->Dispatch((WINDOW_WIDTH + 7) / 8, (WINDOW_HEIGHT + 7) / 8, 1);

    // Transition for blitting/Imgui
    TransitionResource(m_PathTracerOutput, D3D12_RESOURCE_STATE_COPY_SOURCE);
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
    if (FAILED(m_Device.As(&device5))) return;

    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmdList4;
    if (FAILED(m_CommandList.As(&cmdList4))) return;

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
        info.geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

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
        inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
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

bool Renderer::CreateBuffer(GPUBuffer& buffer, UINT64 size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, bool createSRV)
{
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(heapType);
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size);

    if (initialState & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE))
    {
        desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    HRESULT hr = m_Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&buffer.resource));

    if (FAILED(hr)) return false;

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
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_SRVHeap->GetCPUDescriptorHandleForHeapStart();
        srvHandle.ptr += (UINT64)buffer.srvIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

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

    return true;
}

bool Renderer::CreateStructuredBuffer(GPUBuffer& buffer, UINT64 elementSize, UINT64 elementCount, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState)
{
    UINT64 size = elementSize * elementCount;
    if (!CreateBuffer(buffer, size, heapType, initialState, false)) return false;

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
    return true;
}

bool Renderer::CreateTexture(GPUTexture& texture, UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, const FLOAT* clearColor, UINT mipLevels)
{
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
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

    HRESULT hr = m_Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        (flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) ? &clearVal : nullptr,
        IID_PPV_ARGS(&texture.resource));

    if (FAILED(hr)) return false;

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
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = mipLevels;

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

    HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
    if (FAILED(hr))
    {
        std::cerr << "DxcCreateInstance for DxcUtils failed" << std::endl;
        return std::vector<char>();
    }

    hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));
    if (FAILED(hr))
    {
        std::cerr << "DxcCreateInstance for DxcCompiler failed" << std::endl;
        return std::vector<char>();
    }

    // Create blob from source
    Microsoft::WRL::ComPtr<IDxcBlobEncoding> sourceBlob;
    hr = dxcUtils->CreateBlob(source.c_str(), static_cast<UINT32>(source.size()), CP_UTF8, &sourceBlob);
    if (FAILED(hr))
    {
        std::cerr << "CreateBlob failed" << std::endl;
        return std::vector<char>();
    }

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

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = CP_UTF8;

    Microsoft::WRL::ComPtr<IDxcIncludeHandler> includeHandler;
    dxcUtils->CreateDefaultIncludeHandler(&includeHandler);

    Microsoft::WRL::ComPtr<IDxcResult> result;
    hr = dxcCompiler->Compile(&sourceBuffer, arguments.data(), (UINT32)arguments.size(), includeHandler.Get(), IID_PPV_ARGS(&result));
    if (FAILED(hr))
    {
        std::cerr << "Compile failed" << std::endl;
        return std::vector<char>();
    }

    // Check compilation result
    HRESULT statusHr;
    hr = result->GetStatus(&statusHr);
    if (FAILED(hr))
    {
        std::cerr << "GetStatus failed" << std::endl;
        return std::vector<char>();
    }

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
    hr = result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr);
    if (FAILED(hr))
    {
        std::cerr << "GetOutput failed" << std::endl;
        return std::vector<char>();
    }

    std::vector<char> compiledShader(shaderBlob->GetBufferSize());
    memcpy(compiledShader.data(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());

    return compiledShader;
}

void Renderer::UpdateFrameCB(const FrameConstants& frameConstants)
{
    memcpy(m_FrameCB.cpuPtr, &frameConstants, sizeof(FrameConstants));
}

void Renderer::UpdateLightCB(const LightConstants& lightConstants)
{
    memcpy(m_LightCB.cpuPtr, &lightConstants, sizeof(LightConstants));
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
    HRESULT hr = m_CommandQueue->Signal(m_Fence.Get(), fence);
    if (FAILED(hr))
    {
        std::cerr << "CommandQueue Signal failed" << std::endl;
        return;
    }
    m_FenceValue++;

    // Wait until the previous frame is finished
    if (m_Fence->GetCompletedValue() < fence)
    {
        hr = m_Fence->SetEventOnCompletion(fence, m_FenceEvent);
        if (FAILED(hr))
        {
            std::cerr << "SetEventOnCompletion failed" << std::endl;
            return;
        }
        WaitForSingleObject(m_FenceEvent, INFINITE);
    }

    m_FrameIndex = m_SwapChain->GetCurrentBackBufferIndex();
}