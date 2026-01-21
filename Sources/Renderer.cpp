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
    if (!CreateBuffer(m_FrameCB, (sizeof(float) * 16 + 255) & ~255, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
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

        hr = m_Device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_SRVHeap));
        if (FAILED(hr))
        {
            std::cerr << "CreateDescriptorHeap for SRV failed" << std::endl;
            return false;
        }
    }

    // Create GBuffer
    CreateGBuffer();

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
    m_CommandList->SetGraphicsRootDescriptorTable(3, m_SRVHeap->GetGPUDescriptorHandleForHeapStart());

    // Set Frame constant buffer (viewProj)
    m_CommandList->SetGraphicsRootConstantBufferView(0, m_FrameCB.gpuAddress);

    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(WINDOW_WIDTH), static_cast<float>(WINDOW_HEIGHT));
    D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    m_CommandList->RSSetViewports(1, &viewport);
    m_CommandList->RSSetScissorRects(1, &scissorRect);
}

void Renderer::EndFrame()
{
    // Indicate that the back buffer will now be used to present
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_RenderTargets[m_FrameIndex].Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_CommandList->ResourceBarrier(1, &barrier);

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
    D3D12_DESCRIPTOR_RANGE textureRange = {};
    textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    textureRange.NumDescriptors = 4096;
    textureRange.BaseShaderRegister = 0;  // t0
    textureRange.RegisterSpace = 0;
    textureRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER rootParameters[4] = {};

    // Frame CB: viewProj
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameters[0].Descriptor.ShaderRegister = 0;  // b0
    rootParameters[0].Descriptor.RegisterSpace = 0;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Material constants: baseColorFactor, metallicFactor, roughnessFactor, hasBaseColorTexture
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParameters[1].Constants.ShaderRegister = 1;  // b1
    rootParameters[1].Constants.RegisterSpace = 0;
    rootParameters[1].Constants.Num32BitValues = 8;
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Mesh constants: world matrix (16 floats)
    rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParameters[2].Constants.ShaderRegister = 3;  // b3
    rootParameters[2].Constants.RegisterSpace = 0;
    rootParameters[2].Constants.Num32BitValues = 16;
    rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Bindless texture descriptor table
    rootParameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[3].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[3].DescriptorTable.pDescriptorRanges = &textureRange;
    rootParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Static sampler
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.MipLODBias = 0.0f;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0;  // s0
    sampler.RegisterSpace = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = _countof(rootParameters);
    rootSignatureDesc.pParameters = rootParameters;
    rootSignatureDesc.NumStaticSamplers = 1;
    rootSignatureDesc.pStaticSamplers = &sampler;
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
    }
}

void Renderer::CreatePipelineState()
{
    // Define input layout for GLTF models
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(GLTFVertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(GLTFVertex, normal), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(GLTFVertex, texCoord), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    // 1. Depth Pre-Pass PSO
    {
        std::vector<char> vs = CompileShader("Shaders/DepthOnly.hlsl", "VSMain", "vs_6_0");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
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
        std::vector<char> vs = CompileShader("Shaders/Gbuffer.hlsl", "VSMain", "vs_6_0");
        std::vector<char> ps = CompileShader("Shaders/Gbuffer.hlsl", "PSMain", "ps_6_0");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
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
        std::vector<char> vs = CompileShader("Shaders/Lighting.hlsl", "VSMain", "vs_6_0");
        std::vector<char> ps = CompileShader("Shaders/Lighting.hlsl", "PSMain", "ps_6_0");
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

    std::cout << "Pipeline states created successfully" << std::endl;
}

UINT Renderer::AllocateDescriptor()
{
    return m_SrvHeapIndex++;
}

bool Renderer::CreateBuffer(GPUBuffer& buffer, UINT64 size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState)
{
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(heapType);
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size);

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
        srvHandle.ptr += texture.srvIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = (format == DXGI_FORMAT_D32_FLOAT) ? DXGI_FORMAT_R32_FLOAT : format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = mipLevels;

        m_Device->CreateShaderResourceView(texture.resource.Get(), &srvDesc, srvHandle);
    }

    // Create RTV or DSV
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
    {
        static UINT rtvCount = 2; // Start after swap chain RTVs
        texture.rtvHandle = m_RTVHeap->GetCPUDescriptorHandleForHeapStart();
        texture.rtvHandle.ptr += rtvCount++ * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        m_Device->CreateRenderTargetView(texture.resource.Get(), nullptr, texture.rtvHandle);
    }
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        static UINT dsvCount = 0; // Unified allocation starting from 0
        texture.dsvHandle = m_DSVHeap->GetCPUDescriptorHandleForHeapStart();
        texture.dsvHandle.ptr += dsvCount++ * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        m_Device->CreateDepthStencilView(texture.resource.Get(), nullptr, texture.dsvHandle);
    }

    return true;
}

void Renderer::CreateGBuffer()
{
    float blackClear[] = { 0, 0, 0, 0 };
    CreateTexture(m_GBuffer.albedo, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.normal, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.material, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, blackClear);
    CreateTexture(m_GBuffer.depth, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE);
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

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = DXC_CP_ACP;

    Microsoft::WRL::ComPtr<IDxcResult> result;
    hr = dxcCompiler->Compile(&sourceBuffer, arguments.data(), arguments.size(), nullptr, IID_PPV_ARGS(&result));
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

void Renderer::UpdateFrameCB(const DirectX::XMMATRIX& viewProjMatrix)
{
    float viewProj[16];
    DirectX::XMStoreFloat4x4(reinterpret_cast<DirectX::XMFLOAT4X4*>(viewProj), viewProjMatrix);
    memcpy(m_FrameCB.cpuPtr, viewProj, sizeof(viewProj));
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