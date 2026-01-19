#include "Renderer.h"
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
    rtvHeapDesc.NumDescriptors = 2;
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

    // Create depth stencil buffer
    {
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
        dsvHeapDesc.NumDescriptors = 1;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        hr = m_Device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_DSVHeap));
        if (FAILED(hr))
        {
            std::cerr << "CreateDescriptorHeap for DSV failed" << std::endl;
            return false;
        }

        D3D12_RESOURCE_DESC depthStencilDesc;
        depthStencilDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        depthStencilDesc.Alignment = 0;
        depthStencilDesc.Width = WINDOW_WIDTH;
        depthStencilDesc.Height = WINDOW_HEIGHT;
        depthStencilDesc.DepthOrArraySize = 1;
        depthStencilDesc.MipLevels = 1;
        depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
        depthStencilDesc.SampleDesc.Count = 1;
        depthStencilDesc.SampleDesc.Quality = 0;
        depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

        D3D12_CLEAR_VALUE clearValue;
        clearValue.Format = DXGI_FORMAT_D32_FLOAT;
        clearValue.DepthStencil.Depth = 1.0f;
        clearValue.DepthStencil.Stencil = 0;

        hr = m_Device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &depthStencilDesc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &clearValue,
            IID_PPV_ARGS(&m_DepthStencilBuffer)
        );
        if (FAILED(hr))
        {
            std::cerr << "CreateCommittedResource for depth buffer failed" << std::endl;
            return false;
        }

        m_Device->CreateDepthStencilView(m_DepthStencilBuffer.Get(), nullptr, m_DSVHeap->GetCPUDescriptorHandleForHeapStart());
    }

    // Create constant buffer for view-projection matrix
    {
        const UINT constantBufferSize = sizeof(float) * 16; // 4x4 matrix

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

        D3D12_RESOURCE_DESC resourceDesc = {};
        resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resourceDesc.Alignment = 0;
        resourceDesc.Width = constantBufferSize;
        resourceDesc.Height = 1;
        resourceDesc.DepthOrArraySize = 1;
        resourceDesc.MipLevels = 1;
        resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
        resourceDesc.SampleDesc.Count = 1;
        resourceDesc.SampleDesc.Quality = 0;
        resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        hr = m_Device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_ConstantBuffer)
        );
        if (FAILED(hr))
        {
            std::cerr << "CreateCommittedResource for constant buffer failed" << std::endl;
            return false;
        }

        // Map the constant buffer
        D3D12_RANGE readRange = { 0, 0 };
        hr = m_ConstantBuffer->Map(0, &readRange, &m_ConstantBufferData);
        if (FAILED(hr))
        {
            std::cerr << "Map constant buffer failed" << std::endl;
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

    std::cout << "Renderer initialized successfully!" << std::endl;
    return true;
}

void Renderer::Shutdown()
{
    // Wait for the GPU to be done with all resources
    WaitForPreviousFrame();

    // Cleanup constant buffer
    if (m_ConstantBuffer && m_ConstantBufferData)
    {
        m_ConstantBuffer->Unmap(0, nullptr);
        m_ConstantBufferData = nullptr;
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

    hr = m_CommandList->Reset(m_CommandAllocator.Get(), m_PipelineState.Get());
    if (FAILED(hr))
    {
        std::cerr << "CommandList Reset failed" << std::endl;
        return;
    }

    // Set necessary state
    m_CommandList->SetGraphicsRootSignature(m_RootSignature.Get());

    // Set constant buffer (view-projection matrix)
    m_CommandList->SetGraphicsRootConstantBufferView(0, m_ConstantBuffer->GetGPUVirtualAddress());

    m_CommandList->RSSetViewports(1, &CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(WINDOW_WIDTH), static_cast<float>(WINDOW_HEIGHT)));
    m_CommandList->RSSetScissorRects(1, &CD3DX12_RECT(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT));

    // Indicate that the back buffer will be used as a render target
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_RenderTargets[m_FrameIndex].Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_CommandList->ResourceBarrier(1, &barrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_RTVHeap->GetCPUDescriptorHandleForHeapStart());
    rtvHandle.ptr += m_FrameIndex * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle(m_DSVHeap->GetCPUDescriptorHandleForHeapStart());
    m_CommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

    // Clear the render target with configurable background color
    const float clearColor[] = { m_BackgroundColor[0], m_BackgroundColor[1], m_BackgroundColor[2], 1.0f };
    m_CommandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

    // Clear the depth buffer
    m_CommandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
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

void Renderer::CreateRootSignature()
{
    // Create root parameter for constant buffer (view-projection matrix)
    D3D12_ROOT_PARAMETER rootParameter = {};
    rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameter.Descriptor.ShaderRegister = 0;  // b0
    rootParameter.Descriptor.RegisterSpace = 0;
    rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = 1;
    rootSignatureDesc.pParameters = &rootParameter;
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.pStaticSamplers = nullptr;
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
    // Compile shaders at runtime
    std::vector<char> vertexShader = CompileShader("Shaders/Triangle.hlsl", "VSMain", "vs_6_0");
    std::vector<char> pixelShader = CompileShader("Shaders/Triangle.hlsl", "PSMain", "ps_6_0");

    // Define input layout for GLTF models
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(GLTFVertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(GLTFVertex, normal), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(GLTFVertex, texCoord), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    // Create pipeline state
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
    psoDesc.pRootSignature = m_RootSignature.Get();
    psoDesc.VS = { reinterpret_cast<UINT8*>(vertexShader.data()), vertexShader.size() };
    psoDesc.PS = { reinterpret_cast<UINT8*>(pixelShader.data()), pixelShader.size() };
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

    // Configure depth stencil state explicitly
    D3D12_DEPTH_STENCIL_DESC depthStencilDesc = {};
    depthStencilDesc.DepthEnable = TRUE;
    depthStencilDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    depthStencilDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;  // Closer objects pass
    depthStencilDesc.StencilEnable = FALSE;
    psoDesc.DepthStencilState = depthStencilDesc;

    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;

    HRESULT hr = m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PipelineState));
    if (FAILED(hr))
    {
        std::cerr << "CreateGraphicsPipelineState failed" << std::endl;
    }
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
    hr = dxcUtils->CreateBlob(source.c_str(), source.size(), CP_UTF8, &sourceBlob);
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

void Renderer::UpdateViewProjectionMatrix(const DirectX::XMMATRIX& viewMatrix)
{
    float aspectRatio = static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT;
    float fovY = 45.0f * (3.14159265359f / 180.0f); // 45 degrees in radians
    float nearZ = 0.1f;
    float farZ = 1000.0f;

    // Create projection matrix (left-handed perspective)
    DirectX::XMMATRIX projectionMatrix = DirectX::XMMatrixPerspectiveFovLH(fovY, aspectRatio, nearZ, farZ);

    // Combine view and projection matrices (view * projection)
    DirectX::XMMATRIX viewProjectionMatrix = DirectX::XMMatrixMultiply(viewMatrix, projectionMatrix);

    // Store as float array for constant buffer
    float viewProjection[16];
    DirectX::XMStoreFloat4x4(reinterpret_cast<DirectX::XMFLOAT4X4*>(viewProjection), viewProjectionMatrix);

    // Update constant buffer
    memcpy(m_ConstantBufferData, viewProjection, sizeof(viewProjection));
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