#include "Application.h"
#include <SDL.h>
#include <SDL_syswm.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <directx/d3dx12.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <wrl.h>

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const char* WINDOW_TITLE = "TortureRed";

Application::Application()
    : m_IsRunning(false)
    , m_Window(nullptr)
    , m_FrameIndex(0)
    , m_FenceValue(0)
    , m_FenceEvent(nullptr)
{
}

Application::~Application()
{
    Shutdown();
}

void Application::Run()
{
    Initialize();

    m_IsRunning = true;
    Uint32 lastTime = SDL_GetTicks();

    while (m_IsRunning)
    {
        Uint32 currentTime = SDL_GetTicks();
        float deltaTime = (currentTime - lastTime) / 1000.0f;
        lastTime = currentTime;

        ProcessEvents();
        Update(deltaTime);
        Render();

        // Cap frame rate
        SDL_Delay(16); // ~60 FPS
    }
}

void Application::Initialize()
{
    // Initialize SDL
    CHECK_BOOL(SDL_Init(SDL_INIT_VIDEO) == 0, "SDL_Init failed");

    // Create window
    m_Window = SDL_CreateWindow(
        WINDOW_TITLE,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    CHECK_BOOL(m_Window != nullptr, "SDL_CreateWindow failed");

    // Initialize DirectX 12
    InitializeDirectX();

    std::cout << "TortureRed application initialized successfully!" << std::endl;
}

void Application::InitializeDirectX()
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

    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(m_Window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;

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
    rtvHeapDesc.NumDescriptors = 2;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    CHECK_HR(m_Device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_RTVHeap)), "CreateDescriptorHeap failed");

    UINT rtvDescriptorSize = m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Create render target view for each frame
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_RTVHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT n = 0; n < 2; n++)
    {
        CHECK_HR(m_SwapChain->GetBuffer(n, IID_PPV_ARGS(&m_RenderTargets[n])), "GetBuffer failed");
        m_Device->CreateRenderTargetView(m_RenderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.ptr += rtvDescriptorSize;
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
    CHECK_BOOL(m_FenceEvent != nullptr, "CreateEvent failed");

    // Create root signature
    CreateRootSignature();

    // Create pipeline state
    CreatePipelineState();

    // Create vertex buffer
    CreateVertexBuffer();

    // Initialize ImGui
    InitializeImGui();
}

void Application::GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter)
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

void Application::CreateRootSignature()
{
    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = 0;
    rootSignatureDesc.pParameters = nullptr;
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.pStaticSamplers = nullptr;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    CHECK_HR(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error), "D3D12SerializeRootSignature failed");
    CHECK_HR(m_Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_RootSignature)), "CreateRootSignature failed");
}

void Application::CreatePipelineState()
{
    // Compile shaders at runtime
    std::vector<char> vertexShader = CompileShader("Shaders/Triangle.hlsl", "VSMain", "vs_6_0");
    std::vector<char> pixelShader = CompileShader("Shaders/Triangle.hlsl", "PSMain", "ps_6_0");

    // Define input layout
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    // Create pipeline state
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
    psoDesc.pRootSignature = m_RootSignature.Get();
    psoDesc.VS = { reinterpret_cast<UINT8*>(vertexShader.data()), vertexShader.size() };
    psoDesc.PS = { reinterpret_cast<UINT8*>(pixelShader.data()), pixelShader.size() };
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;

    CHECK_HR(m_Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PipelineState)), "CreateGraphicsPipelineState failed");
}

void Application::CreateVertexBuffer()
{
    // Define triangle vertices
    struct Vertex
    {
        float position[3];
        float color[3];
    };

    Vertex triangleVertices[] =
    {
        { { 0.0f, 0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f } }, // Top vertex - red
        { { 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f } }, // Bottom right - green
        { { -0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f } } // Bottom left - blue
    };

    const UINT vertexBufferSize = sizeof(triangleVertices);

    // Create vertex buffer
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = vertexBufferSize;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    CHECK_HR(m_Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_VertexBuffer)
    ), "CreateCommittedResource failed");

    // Copy vertex data to buffer
    UINT8* pVertexDataBegin;
    D3D12_RANGE readRange = { 0, 0 };
    CHECK_HR(m_VertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)), "Map failed");
    memcpy(pVertexDataBegin, triangleVertices, sizeof(triangleVertices));
    m_VertexBuffer->Unmap(0, nullptr);

    // Initialize vertex buffer view
    m_VertexBufferView.BufferLocation = m_VertexBuffer->GetGPUVirtualAddress();
    m_VertexBufferView.StrideInBytes = sizeof(Vertex);
    m_VertexBufferView.SizeInBytes = vertexBufferSize;
}

void Application::InitializeImGui()
{
    // Create descriptor heap for ImGui
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 1;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    CHECK_HR(m_Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_ImGuiDescriptorHeap)), "CreateDescriptorHeap for ImGui failed");

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    CHECK_BOOL(ImGui_ImplSDL2_InitForD3D(m_Window), "ImGui_ImplSDL2_InitForD3D failed");
    CHECK_BOOL(ImGui_ImplDX12_Init(m_Device.Get(), 2,
        DXGI_FORMAT_R8G8B8A8_UNORM, m_ImGuiDescriptorHeap.Get(),
        m_ImGuiDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        m_ImGuiDescriptorHeap->GetGPUDescriptorHandleForHeapStart()), "ImGui_ImplDX12_Init failed");
}

std::vector<char> Application::LoadShader(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    CHECK_BOOL(file.is_open(), "Failed to open shader file: " + filename);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    return buffer;
}

std::vector<char> Application::CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target)
{
    // Load HLSL source
    std::ifstream file(filename);
    CHECK_BOOL(file.is_open(), "Failed to open HLSL file: " + filename);

    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Create DXC compiler and utils
    Microsoft::WRL::ComPtr<IDxcUtils> dxcUtils;
    Microsoft::WRL::ComPtr<IDxcCompiler3> dxcCompiler;

    CHECK_HR(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils)), "DxcCreateInstance for DxcUtils failed");
    CHECK_HR(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler)), "DxcCreateInstance for DxcCompiler failed");

    // Create blob from source
    Microsoft::WRL::ComPtr<IDxcBlobEncoding> sourceBlob;
    CHECK_HR(dxcUtils->CreateBlob(source.c_str(), source.size(), CP_UTF8, &sourceBlob), "CreateBlob failed");

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
    CHECK_HR(dxcCompiler->Compile(&sourceBuffer, arguments.data(), arguments.size(), nullptr, IID_PPV_ARGS(&result)), "Compile failed");

    // Check compilation result
    HRESULT hr;
    CHECK_HR(result->GetStatus(&hr), "GetStatus failed");

    if (!SUCCEEDED(hr))
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

        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Shader compilation failed for %s (%s -> %s)", filename.c_str(), entryPoint.c_str(), target.c_str());
        assert(false);
    }

    // Get compiled shader
    Microsoft::WRL::ComPtr<IDxcBlob> shaderBlob;
    CHECK_HR(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr), "GetOutput failed");

    std::vector<char> compiledShader(shaderBlob->GetBufferSize());
    memcpy(compiledShader.data(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());

    return compiledShader;
}

void Application::Shutdown()
{
    // Wait for the GPU to be done with all resources
    WaitForPreviousFrame();

    // Shutdown ImGui
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    if (m_FenceEvent)
    {
        CloseHandle(m_FenceEvent);
        m_FenceEvent = nullptr;
    }

    if (m_Window)
    {
        SDL_DestroyWindow(m_Window);
        m_Window = nullptr;
    }

    SDL_Quit();
    std::cout << "Application shutdown complete." << std::endl;
}

void Application::ProcessEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        // Pass events to ImGui first
        ImGui_ImplSDL2_ProcessEvent(&event);

        switch (event.type)
        {
        case SDL_QUIT:
            m_IsRunning = false;
            break;

        case SDL_KEYDOWN:
            if (event.key.keysym.sym == SDLK_ESCAPE)
            {
                m_IsRunning = false;
            }
            break;

        default:
            break;
        }
    }
}

void Application::Update(float deltaTime)
{
    // Update game logic here
    // For now, just a placeholder
}

void Application::Render()
{
    // Record commands
    CHECK_HR(m_CommandAllocator->Reset(), "CommandAllocator Reset failed");
    CHECK_HR(m_CommandList->Reset(m_CommandAllocator.Get(), m_PipelineState.Get()), "CommandList Reset failed");

    // Set necessary state
    m_CommandList->SetGraphicsRootSignature(m_RootSignature.Get());
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
    m_CommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    // Clear the render target with configurable background color
    const float clearColor[] = { m_BackgroundColor[0], m_BackgroundColor[1], m_BackgroundColor[2], 1.0f };
    m_CommandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

    // Draw triangle
    m_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_CommandList->IASetVertexBuffers(0, 1, &m_VertexBufferView);
    m_CommandList->DrawInstanced(3, 1, 0, 0);

    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    // Render ImGui UI
    RenderImGui();

    // Render ImGui draw data
    ImGui::Render();
    ID3D12DescriptorHeap* descriptorHeaps[] = { m_ImGuiDescriptorHeap.Get() };
    m_CommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_CommandList.Get());

    // Indicate that the back buffer will now be used to present
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    m_CommandList->ResourceBarrier(1, &barrier);

    CHECK_HR(m_CommandList->Close(), "CommandList Close failed");

    // Execute the command list
    ID3D12CommandList* ppCommandLists[] = { m_CommandList.Get() };
    m_CommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present the frame
    CHECK_HR(m_SwapChain->Present(1, 0), "Present failed");

    // Wait for the GPU to finish
    WaitForPreviousFrame();
}

void Application::WaitForPreviousFrame()
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

void Application::RenderImGui()
{
    // Create a simple debug window
    ImGui::Begin("Renderer Debug");

    // RGB Color Picker for background
    ImGui::ColorEdit3("Background Color", m_BackgroundColor);

    // Display current FPS (basic implementation)
    static float lastTime = 0.0f;
    static int frameCount = 0;
    static float fps = 0.0f;

    float currentTime = SDL_GetTicks() / 1000.0f;
    frameCount++;

    if (currentTime - lastTime >= 1.0f)
    {
        fps = frameCount / (currentTime - lastTime);
        frameCount = 0;
        lastTime = currentTime;
    }

    ImGui::Text("FPS: %.1f", fps);

    ImGui::End();
}