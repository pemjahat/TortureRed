#include "Application.h"
#include <SDL.h>
#include <SDL_syswm.h>
#include <iostream>
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <DirectXCollision.h>

const char* WINDOW_TITLE = "TortureRed";

Application::Application()
    : m_IsRunning(false)
    , m_Window(nullptr)
    , m_RightMouseButtonHeld(false)
    , m_LastMouseX(0)
    , m_LastMouseY(0)
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

    // Initialize renderer
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(m_Window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;

    CHECK_BOOL(m_Renderer.Initialize(hwnd), "Renderer initialization failed");

    // Set camera projection parameters
    float aspectRatio = static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT;
    float fovY = 60.0f * (3.14159265359f / 180.0f); // 60 degrees
    float nearZ = 0.1f;
    float farZ = 1000.0f;
    m_Camera.SetProjectionParameters(fovY, aspectRatio, nearZ, farZ);

    // Load GLTF model
    if (!m_Model.LoadGLTFModel(&m_Renderer, "Content/CesiumMilkTruck/CesiumMilkTruck.gltf"))
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load GLTF model");
    }

    // Upload textures to GPU
    m_Model.UploadTextures(m_Renderer.GetDevice(), m_Renderer.GetCommandList(), m_Renderer.GetCommandQueue(), m_Renderer.GetCommandAllocator(), &m_Renderer);

    // Build ray tracing acceleration structures
    m_Renderer.BuildAccelerationStructures(&m_Model);

    // Initialize ImGui
    InitializeImGui();

    // Initialize directional light
    m_MainLight.color = { 1.0f, 0.9f, 0.8f, 1.0f };
    m_MainLight.direction = { -1.0f, -1.0f, 1.0f, 0.0f };
    m_MainLight.position = { 0.0f, 10.0f, 0.0f, 1.0f }; // Not used for dir light but good to have

    std::cout << "TortureRed application initialized successfully!" << std::endl;
}

void Application::InitializeImGui()
{
    // Create descriptor heap for ImGui
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 1;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    CHECK_HR(m_Renderer.GetDevice()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_ImGuiDescriptorHeap)), "CreateDescriptorHeap for ImGui failed");

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
    CHECK_BOOL(ImGui_ImplDX12_Init(m_Renderer.GetDevice(), 2,
        DXGI_FORMAT_R8G8B8A8_UNORM, m_ImGuiDescriptorHeap.Get(),
        m_ImGuiDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        m_ImGuiDescriptorHeap->GetGPUDescriptorHandleForHeapStart()), "ImGui_ImplDX12_Init failed");
}

void Application::Shutdown()
{
    // Shutdown ImGui
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    // Shutdown renderer (this will handle GPU cleanup)
    m_Renderer.Shutdown();

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

        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_RIGHT)
            {
                m_RightMouseButtonHeld = true;
                m_Camera.SetCameraMode(true);
                SDL_GetMouseState(&m_LastMouseX, &m_LastMouseY);
                SDL_SetRelativeMouseMode(SDL_TRUE); // Capture mouse
            }
            break;

        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_RIGHT)
            {
                m_RightMouseButtonHeld = false;
                m_Camera.SetCameraMode(false);
                SDL_SetRelativeMouseMode(SDL_FALSE); // Release mouse
            }
            break;

        case SDL_MOUSEMOTION:
            if (m_RightMouseButtonHeld)
            {
                int mouseX, mouseY;
                SDL_GetRelativeMouseState(&mouseX, &mouseY);
                m_Camera.ProcessMouseMovement(static_cast<float>(mouseX), static_cast<float>(mouseY));

                // Keep mouse cursor centered to prevent hitting screen edges
                SDL_WarpMouseInWindow(m_Window, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
            }
            break;

        case SDL_MOUSEWHEEL:
            m_Camera.ProcessMouseWheel(static_cast<float>(event.wheel.y));
            break;

        default:
            break;
        }
    }

    // Handle continuous keyboard input for camera movement (always available)
    const Uint8* keyboardState = SDL_GetKeyboardState(nullptr);
    m_Camera.ProcessKeyboard(
        keyboardState[SDL_SCANCODE_W],
        keyboardState[SDL_SCANCODE_S],
        keyboardState[SDL_SCANCODE_A],
        keyboardState[SDL_SCANCODE_D]
    );
}

void Application::Update(float deltaTime)
{
    // Update camera
    m_Camera.Update(deltaTime);

    // Update model animation
    m_Model.UpdateAnimation(deltaTime);

    // Compute view-projection matrix
    DirectX::XMMATRIX view = m_Camera.GetViewMatrix();
    DirectX::XMMATRIX proj = m_Camera.GetProjMatrix();
    m_ViewProj = view * proj;

    // Update Frame Constants
    DirectX::XMStoreFloat4x4(&m_FrameConstants.viewProj, m_ViewProj);
    DirectX::XMStoreFloat4x4(&m_FrameConstants.viewInverse, m_Camera.GetInvViewMatrix());
    DirectX::XMStoreFloat4x4(&m_FrameConstants.projectionInverse, DirectX::XMMatrixInverse(nullptr, proj));
    m_FrameConstants.cameraPosition = { m_Camera.GetPosition().x, m_Camera.GetPosition().y, m_Camera.GetPosition().z, 1.0f };
    m_FrameConstants.frameIndex++;

    const auto& gbuffer = m_Renderer.GetGBuffer();
    m_FrameConstants.albedoIndex = gbuffer.albedo.srvIndex;
    m_FrameConstants.normalIndex = gbuffer.normal.srvIndex;
    m_FrameConstants.materialIndex = gbuffer.material.srvIndex;
    m_FrameConstants.depthIndex = gbuffer.depth.srvIndex;
    m_FrameConstants.shadowMapIndex = m_Renderer.GetShadowMap().srvIndex;

    m_Renderer.UpdateFrameCB(m_FrameConstants);

    // Update Light CB
    DirectX::XMVECTOR lightDir = DirectX::XMLoadFloat4(&m_MainLight.direction);
    DirectX::XMVECTOR lightPos = DirectX::XMVectorScale(lightDir, -20.0f); // Position light back along direction
    DirectX::XMMATRIX lightView = DirectX::XMMatrixLookToLH(lightPos, lightDir, DirectX::XMVectorSet(0, 1, 0, 0));
    DirectX::XMMATRIX lightProj = DirectX::XMMatrixOrthographicLH(40.0f, 40.0f, 0.1f, 100.0f);
    DirectX::XMMATRIX lightViewProj = lightView * lightProj;
    DirectX::XMStoreFloat4x4(&m_MainLight.viewProj, lightViewProj);
    m_Renderer.UpdateLightCB(m_MainLight);
}

void Application::Render()
{
    // Begin frame rendering
    m_Renderer.BeginFrame();

    if (m_UsePathTracer && m_Renderer.IsRayTracingSupported())
    {
        m_Renderer.DispatchRays(&m_Model, m_FrameConstants, m_MainLight);
        m_Renderer.CopyTextureToBackBuffer(m_Renderer.GetPathTracerOutput());
    }
    else
    {
        auto cmdList = m_Renderer.GetCommandList();
        auto& gbuffer = m_Renderer.GetGBuffer();
        auto& shadowMap = m_Renderer.GetShadowMap();

        // 0. Shadow Pass
        {
            m_Renderer.TransitionResource(shadowMap, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            cmdList->ClearDepthStencilView(shadowMap.dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            cmdList->OMSetRenderTargets(0, nullptr, FALSE, &shadowMap.dsvHandle);

            D3D12_VIEWPORT shadowViewport = CD3DX12_VIEWPORT(0.0f, 0.0f, 2048.0f, 2048.0f);
            D3D12_RECT shadowScissor = CD3DX12_RECT(0, 0, 2048, 2048);
            cmdList->RSSetViewports(1, &shadowViewport);
            cmdList->RSSetScissorRects(1, &shadowScissor);

            cmdList->SetPipelineState(m_Renderer.GetShadowPSO());

            // Temporarily bind light viewProj to root param 0 for shadow pass
            cmdList->SetGraphicsRootConstantBufferView(0, m_Renderer.GetLightGPUAddress());

            // Calculate shadow frustum in world space
            DirectX::XMVECTOR lightDir = DirectX::XMLoadFloat4(&m_MainLight.direction);
            DirectX::XMVECTOR lightPos = DirectX::XMVectorScale(lightDir, -20.0f);
            DirectX::XMMATRIX lightView = DirectX::XMMatrixLookToLH(lightPos, lightDir, DirectX::XMVectorSet(0, 1, 0, 0));
            DirectX::XMMATRIX lightProj = DirectX::XMMatrixOrthographicLH(40.0f, 40.0f, 0.1f, 100.0f);
            
            DirectX::BoundingFrustum shadowFrustum(lightProj, false);
            DirectX::XMMATRIX invLightView = DirectX::XMMatrixInverse(nullptr, lightView);
            shadowFrustum.Transform(shadowFrustum, invLightView);

            m_Model.Render(cmdList, &m_Renderer, shadowFrustum, AlphaMode::Opaque);

            m_Renderer.TransitionResource(shadowMap, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        // Reset viewport and scissor for main pass
        D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(WINDOW_WIDTH), static_cast<float>(WINDOW_HEIGHT));
        D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        cmdList->RSSetViewports(1, &viewport);
        cmdList->RSSetScissorRects(1, &scissorRect);
    
        // Restore camera viewProj to root param 0
        cmdList->SetGraphicsRootConstantBufferView(0, m_Renderer.GetFrameGPUAddress());

        // Compute frustum for culling
        DirectX::XMMATRIX proj = m_Camera.GetProjMatrix();
        DirectX::BoundingFrustum frustum(proj, false);

        // Transform frustum to world space (inverse view matrix)
        DirectX::XMMATRIX invView = m_Camera.GetInvViewMatrix();
        frustum.Transform(frustum, invView);

        // 1. Depth Pre-Pass
        if (m_EnableDepthPrePass)
        {
            m_Renderer.TransitionResource(gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            cmdList->SetPipelineState(m_Renderer.GetDepthPrePassPSO());
            
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = gbuffer.depth.dsvHandle;
            cmdList->OMSetRenderTargets(0, nullptr, FALSE, &dsvHandle);
            cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

            m_Model.Render(cmdList, &m_Renderer, frustum, AlphaMode::Opaque);
        }

        // 2. G-Buffer Pass
        {
            // Transition G-Buffer targets to RTV state
            m_Renderer.TransitionResource(gbuffer.albedo, D3D12_RESOURCE_STATE_RENDER_TARGET);
            m_Renderer.TransitionResource(gbuffer.normal, D3D12_RESOURCE_STATE_RENDER_TARGET);
            m_Renderer.TransitionResource(gbuffer.material, D3D12_RESOURCE_STATE_RENDER_TARGET);

            float clearColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
            cmdList->ClearRenderTargetView(gbuffer.albedo.rtvHandle, clearColor, 0, nullptr);
            cmdList->ClearRenderTargetView(gbuffer.normal.rtvHandle, clearColor, 0, nullptr);
            cmdList->ClearRenderTargetView(gbuffer.material.rtvHandle, clearColor, 0, nullptr);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = { gbuffer.albedo.rtvHandle, gbuffer.normal.rtvHandle, gbuffer.material.rtvHandle };
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = gbuffer.depth.dsvHandle;

            // If pre-pass was skipped, we MUST clear the depth buffer here
            if (!m_EnableDepthPrePass)
            {
                m_Renderer.TransitionResource(gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
                cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            }

            cmdList->OMSetRenderTargets(_countof(rtvs), rtvs, FALSE, &dsvHandle);

            if (m_EnableDepthPrePass)
                cmdList->SetPipelineState(m_Renderer.GetGBufferPSO());
            else
                cmdList->SetPipelineState(m_Renderer.GetGBufferWritePSO());

            m_Model.Render(cmdList, &m_Renderer, frustum, AlphaMode::Opaque);
            m_Model.Render(cmdList, &m_Renderer, frustum, AlphaMode::Mask);
        }

        // 3. Lighting Pass
        {
            // Transition G-Buffer targets to SRV state
            m_Renderer.TransitionResource(gbuffer.albedo, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            m_Renderer.TransitionResource(gbuffer.normal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            m_Renderer.TransitionResource(gbuffer.material, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            m_Renderer.TransitionResource(gbuffer.depth, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

            // Transition backbuffer to RTV
            m_Renderer.TransitionBackBuffer(D3D12_RESOURCE_STATE_RENDER_TARGET);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_Renderer.GetCurrentBackBufferRTV();
            cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
            
            const float clearColor[] = { m_Renderer.m_BackgroundColor[0], m_Renderer.m_BackgroundColor[1], m_Renderer.m_BackgroundColor[2], 1.0f };
            cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

            cmdList->SetPipelineState(m_DebugShadowMap ? m_Renderer.GetDebugPSO() : m_Renderer.GetLightingPSO());

            cmdList->DrawInstanced(3, 1, 0, 0); // Fullscreen triangle
        }

        // 4. Transparency Pass (Forward)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_Renderer.GetCurrentBackBufferRTV();
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = gbuffer.depth.dsvHandle;
            
            // Ensure depth is in read state for forward pass
            m_Renderer.TransitionResource(gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_READ);
            
            cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

            // Use the original pipeline state for forward rendering (updated for transparency if needed)
            // For now, we'll use the G-Buffer PSO but modified or just the original one
            if (m_Renderer.GetPipelineState())
            {
                cmdList->SetPipelineState(m_Renderer.GetPipelineState());
                m_Model.Render(cmdList, &m_Renderer, frustum, AlphaMode::Blend);
            }
        }
    }

    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    // Render ImGui UI
    RenderImGui();

    // Render ImGui draw data
    ImGui::Render();
    ID3D12DescriptorHeap* descriptorHeaps[] = { m_ImGuiDescriptorHeap.Get() };
    m_Renderer.GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_Renderer.GetCommandList());

    // End frame rendering (includes present)
    m_Renderer.EndFrame();
}

void Application::RenderImGui()
{
    // Create a simple debug window
    ImGui::Begin("Renderer Debug");

    // RGB Color Picker for background
    ImGui::ColorEdit3("Background Color", m_Renderer.m_BackgroundColor);

    ImGui::Checkbox("Enable Depth Pre-Pass", &m_EnableDepthPrePass);

    ImGui::Checkbox("Debug Shadow Map", &m_DebugShadowMap);

    if (m_Renderer.IsRayTracingSupported())
    {
        ImGui::Checkbox("Use Path Tracer", &m_UsePathTracer);
    }
    else
    {
        ImGui::TextDisabled("Path Tracer (DXR not supported)");
    }

    ImGui::Separator();
    ImGui::Text("Direct Light");
    ImGui::DragFloat3("Direction", &m_MainLight.direction.x, 0.01f, -1.0f, 1.0f);
    ImGui::ColorEdit3("Light Color", &m_MainLight.color.x);
    
    // Normalize light direction
    DirectX::XMVECTOR lightDir = DirectX::XMLoadFloat4(&m_MainLight.direction);
    lightDir = DirectX::XMVector3Normalize(lightDir);
    DirectX::XMStoreFloat4(&m_MainLight.direction, lightDir);

    ImGui::Separator();

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

    // Debug values from Model
    ImGui::Text("Total Nodes Read: %zu", m_Model.GetTotalNodes());
    ImGui::Text("Total Root Nodes: %zu", m_Model.GetTotalRootNodes());
    ImGui::Text("Nodes Survive Frustum: %zu", m_Model.GetNodesSurviveFrustum());

    ImGui::End();
}