#include "Application.h"
#include <SDL.h>
#include <SDL_syswm.h>
#include <iostream>
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

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

    // Load GLTF model
    if (!m_Model.LoadGLTFModel(m_Renderer.GetDevice(), "Content/CesiumMilkTruck/CesiumMilkTruck.gltf"))
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load GLTF model");
    }

    // Upload textures to GPU
    m_Model.UploadTextures(m_Renderer.GetDevice(), m_Renderer.GetCommandList(), m_Renderer.GetCommandQueue(), m_Renderer.GetCommandAllocator(), m_Renderer.GetSRVHeap());

    // Initialize ImGui
    InitializeImGui();

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

    // Update view-projection matrix (camera can move anytime in first-person mode)
    m_Renderer.UpdateViewProjectionMatrix(m_Camera.GetViewMatrix());
}

void Application::Render()
{
    // Begin frame rendering
    m_Renderer.BeginFrame();

    // Render GLTF model
    m_Model.Render(m_Renderer.GetCommandList());

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