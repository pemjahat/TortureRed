#include "pch.h"

#include "Application.h"
#include <SDL_syswm.h>
#include <DirectXCollision.h>

const char* WINDOW_TITLE = "TortureRed";

Application::Application()
    : m_IsRunning(false)
    , m_Window(nullptr)
    , m_RightMouseButtonHeld(false)
    , m_LastMouseX(0)
    , m_LastMouseY(0)
    , m_FrameConstants{}
{
    m_LastViewMatrix = DirectX::XMMatrixIdentity();
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
    m_Renderer.CreateLightsBuffer();
    m_Renderer.CreateLightLUTBuffer();

    // Set camera projection parameters
    float aspectRatio = static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT;
    float fovY = 60.0f * (3.14159265359f / 180.0f); // 60 degrees
    float nearZ = 0.1f;
    float farZ = 1000.0f;
    m_Camera.SetProjectionParameters(fovY, aspectRatio, nearZ, farZ);
    
    m_FrameConstants.enableRestir = 0;
    m_FrameConstants.enableAvoidCaustics = 1;
    m_FrameConstants.enableIndirectSpecular = 0;
    m_FrameConstants.enableReservoirLobeCheck = 1;
    m_FrameConstants.enableNrdRelax = 1;
    m_FrameConstants.enableNrdValidation = 0;
    m_FrameConstants.rtrRoughReuseThreshold = 0.6f;
    m_FrameConstants.lightSamplingMode = 0; // 0=uniform, 1=importance, 2=brute force
    m_FrameConstants.sharcSceneScale = 50.0f;
    m_FrameConstants.sharcAccumulationFrameNum = 128;
    m_FrameConstants.sharcStaleFrameNum = 32;
    m_FrameConstants.sharcDebug = 0;
    m_FrameConstants.restirReservoirDebugMode = RESTIR_RESERVOIR_DEBUG_OFF;

    // Load Scene
    if (!m_Scene.LoadScene("Content/Scenes/sponza.scene.json"))
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load scene");
        // Fallback or exit? For now just log
    }

    // Load GLTF model
    std::string modelPath = m_Scene.GetModelPath();
    if (modelPath.empty()) modelPath = "Content/Sponza/Sponza.gltf"; // Default

    if (!m_Model.LoadGLTFModel(&m_Renderer, modelPath))
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to load GLTF model");
    }

    // Upload textures to GPU
    m_Model.UploadTextures(m_Renderer.GetDevice(), m_Renderer.GetCommandList(), m_Renderer.GetCommandQueue(), m_Renderer.GetCommandAllocator(), &m_Renderer);

    // Build ray tracing acceleration structures
    m_Renderer.BuildAccelerationStructures(&m_Model);

    // Initialize ImGui
    InitializeImGui();

    // Initialize/Upload lights
    if (m_Scene.GetLights().empty())
    {
        // Fallback default light if scene has none
        LightConstants defaultLight = {};
        defaultLight.color = { 1.0f, 0.9f, 0.8f, 1.0f };
        defaultLight.intensity = 1.0f;
        defaultLight.direction = { -1.0f, -1.0f, 1.0f, 0.0f };
        defaultLight.position = { 0.0f, 10.0f, 0.0f, 1.0f };
        m_Scene.GetLights().push_back(defaultLight);
    }
    
    m_Renderer.UpdateLightsBuffer(m_Scene.GetLights());

    // Initialize Rasterizer Indirect GI Resources
    m_Renderer.CreateRasterIndirectGIResources();
    m_Renderer.CreateRasterIndirectGIPipelines();

    m_LastViewMatrix = m_Camera.GetViewMatrix();
    DirectX::XMStoreFloat4x4(&m_LastViewProj, m_Camera.GetViewMatrix() * m_Camera.GetProjMatrix());
    DirectX::XMStoreFloat4x4(&m_LastViewInverse, m_Camera.GetInvViewMatrix());
    m_LastCameraPos = { m_Camera.GetPosition().x, m_Camera.GetPosition().y, m_Camera.GetPosition().z, 1.0f };

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
            else if (event.button.button == SDL_BUTTON_LEFT)
            {
                if (!ImGui::GetIO().WantCaptureMouse)
                {
                    m_FrameConstants.mouseSelectedPixelX = (uint32_t)event.button.x;
                    m_FrameConstants.mouseSelectedPixelY = (uint32_t)event.button.y;
                    m_PathVizJustClicked = true;
                }
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
    // Update camera (handles W, S, A, D movement)
    m_Camera.Update(deltaTime);

    // Update model animation
    m_Model.UpdateAnimation(deltaTime);

    // Check for changed shader files and hot-reload affected PSOs
    m_Renderer.CheckAndReloadShaders();

    // Compute view-projection matrix
    DirectX::XMMATRIX view = m_Camera.GetViewMatrix();
    DirectX::XMMATRIX proj = m_Camera.GetProjMatrix();
    m_ViewProj = view * proj;

    // Reset frame index if camera moved or rotated (checked against previous frame's view matrix)
    bool cameraMoved = false;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float diff = m_LastViewMatrix.r[i].m128_f32[j] - view.r[i].m128_f32[j];
            if (diff > 1e-4f || diff < -1e-4f) {
                cameraMoved = true;
                break;
            }
        }
    }

    if (cameraMoved && m_UsePathTracer) {
        m_FrameConstants.frameIndex = 0;
    }

    // Save view matrix for next frame's comparison
    m_LastViewMatrix = view;

    // Update Frame Constants
    m_FrameConstants.viewProjPrevious = m_LastViewProj;
    m_FrameConstants.viewInversePrevious = m_LastViewInverse;
    m_FrameConstants.prevCameraPosition = m_LastCameraPos;

    DirectX::XMStoreFloat4x4(&m_FrameConstants.viewProj, m_ViewProj);
    DirectX::XMStoreFloat4x4(&m_FrameConstants.viewInverse, m_Camera.GetInvViewMatrix());
    DirectX::XMStoreFloat4x4(&m_FrameConstants.projectionInverse, DirectX::XMMatrixInverse(nullptr, proj));
    m_FrameConstants.cameraPosition = { m_Camera.GetPosition().x, m_Camera.GetPosition().y, m_Camera.GetPosition().z, 1.0f };
    
    m_LastViewProj = m_FrameConstants.viewProj;
    m_LastViewInverse = m_FrameConstants.viewInverse;
    m_LastCameraPos = m_FrameConstants.cameraPosition;

    // Increment frame index only if not reset
    m_FrameConstants.frameIndex++;

    const auto& gbuffer = m_Renderer.GetGBuffer();
    m_FrameConstants.albedoIndex = gbuffer.albedo.srvIndex;
    m_FrameConstants.normalIndex = gbuffer.normal.srvIndex;
    m_FrameConstants.materialIndex = gbuffer.material.srvIndex;
    m_FrameConstants.depthIndex = gbuffer.depth.srvIndex;
    m_FrameConstants.exposure = m_Exposure;
    m_FrameConstants.numLights = (uint32_t)m_Scene.GetLights().size();
    m_FrameConstants.lightLUTBufferIndex = m_Renderer.GetLightLUTDescriptorIndex();
    m_FrameConstants.screenWidth = (uint32_t)WINDOW_WIDTH;
    m_FrameConstants.screenHeight = (uint32_t)WINDOW_HEIGHT;

    // Update Light in scene and then sync
    if (!m_Scene.GetLights().empty())
    {
        LightConstants& sun = m_Scene.GetLights()[0];
        
        DirectX::XMVECTOR lightDir = DirectX::XMLoadFloat4(&sun.direction);
        DirectX::XMVECTOR lightPos = DirectX::XMVectorScale(lightDir, -20.0f); // Position light back along direction
        DirectX::XMMATRIX lightView = DirectX::XMMatrixLookToRH(lightPos, lightDir, DirectX::XMVectorSet(0, 1, 0, 0));
        DirectX::XMMATRIX lightProj = DirectX::XMMatrixOrthographicRH(40.0f, 40.0f, 0.1f, 100.0f);
        DirectX::XMMATRIX lightViewProj = lightView * lightProj;
        DirectX::XMStoreFloat4x4(&sun.viewProj, lightViewProj);
    }

    m_Renderer.UpdateLightsBuffer(m_Scene.GetLights());

    // Path visualization: enable for exactly one frame after left-click
    m_FrameConstants.pathVizEnabled = m_PathVizJustClicked ? 1u : 0u;
    if (m_PathVizJustClicked)
    {
        m_PathVizEverCaptured = true;
        m_PathVizJustClicked = false;
    }

    m_Renderer.UpdateFrameCB(m_FrameConstants);
}

void Application::Render()
{
    // Begin frame rendering
    m_Renderer.BeginFrame();

    auto cmdList = m_Renderer.GetCommandList();
    auto& gbuffer = m_Renderer.GetGBuffer();
    const bool usePathTracingFrame = m_UsePathTracer && m_Renderer.IsRayTracingSupported();

    // Reset viewport and scissor for main pass
    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(WINDOW_WIDTH), static_cast<float>(WINDOW_HEIGHT));
    D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    cmdList->RSSetViewports(1, &viewport);
    cmdList->RSSetScissorRects(1, &scissorRect);

    // Set camera viewProj to root param 0
    m_Renderer.UpdateFrameCB(m_FrameConstants);
    cmdList->SetGraphicsRootConstantBufferView(0, m_Renderer.GetFrameGPUAddress());

    // Compute frustum for culling
    DirectX::XMMATRIX proj = m_Camera.GetProjMatrix();
    DirectX::BoundingFrustum frustum(proj, true);

    // Transform frustum to world space (inverse view matrix)
    DirectX::XMMATRIX invView = m_Camera.GetInvViewMatrix();
    frustum.Transform(frustum, invView);
    if (usePathTracingFrame)
    {
        // Transition G-Buffer to NON_PIXEL_SHADER_RESOURCE for Path Tracer (Compute)
        GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.albedo, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.normal, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.material, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        m_Renderer.DispatchRays(&m_Model, m_FrameConstants, m_Scene.GetLights()[0]);
        m_Renderer.CopyTextureToBackBuffer(m_Renderer.GetPathTracerOutput());

        // Draw path visualization lines on top of the PT output (non-RTXDI ReSTIR only)
        if (m_PathVizEverCaptured && m_FrameConstants.enableRestir && !m_FrameConstants.useRTXDI)
            m_Renderer.DrawPathVizLines(m_FrameConstants);

        // Setup viewport and RTV for ImGui rendering on top of PT output
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_Renderer.GetCurrentBackBufferRTV();
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
    }
    else
    {
        // 1. Depth Pre-Pass
        {
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            cmdList->SetPipelineState(m_Renderer.GetDepthPrePassPSO());

            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = gbuffer.depth.dsvHandle;
            cmdList->OMSetRenderTargets(0, nullptr, FALSE, &dsvHandle);
            cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

            m_Model.Render(cmdList, &m_Renderer, frustum, AlphaMode::Opaque);
        }
        // 2. G-Buffer Pass
        {
            // Transition G-Buffer targets to RTV state
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.albedo, D3D12_RESOURCE_STATE_RENDER_TARGET);
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.normal, D3D12_RESOURCE_STATE_RENDER_TARGET);
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.material, D3D12_RESOURCE_STATE_RENDER_TARGET);

            float clearColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
            cmdList->ClearRenderTargetView(gbuffer.albedo.rtvHandle, clearColor, 0, nullptr);
            cmdList->ClearRenderTargetView(gbuffer.normal.rtvHandle, clearColor, 0, nullptr);
            cmdList->ClearRenderTargetView(gbuffer.material.rtvHandle, clearColor, 0, nullptr);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = { gbuffer.albedo.rtvHandle, gbuffer.normal.rtvHandle, gbuffer.material.rtvHandle };
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = gbuffer.depth.dsvHandle;

            // If pre-pass was skipped, we MUST clear the depth buffer here
            if (!m_EnableDepthPrePass)
            {
                GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
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
        
        // 2.5 Rasterizer Indirect GI Passes
        m_Renderer.DispatchRasterIndirectGI(&m_Model, m_FrameConstants);

        // 3. Lighting Pass
        {
            BindlessIndices indices = {};

            // Transition G-Buffer targets to SRV state
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.albedo, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.normal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.material, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.depth, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            
            if (m_FrameConstants.enableRasterIndirectGI)
            {
                GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), m_Renderer.GetRasterIndirectLightingTex(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            }

            // Transition backbuffer to RTV
            m_Renderer.TransitionBackBuffer(D3D12_RESOURCE_STATE_RENDER_TARGET);

            // Need this binding for shadow ray in pixel shader
            cmdList->SetGraphicsRootShaderResourceView(1, m_Model.GetMaterialBufferAddress());
            cmdList->SetGraphicsRootShaderResourceView(2, m_Model.GetDrawNodeBufferAddress());
            cmdList->SetGraphicsRootShaderResourceView(5, m_Model.GetGlobalIndexBufferAddress());
            cmdList->SetGraphicsRootShaderResourceView(6, m_Model.GetGlobalVertexBufferAddress());

            if (m_FrameConstants.enableRasterIndirectGI)
            {
                indices.InputIdx0 = m_Renderer.GetRasterIndirectLightingTex().srvIndex;    
            }
            cmdList->SetGraphicsRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices

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
            GraphicsHelper::TransitionResource(m_Renderer.GetCommandList(), gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_READ);

            cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

            if (m_Renderer.GetTransparentPSO())
            {
                cmdList->SetPipelineState(m_Renderer.GetTransparentPSO());
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
        if (ImGui::Checkbox("Use Path Tracer", &m_UsePathTracer))
        {
            m_FrameConstants.frameIndex = 0;
        }

        if (m_UsePathTracer)
        {
            ImGui::Indent();
            bool enableRestir = (m_FrameConstants.enableRestir != 0);
            if (ImGui::Checkbox("Enable ReSTIR GI", &enableRestir))
            {
                m_FrameConstants.enableRestir = enableRestir ? 1 : 0;
                m_FrameConstants.frameIndex = 0;
            }

            bool useRTXDI = (m_FrameConstants.useRTXDI != 0);
            if (ImGui::Checkbox("Use NVIDIA RTXDI", &useRTXDI))
            {
                m_FrameConstants.useRTXDI = useRTXDI ? 1 : 0;
                m_FrameConstants.frameIndex = 0;
            }
            
            // Local-light sampling mode. The main directional light remains exact and shadowed.
            const char* samplingModes[] = { "Uniform Local", "Importance Local (LUT)", "All Local Lights (Brute Force)" };
            int currentMode = (int)m_FrameConstants.lightSamplingMode;
            if (ImGui::Combo("Local Light Sampling", &currentMode, samplingModes, IM_ARRAYSIZE(samplingModes)))
            {
                m_FrameConstants.lightSamplingMode = (uint32_t)currentMode;
                m_FrameConstants.frameIndex = 0;
            }

            // Path Visualization
            if (m_FrameConstants.enableRestir && !m_FrameConstants.useRTXDI)
            {
                ImGui::Separator();
                ImGui::Text("Path Visualization");
                if (m_PathVizEverCaptured)
                {
                    ImGui::Text("Captured at pixel (%u, %u)",
                        m_FrameConstants.mouseSelectedPixelX,
                        m_FrameConstants.mouseSelectedPixelY);
                    if (ImGui::Button("Clear Capture"))
                        m_PathVizEverCaptured = false;
                }
                else
                {
                    ImGui::TextDisabled("Left-click viewport to capture a path");
                }
            }

            ImGui::Unindent();
        }
    }
    else
    {
        ImGui::TextDisabled("Path Tracer (DXR not supported)");
    }

    if (!m_UsePathTracer)
    {
        bool enableRasterIndirectGI = (m_FrameConstants.enableRasterIndirectGI != 0);
        if (ImGui::Checkbox("Enable Raster Indirect GI", &enableRasterIndirectGI))
        {
            m_FrameConstants.enableRasterIndirectGI = enableRasterIndirectGI ? 1 : 0;
        }

        if (enableRasterIndirectGI)
        {
            bool enableNrdRelax = (m_FrameConstants.enableNrdRelax != 0);
            if (ImGui::Checkbox("Enable NRD RELAX", &enableNrdRelax))
            {
                m_FrameConstants.enableNrdRelax = enableNrdRelax ? 1 : 0;
                m_FrameConstants.frameIndex = 0;
            }

            bool enableNrdValidation = (m_FrameConstants.enableNrdValidation != 0);
            if (!enableNrdRelax)
                ImGui::BeginDisabled();
            if (ImGui::Checkbox("NRD Validation Debug", &enableNrdValidation))
            {
                m_FrameConstants.enableNrdValidation = enableNrdValidation ? 1 : 0;
                m_FrameConstants.frameIndex = 0;
            }
            if (!enableNrdRelax)
                ImGui::EndDisabled();

            const char* sharcDebugModes[] = { "Off", "SHaRC Output", "Bounce Heatmap" };
            int sharcDebugMode = (int)m_FrameConstants.sharcDebug;
            ImGui::SetNextItemWidth(180.f);
            if (ImGui::Combo("Debug Vis Mode", &sharcDebugMode, sharcDebugModes, 3))
                m_FrameConstants.sharcDebug = (uint32_t)sharcDebugMode;
        }
    }

    // Shared Indirect GI options — apply to both path tracer and raster indirect GI
    //if (ImGui::CollapsingHeader("Tracing Options"))
    ImGui::SeparatorText("Tracing Options");
    {
        bool avoidCaustics = (m_FrameConstants.enableAvoidCaustics != 0);
        if (ImGui::Checkbox("Avoid Caustic Paths", &avoidCaustics))
        {
            m_FrameConstants.enableAvoidCaustics = avoidCaustics ? 1 : 0;
            m_FrameConstants.frameIndex = 0;
        }

        bool enableIndirectSpecular = (m_FrameConstants.enableIndirectSpecular != 0);
        if (ImGui::Checkbox("Enable Indirect Specular", &enableIndirectSpecular))
        {
            m_FrameConstants.enableIndirectSpecular = enableIndirectSpecular ? 1 : 0;
            m_FrameConstants.frameIndex = 0;
        }

        bool enableReservoirLobeCheck = (m_FrameConstants.enableReservoirLobeCheck != 0);
        if (ImGui::Checkbox("Reservoir Lobe Check (Temporal+Spatial)", &enableReservoirLobeCheck))
        {
            m_FrameConstants.enableReservoirLobeCheck = enableReservoirLobeCheck ? 1 : 0;
            m_FrameConstants.frameIndex = 0;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Lobe-matched target PDF, roughness-scaled Jacobian/history caps.\nDisable to compare against unguarded reuse.");

        if (ImGui::SliderFloat("RTR Rough Reuse Threshold", &m_FrameConstants.rtrRoughReuseThreshold, 0.3f, 1.0f, "%.2f"))
        {
            m_FrameConstants.frameIndex = 0;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Roughness above which specular pass reuses diffuse candidate ray\ninstead of tracing its own VNDF ray (Kajiya strategy).");
    }

    ImGui::SeparatorText("Reservoir Debug");
    {
        const bool supportsReservoirDebug =
            (m_UsePathTracer && (m_FrameConstants.enableRestir || m_FrameConstants.useRTXDI)) ||
            (!m_UsePathTracer && m_FrameConstants.enableRasterIndirectGI);

        if (!supportsReservoirDebug)
        {
            ImGui::BeginDisabled();
        }

        const char* reservoirDebugModes[] = {
            "Off",
            "Position",
            "Normal",
            "Radiance",
            "WeightSum",
            "Source PDF (Collect)",
            "Target PDF (Collect)",
            "Target Shape (Collect)",
            "Target PDF (Temporal)",
            "Target PDF (Spatial)",
            "W"
        };

        int debugMode = static_cast<int>(m_FrameConstants.restirReservoirDebugMode);
        if (ImGui::Combo("Reservoir Field", &debugMode, reservoirDebugModes, IM_ARRAYSIZE(reservoirDebugModes)))
        {
            m_FrameConstants.restirReservoirDebugMode = static_cast<uint32_t>(debugMode);
            m_FrameConstants.frameIndex = 0;
        }

        if (!supportsReservoirDebug)
        {
            ImGui::EndDisabled();
            ImGui::TextDisabled("Enable ReSTIR path tracing or Raster Indirect GI to view reservoir fields.");
        }
    }

    ImGui::Separator();
    ImGui::Text("Light Editor");
    
    std::vector<LightConstants>& lights = m_Scene.GetLights();
    if (!lights.empty())
    {
        std::vector<std::string> lightNames;
        for (size_t i = 0; i < lights.size(); ++i) {
            std::string type = (lights[i].direction.w < 0.5f) ? "Dir" : "Spot";
            lightNames.push_back("Light " + std::to_string(i) + " (" + type + ")");
        }

        if (ImGui::BeginCombo("Select Light", lightNames[m_SelectedLightIndex].c_str()))
        {
            for (int i = 0; i < (int)lights.size(); ++i)
            {
                bool isSelected = (m_SelectedLightIndex == i);
                if (ImGui::Selectable(lightNames[i].c_str(), isSelected))
                {
                    m_SelectedLightIndex = i;
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        LightConstants& selectedLight = lights[m_SelectedLightIndex];
        bool changed = false;

        if (ImGui::DragFloat("Intensity", &selectedLight.intensity, 0.1f, 0.0f, 1000.0f))
        {
            changed = true;
        }

        if (ImGui::ColorEdit3("Color", &selectedLight.color.x))
        {
            changed = true;
        }

        if (selectedLight.direction.w < 0.5f) // Directional
        {
            if (ImGui::DragFloat3("Direction", &selectedLight.direction.x, 0.01f, -1.0f, 1.0f))
            {
                // Normalize direction
                DirectX::XMVECTOR lightDir = DirectX::XMLoadFloat4(&selectedLight.direction);
                lightDir = DirectX::XMVector3Normalize(lightDir);
                DirectX::XMStoreFloat4(&selectedLight.direction, lightDir);
                changed = true;
            }
        }
        else // Point/Spot
        {
            if (ImGui::DragFloat3("Position", &selectedLight.position.x, 0.1f))
            {
                changed = true;
            }
            
            // If it's a spot light (direction is used)
            if (ImGui::DragFloat3("Direction", &selectedLight.direction.x, 0.01f, -1.0f, 1.0f))
            {
                DirectX::XMVECTOR lightDir = DirectX::XMLoadFloat4(&selectedLight.direction);
                lightDir = DirectX::XMVector3Normalize(lightDir);
                DirectX::XMStoreFloat4(&selectedLight.direction, lightDir);
                changed = true;
            }

            // Draw wireframe sphere around selected light
            ImDrawList* drawList = ImGui::GetBackgroundDrawList();
            
            auto Project = [&](const DirectX::XMVECTOR& pos3D) -> ImVec2 {
                DirectX::XMVECTOR pos2D = DirectX::XMVector3Project(pos3D, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, 1.0f, m_Camera.GetProjMatrix(), m_Camera.GetViewMatrix(), DirectX::XMMatrixIdentity());
                return ImVec2(DirectX::XMVectorGetX(pos2D), DirectX::XMVectorGetY(pos2D));
            };

            float radius = 0.5f;
            ImU32 color = IM_COL32(255, 255, 0, 255); // Yellow
            int segments = 16;

            for (int plane = 0; plane < 3; ++plane)
            {
                ImVec2 prevPoint;
                bool prevValid = false;
                for (int i = 0; i <= segments; ++i)
                {
                    float angle = (float)i / segments * DirectX::XM_2PI;
                    float c = cosf(angle) * radius;
                    float s = sinf(angle) * radius;
                    
                    DirectX::XMVECTOR point3D;
                    if (plane == 0) point3D = DirectX::XMVectorSet(selectedLight.position.x + c, selectedLight.position.y + s, selectedLight.position.z, 1.0f);
                    else if (plane == 1) point3D = DirectX::XMVectorSet(selectedLight.position.x, selectedLight.position.y + c, selectedLight.position.z + s, 1.0f);
                    else point3D = DirectX::XMVectorSet(selectedLight.position.x + c, selectedLight.position.y, selectedLight.position.z + s, 1.0f);
                    
                    DirectX::XMVECTOR viewPos = DirectX::XMVector3Transform(point3D, m_Camera.GetViewMatrix());
                    bool valid = DirectX::XMVectorGetZ(viewPos) >= 0.1f;

                    if (valid)
                    {
                        ImVec2 p = Project(point3D);
                        if (i > 0 && prevValid)
                        {
                            drawList->AddLine(prevPoint, p, color, 2.0f);
                        }
                        prevPoint = p;
                    }
                    prevValid = valid;
                }
            }
        }

        if (changed)
        {
            m_FrameConstants.frameIndex = 0; // Reset path tracer if light changes
        }
    }
    else
    {
        ImGui::Text("No lights in scene.");
    }
    
    if (ImGui::DragFloat("Exposure", &m_Exposure, 0.01f, 0.0f, 10.0f))
    {
        // exposure doesn't require reset as it's just post-process
    }

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

    ImGui::Separator();
    ImGui::Checkbox("Show ImGui Demo Window", &m_ShowDemoWindow);

    ImGui::End();

    if (m_ShowDemoWindow)
        ImGui::ShowDemoWindow(&m_ShowDemoWindow);
}