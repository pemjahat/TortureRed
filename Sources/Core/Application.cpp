#include "pch.h"

#include "Application.h"
#include <SDL_syswm.h>
#include <DirectXCollision.h>
#include <cstdio>
#include "Rendering/BakedGI.h"

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
        Uint32 frameStart = SDL_GetTicks();
        float deltaTime = (frameStart - lastTime) / 1000.0f;
        lastTime = frameStart;

        ProcessEvents();
        Update(deltaTime);
        Render();

        // Adaptive FPS limiter — sleeps only the remaining budget instead of
        // a fixed 16 ms, so fast frames don't burn CPU and slow frames aren't
        // pushed further behind.
        if (m_FPSLimitEnabled)
        {
            Uint32 frameElapsed = SDL_GetTicks() - frameStart;
            Uint32 targetMs = static_cast<Uint32>(1000.0f / m_TargetFPS);
            if (frameElapsed < targetMs)
                SDL_Delay(targetMs - frameElapsed);
        }
    }
}

void Application::Initialize()
{
    // RenderDoc must enter the process before the D3D12 device is created.
    // Then the capture layer can wrap the device. A capture is impossible when
    // this call happens later.
    m_RenderDoc.Initialize(RenderDocCapture::kDefaultCapturePathTemplate);

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

    // Initialize microprofile
    MicroProfileOnThreadCreate("Main");
    MicroProfileSetEnableAllGroups(true);
    MicroProfileSetForceMetaCounters(true);
    void* commandQueue = m_Renderer.GetCommandQueue();
    void* copyQueue = m_Renderer.GetCopyQueue();
    MicroProfileGpuInitD3D12(m_Renderer.GetDevice(), 1, &commandQueue, &copyQueue);
    MICROPROFILE_CONDITIONAL(MICROPROFILE_GPU_INIT_QUEUE("GPU-Graphics-Queue"));
    MicroProfileSetCurrentNodeD3D12(0);

    m_Renderer.CreateLightsBuffer();
    m_Renderer.CreateLightLUTBuffer();

    // Set camera projection parameters (reverse-Z: nearZ controls NDC depth=1.0 plane)
    float aspectRatio = static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT;
    float fovY = 60.0f * (PI / 180.0f); // 60 degrees
    float nearZ = 0.1f;  // Reverse-Z: near clip at z_view = -1.0
    float farZ  = 0.0f;  // Reverse-Z sentinel: infinite far plane
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
    // Indirect GI defaults: enabled, probe (baked) method — matches
    // m_IndirectGIEnabled/m_IndirectGIMethod; the checkbox + enum in
    // RenderImGui arbitrates the exclusive engine flags from that state.
    m_FrameConstants.bakedGIMode = 1;
    m_FrameConstants.bakedGIValid = 0;
    m_FrameConstants.bakedGIProbeSRVIndex = 0;
    m_FrameConstants.bakedGIProbeMetaSRVIndex = 0;
    m_FrameConstants.bakedGIProbeSkySRVIndex = 0;
    m_FrameConstants.bakedGIProbeSunSRVIndex = 0;
    m_FrameConstants.bakedGIProbeVisSRVIndex = 0;
    m_FrameConstants.bakedGIGridMinX = 0.0f;
    m_FrameConstants.bakedGIGridMinY = 0.0f;
    m_FrameConstants.bakedGIGridMinZ = 0.0f;
    m_FrameConstants.bakedGISpacing = 2.0f;
    m_FrameConstants.bakedGIDimX = 0;
    m_FrameConstants.bakedGIDimY = 0;
    m_FrameConstants.bakedGIDimZ = 0;
    m_FrameConstants.bakedGIDebugView = 0;
    m_FrameConstants.bakedGIResponseSRVIndex = 0;
    m_FrameConstants.bakedGIDebugScale = 1.0f;
    // Ray-debug constants: the SRV index starts at the "no buffer" sentinel
    // (0xFFFFFFFF), not 0, because 0 is a legal descriptor index. The Ray Fan
    // shader early-outs on the sentinel. Application::FillRayDebugConstants
    // overwrites these every frame.
    m_FrameConstants.bakedGIRayDebugSRVIndex = 0xFFFFFFFFu;
    m_FrameConstants.bakedGIRayDebugProbe = 0;
    m_FrameConstants.bakedGIRayDebugChannel = 1;   // albedo
    m_FrameConstants.bakedGIRayDebugLength = 8.0f; // probe spacings
    m_FrameConstants.bakedGIRayDebugDir = 0;       // sun-dominant direction, set each frame
    // Cage Trace (view 9) ray override: the "no override" sentinel, so the first
    // frame traces the shader's own argmax. Application::FillRayDebugConstants
    // overwrites it every frame from m_BakedGIRayDebugRay.
    m_FrameConstants.bakedGIRayDebugRay = BakedGI::kNoRayPick;
    // Efb-contribution balls (view 8): the sentinel until a bake folds the
    // terms, exactly like the ray-debug buffer above. Application::Update
    // overwrites both every frame from the live BakedGI getters.
    m_FrameConstants.bakedGIEfbIterSRVIndex = 0xFFFFFFFFu;
    m_FrameConstants.bakedGIEfbBaseSRVIndex = 0xFFFFFFFFu;
    // Sun-fold weights (the Ray Fan's DeltaDirections channel). Zero until the
    // first Application::Update fills them, so that channel enabled on frame 0
    // paints all 16 boxes black rather than reading whatever the allocation held.
    // The weight lives in .x: the field is a float4 array so that the CPU and the
    // GPU agree on its stride (SharedTypes.h).
    for (uint32_t sunDir = 0; sunDir < BakedGI::kDirections; ++sunDir)
        m_FrameConstants.bakedGISunWeight[sunDir].x = 0.0f;
    m_FrameConstants.bakedGILeakDebug = 0;
    m_FrameConstants.bakedGIVisCheck = 1; // step-3 weights on by default (A/B off = plain step-1 fetch)
    // DDGI defaults: off until the ImGui method selector picks it. The atlas and
    // the probe buffers do not exist until SetupDDGI runs, so ddgiValid stays 0.
    m_FrameConstants.ddgiMode = 0;
    m_FrameConstants.ddgiValid = 0;
    m_FrameConstants.ddgiIrradianceSRVIndex = 0;
    m_FrameConstants.ddgiDepthSRVIndex = 0;
    m_FrameConstants.ddgiProbeStateSRVIndex = 0;
    m_FrameConstants.ddgiProbeOffsetSRVIndex = 0;
    m_FrameConstants.ddgiGridMinX = 0.0f;
    m_FrameConstants.ddgiGridMinY = 0.0f;
    m_FrameConstants.ddgiGridMinZ = 0.0f;
    m_FrameConstants.ddgiSpacing = 2.0f;
    m_FrameConstants.ddgiDimX = 0;
    m_FrameConstants.ddgiDimY = 0;
    m_FrameConstants.ddgiDimZ = 0;
    m_FrameConstants.ddgiProbesPerRow = 0;
    m_FrameConstants.ddgiNumRaysPerProbe = 0;
    m_FrameConstants.ddgiMaxRaysPerProbe = 0;
    m_FrameConstants.ddgiDebugView = 0;
    m_FrameConstants.ddgiDebugScale = 1.0f;
    // Indirect-only scene view: off by default. Source split defaults to the
    // total (sky + sun), matching the normal fetch.
    m_FrameConstants.giIndirectOnly = 0;
    m_FrameConstants.giIndirectOnlySource = 0;
    m_FrameConstants.giIndirectOnlyAlbedo = 0;
    m_FrameConstants.enableRestirDI = 0;
    m_FrameConstants.restirDIDebugMode = RESTIR_DI_DEBUG_OFF;

    // Load Scene
    if (!m_Scene.LoadScene("Content/Scenes/bistro.scene.json"))
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

    // Upload buffers to GPU (material, vertices, indices, indirect commands, meshlet streams)
    m_Model.UploadBuffers(&m_Renderer);

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

    // Baked GI probe system : pipelines + runtime bake after
    // the scene is fully loaded (consumed at the start of the first Render).
    m_Renderer.CreateBakedGIPipelines();
    if (m_IndirectGIEnabled && ProbeMethodActive())
        m_PendingGIProbeBake = true; // auto-bake the default probe method

    // DDGI probe system : pipelines + grid setup. The grid fits the loaded model
    // bounds, so the setup runs at the start of the first Render.
    m_Renderer.CreateDDGIPipelines();
    if (m_IndirectGIEnabled && m_IndirectGIMethod == 2)
        m_PendingDDGISetup = true; // auto-fit the grid when DDGI is the default method

    // Initialize TAA / Temporal Super-Resolution
    // Output is always 1920x1080 (WINDOW_WIDTH x WINDOW_HEIGHT).
    // Internal resolution equals output resolution unless TAA upscaling is active.
    // When TAA is disabled there is no upscaler, so rendering at a lower internal
    // resolution would produce a mismatched depth buffer and incorrect depth tests.
    m_OutputWidth = WINDOW_WIDTH;
    m_OutputHeight = WINDOW_HEIGHT;
    const bool taaDefaultOn = (m_AntiAliasingMode == AA_MODE_TAA);
    m_InternalWidth  = taaDefaultOn ? std::max(320u, std::min((uint32_t)(m_OutputWidth  / m_TaaUpsamplingFactor), m_OutputWidth))  : m_OutputWidth;
    m_InternalHeight = taaDefaultOn ? std::max(180u, std::min((uint32_t)(m_OutputHeight / m_TaaUpsamplingFactor), m_OutputHeight)) : m_OutputHeight;

    // Create all internal-resolution resources (GBuffer, path tracer, NRD, reservoirs, etc.)
    m_Renderer.CreateInternalResolutionResources(m_InternalWidth, m_InternalHeight);
    m_Renderer.CreateTaaResources(m_OutputWidth, m_OutputHeight, m_InternalWidth, m_InternalHeight);
    m_Renderer.CreateTaaPipelines();

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

    // Shutdown microprofile
    MicroProfileShutdown();

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

    // Compute TAA jitter for this frame (only in TAA mode)
    const bool taaActive = (m_AntiAliasingMode == AA_MODE_TAA);
    if (taaActive)
    {
        // Use a dedicated monotonically-increasing counter for the jitter sequence.
        // frameIndex is reset to 0 every frame in TAA mode (to prevent PT accumulation),
        // so it can't be used for jitter — it would freeze the Halton sequence.
        m_TaaFrameCounter++;
        uint32_t N = (uint32_t)std::max(8.0f, std::ceil(m_TaaUpsamplingFactor * m_TaaUpsamplingFactor));
        uint32_t jitterIdx = m_TaaFrameCounter % N;

        // Halton base-2
        float haltonX = 0.0f;
        {
            float f = 0.5f;
            uint32_t i = jitterIdx + 1;
            while (i > 0) { haltonX += f * (i % 2); i /= 2; f *= 0.5f; }
        }
        // Halton base-3
        float haltonY = 0.0f;
        {
            float f = 1.0f / 3.0f;
            uint32_t i = jitterIdx + 1;
            while (i > 0) { haltonY += f * (i % 3); i /= 3; f /= 3.0f; }
        }

        // Center around 0: range [-0.5, 0.5] in pixel units of internal resolution
        m_FrameConstants.taaJitter = { haltonX - 0.5f, haltonY - 0.5f };
    }
    else
    {
        m_FrameConstants.taaJitter = { 0.0f, 0.0f };
    }

    // Apply TAA jitter to projection matrix (only in TAA mode)
    DirectX::XMMATRIX jitteredProj = proj;
    if (taaActive)
    {
        // Offset in clip space: (2 * jitter.x / internalWidth, -2 * jitter.y / internalHeight).
        // Reverse-Z matrix layout: _31 = _32 = 0 (no z→x or z→y coupling), so adding
        // jitter to these elements shifts x'/y' by jitter*z_view, producing the correct
        // sub-pixel offset after perspective divide (same approach as standard-Z).
        float jitterX = 2.0f * m_FrameConstants.taaJitter.x / (float)m_InternalWidth;
        float jitterY = -2.0f * m_FrameConstants.taaJitter.y / (float)m_InternalHeight;
        DirectX::XMFLOAT4X4 projF;
        DirectX::XMStoreFloat4x4(&projF, proj);
        projF._31 += jitterX;
        projF._32 += jitterY;
        jitteredProj = DirectX::XMLoadFloat4x4(&projF);
    }

    m_ViewProj = view * jitteredProj;

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

    // Frame index reset logic driven by anti-aliasing mode:
    //  - No AA:         Reset every frame (single sample, no accumulation)
    //  - Accumulation:  Reset only on camera movement (progressive convergence)
    //  - TAA:           Reset every frame (TAA handles temporal accumulation;
    //                   letting the path tracer also accumulate conflicts with jitter)
    //
    // In TAA mode, frameIndex is forced to 0 so the path tracer outputs a fresh
    // single sample each frame (no progressive accumulation). However, the RNG
    // seed must still vary per frame so TAA sees different noise to converge.
    // We use taaFrameCounter for that — see PathTracer.hlsl.
    if (m_AntiAliasingMode == AA_MODE_NONE && m_UsePathTracer) {
        m_FrameConstants.frameIndex = 0;
    }
    else if (m_AntiAliasingMode == AA_MODE_ACCUMULATION && cameraMoved && m_UsePathTracer) {
        m_FrameConstants.frameIndex = 0;
    }
    else if (m_AntiAliasingMode == AA_MODE_TAA && m_UsePathTracer) {
        m_FrameConstants.frameIndex = 0;
    }

    // Handle TAA history reset (mode/factor changed)
    if (m_TaaResetHistory)
    {
        m_FrameConstants.frameIndex = 0;
        m_TaaResetHistory = false;
    }

    // Save view matrix for next frame's comparison
    m_LastViewMatrix = view;

    // Update Frame Constants
    m_FrameConstants.viewProjPrevious = m_LastViewProj;
    m_FrameConstants.viewInversePrevious = m_LastViewInverse;
    m_FrameConstants.prevCameraPosition = m_LastCameraPos;

    DirectX::XMStoreFloat4x4(&m_FrameConstants.viewProj, m_ViewProj);
    DirectX::XMStoreFloat4x4(&m_FrameConstants.viewInverse, m_Camera.GetInvViewMatrix());
    DirectX::XMStoreFloat4x4(&m_FrameConstants.projectionInverse, DirectX::XMMatrixInverse(nullptr, jitteredProj));
    DirectX::XMStoreFloat4x4(&m_FrameConstants.projectionInverseUnjittered, DirectX::XMMatrixInverse(nullptr, proj));
    m_FrameConstants.cameraPosition = { m_Camera.GetPosition().x, m_Camera.GetPosition().y, m_Camera.GetPosition().z, 1.0f };

    // Store the *unjittered* viewProj as the previous-frame matrix for motion vectors.
    // Motion vectors should encode pure camera motion, not jitter differences.
    // The TAA resolve pass handles jitter compensation separately via its unjitter logic.
    DirectX::XMMATRIX unjitteredViewProj = view * proj;
    DirectX::XMStoreFloat4x4(&m_FrameConstants.viewProjUnjittered, unjitteredViewProj); // post-composite debug overlays
    DirectX::XMFLOAT4X4 unjitteredVP;
    DirectX::XMStoreFloat4x4(&unjitteredVP, unjitteredViewProj);
    m_LastViewProj = unjitteredVP;
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
    m_FrameConstants.screenWidth = m_InternalWidth;
    m_FrameConstants.screenHeight = m_InternalHeight;

    // TAA / Temporal Super-Resolution
    m_FrameConstants.taaEnabled = taaActive ? 1u : 0u;
    m_FrameConstants.internalWidth = m_InternalWidth;
    m_FrameConstants.internalHeight = m_InternalHeight;
    m_FrameConstants.taaUpsamplingFactor = m_TaaUpsamplingFactor;
    m_FrameConstants.outputWidth = m_OutputWidth;
    m_FrameConstants.outputHeight = m_OutputHeight;
    m_FrameConstants.taaHistoryIndex = 0; // Will be set by renderer
    m_FrameConstants.taaFrameCounter = m_TaaFrameCounter;

    // Sky system : bindless indices for cubemap + SH9 buffer.
    m_FrameConstants.skyCubemapIndex  = m_Renderer.GetSkyCubemapSRVIndex();
    m_FrameConstants.skySH9BufferIndex = m_Renderer.GetSkySH9BufferSRVIndex();
    m_FrameConstants.skyTurbidity     = 3.0f;
    m_FrameConstants.skyGroundAlbedo  = 0.3f;

    // Baked GI probe system state.
    {
        BakedGI& gi = m_Renderer.GetBakedGI();
        m_FrameConstants.bakedGIValid             = gi.IsValid() ? 1u : 0u;
        m_FrameConstants.bakedGIProbeSRVIndex      = gi.IsValid() ? gi.GetLitSRVIndex() : 0u;
        m_FrameConstants.bakedGIProbeMetaSRVIndex  = gi.IsValid() ? gi.GetMetaSRVIndex() : 0u;
        m_FrameConstants.bakedGIProbeSkySRVIndex   = gi.IsValid() ? gi.GetLitSkySRVIndex() : 0u;
        m_FrameConstants.bakedGIProbeSunSRVIndex   = gi.IsValid() ? gi.GetLitSunSRVIndex() : 0u;
        m_FrameConstants.bakedGIProbeVisSRVIndex   = gi.IsValid() ? gi.GetVisibilitySRVIndex() : 0u;
        m_FrameConstants.bakedGIResponseSRVIndex   = gi.IsValid() ? gi.GetResponseSRVIndex() : 0u;
        m_FrameConstants.bakedGIDebugView          = (uint32_t)m_BakedGIDebugView;
        m_FrameConstants.bakedGIDebugScale         = m_BakedGIDebugScale;
        FillEfbBallConstants();
        // Ray Fan (view 7) constants. The pending dump runs later, in
        // Application::Render, so the dump-derived values here (SRV index, probe,
        // validity) belong to the PREVIOUS dump. Application::Render calls the
        // same fill again right after the dump, so the frame that binds the
        // constants for the fan carries the new buffer, probe and verdict.
        FillRayDebugConstants();
        // The sun fold for THIS frame: the Ray Fan's direction and the Sun
        // Directions view's 16 weights. Refilled every frame because the sun
        // moves; a dump does not change it, so Render does not call it again.
        FillSunFoldConstants();
        m_FrameConstants.bakedGIVisCheck           = m_BakedGIVisCheck ? 1u : 0u;
        // Indirect-only view state. Lives here because the BakedGI block always
        // runs, whatever indirect method is selected, and the view covers DDGI
        // too — only the sun/sky split is baked-probe specific.
        m_FrameConstants.giIndirectOnly            = m_GIIndirectOnly ? 1u : 0u;
        m_FrameConstants.giIndirectOnlySource      = (uint32_t)m_GIIndirectOnlySource;
        m_FrameConstants.giIndirectOnlyAlbedo      = (uint32_t)m_GIIndirectOnlyAlbedo;
        const DirectX::XMFLOAT3 gmin = gi.GetGridMin();
        m_FrameConstants.bakedGIGridMinX = gmin.x;
        m_FrameConstants.bakedGIGridMinY = gmin.y;
        m_FrameConstants.bakedGIGridMinZ = gmin.z;
        m_FrameConstants.bakedGISpacing  = gi.GetSpacing();
        m_FrameConstants.bakedGIDimX = gi.GetDimX();
        m_FrameConstants.bakedGIDimY = gi.GetDimY();
        m_FrameConstants.bakedGIDimZ = gi.GetDimZ();
        if (m_FrameConstants.bakedGIMode != 0)
            m_FrameConstants.enableRestirGI = 0; // exclusive modes
    }

    // DDGI probe system state. The getters and RecordDDGIFrame both derive the
    // atlas parity from the same serial, so the indices published here always
    // match the pair that the next DDGI record writes.
    {
        DDGI& ddgi = m_Renderer.GetDDGI();
        m_FrameConstants.ddgiValid                = ddgi.IsValid() ? 1u : 0u;
        m_FrameConstants.ddgiIrradianceSRVIndex    = ddgi.IsValid() ? ddgi.GetIrradianceSRVIndex() : 0u;
        m_FrameConstants.ddgiDepthSRVIndex         = ddgi.IsValid() ? ddgi.GetDepthSRVIndex() : 0u;
        m_FrameConstants.ddgiProbeStateSRVIndex    = ddgi.IsValid() ? ddgi.GetProbeStateSRVIndex() : 0u;
        m_FrameConstants.ddgiProbeOffsetSRVIndex   = ddgi.IsValid() ? ddgi.GetProbeOffsetSRVIndex() : 0u;
        m_FrameConstants.ddgiDebugView             = (uint32_t)m_DDGIDebugView;
        m_FrameConstants.ddgiDebugScale            = m_DDGIDebugScale;
        const DirectX::XMFLOAT3 gmin = ddgi.GetGridMin();
        m_FrameConstants.ddgiGridMinX = gmin.x;
        m_FrameConstants.ddgiGridMinY = gmin.y;
        m_FrameConstants.ddgiGridMinZ = gmin.z;
        m_FrameConstants.ddgiSpacing        = ddgi.GetSpacing();
        m_FrameConstants.ddgiDimX           = ddgi.GetDimX();
        m_FrameConstants.ddgiDimY           = ddgi.GetDimY();
        m_FrameConstants.ddgiDimZ           = ddgi.GetDimZ();
        m_FrameConstants.ddgiProbesPerRow   = ddgi.GetProbesPerRow();
        m_FrameConstants.ddgiNumRaysPerProbe = DDGI::kRaysPerProbe;
        m_FrameConstants.ddgiMaxRaysPerProbe = DDGI::kRaysPerProbe;
        if (m_FrameConstants.ddgiMode != 0)
        {
            // Exclusive modes: DDGI replaces the other two indirect sources.
            m_FrameConstants.bakedGIMode   = 0;
            m_FrameConstants.enableRestirGI = 0;
        }
    }

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

    // Freeze culling: refresh the snapshot only while unfrozen, so the toggle
    // always freezes the view at the moment it is enabled. While frozen the
    // snapshot stays stale and the cull dispatches bind it instead of m_FrameCB.
    if (!m_FreezeCulling)
    {
        m_CullFrameConstants = m_FrameConstants;
    }
    m_Renderer.UpdateCullFrameCB(m_CullFrameConstants);

    // Occluded-rect recording toggle — read by the cull dispatches this frame.
    // Recording also runs when only the depth labels need the records.
    m_Renderer.SetOccludedRectDebug(m_OccludedRectDebug || (m_DebugScreenText && m_DebugDepthLabels));
}

// Fills the dump-derived ray-debug FrameConstants (SRV index, probe, channel,
// length cap, validity verdict) from the live BakedGI getters. One helper, two
// call sites (Update and, again, Render right after the dump), so the two fills
// cannot drift.
//
// The SRV index and the probe come from the DUMP, not from the UI fields. The
// dump clamps its index into the grid and stores it, so the fan and the text
// describe one probe even when the UI field changed after a dump.
//
// 0xFFFFFFFF is the "no ray-debug buffer" sentinel. 0 is a legal descriptor
// index, so it cannot mean "none": the fan shader early-outs on the sentinel.
//
// It deliberately does NOT touch bakedGIRayDebugDir: that is the sun fold's
// own output, FillSunFoldConstants owns it, and the post-dump call must not
// re-derive a pick the dump did not change.
void Application::FillRayDebugConstants()
{
    BakedGI& gi = m_Renderer.GetBakedGI();
    const bool hasRayDebugBuffer = gi.HasRayDebugBuffer();
    m_FrameConstants.bakedGIRayDebugSRVIndex = hasRayDebugBuffer ? gi.GetRayDebugSRVIndex() : 0xFFFFFFFFu;
    m_FrameConstants.bakedGIRayDebugProbe    = hasRayDebugBuffer ? gi.GetRayDebugProbe() : 0u;
    m_FrameConstants.bakedGIRayDebugChannel  = (uint32_t)m_BakedGIRayDebugChannel;
    m_FrameConstants.bakedGIRayDebugLength   = m_BakedGIRayDebugLength;
    // Cached by the last dump (BakedGI::GetRayDebugKnownInvalid). True is the
    // no-claim value, so an unknown verdict does not dim the fan.
    m_FrameConstants.bakedGIRayDebugValid    = gi.GetRayDebugKnownInvalid() ? 0u : 1u;
    // Cage Trace ray override. The member's -1 for "shader's own argmax" is
    // published as the sentinel, and anything outside the ray range is treated as
    // "no override" rather than clamped: a clamp would silently trace a DIFFERENT
    // ray than the one the panel names. The shader applies the same test, so the
    // text and the picture agree even if this fill never ran.
    m_FrameConstants.bakedGIRayDebugRay =
        (m_BakedGIRayDebugRay < 0 || (uint32_t)m_BakedGIRayDebugRay >= BakedGI::kDebugGatherRays)
            ? BakedGI::kNoRayPick
            : (uint32_t)m_BakedGIRayDebugRay;
}

// Publishes the two per-bake-term response balls the "Efb Contribution" view
// (bakedGIDebugView == 8) differences — the SELECTED term's and term 1's, the
// no-feedback baseline (see SharedTypes.h) — plus the two transport TABLES and
// the fold direction the view's single-direction mode needs. All of it comes from
// the bake, and every index is BakedGI::kEfbBallNoSRV until one has run, which is
// how the view knows it has nothing to read.
//
// Clamped here as well as at the combo, so a stale member after a
// kBakeIterations change can never index a ball that does not exist.
//
// ONE helper, TWO call sites — Update, and Render right after "Rebuild GI
// Probes". The second is what keeps a fresh bake's indices off the frame the
// bake ran on: Update uploaded the constants BEFORE the bake, so without it the
// view would spend one frame on the sentinel. Same reason, and the same shape,
// as FillRayDebugConstants' post-dump call.
void Application::FillEfbBallConstants()
{
    static_assert(BakedGI::kBakeIterations == 3,
                  "the \"Iteration\" combo has exactly three entries; "
                  "add one when the bake term count changes");

    const BakedGI& gi = m_Renderer.GetBakedGI();
    const int sel = (m_BakedGIEfbIter < 0) ? 0
                  : (m_BakedGIEfbIter >= (int)BakedGI::kBakeIterations
                        ? (int)BakedGI::kBakeIterations - 1 : m_BakedGIEfbIter);
    m_FrameConstants.bakedGIEfbIterSRVIndex = gi.GetEfbBallSRVIndex((uint32_t)sel);
    m_FrameConstants.bakedGIEfbBaseSRVIndex = gi.GetEfbBallSRVIndex(0u);

    // Single-direction mode. The shader's test is "is this a direction index?", so
    // the member's -1 for "All" needs no translation: it publishes as 0xFFFFFFFF
    // and the view falls through to the folded ball. Clamped like the term above,
    // so a stale value can only ever land on a direction that exists.
    if (m_BakedGIEfbDir < -1 || m_BakedGIEfbDir >= (int)BakedGI::kDirections)
        m_BakedGIEfbDir = -1;
    m_FrameConstants.bakedGIEfbTermSelSRVIndex  = gi.GetEfbTermSRVIndex((uint32_t)sel);
    m_FrameConstants.bakedGIEfbTermBaseSRVIndex = gi.GetEfbTermSRVIndex(0u);
    m_FrameConstants.bakedGIEfbDir = (uint32_t)m_BakedGIEfbDir;
}

// Publishes the sun fold for this frame, in both forms its two consumers need:
//   - the 16 normalised kernel weights, for the Ray Fan's DeltaDirections
//     channel (FrameConstants::bakedGISunWeight), and
//   - the argmax and its weight, for the Ray Fan's direction
//     (FrameConstants::bakedGIRayDebugDir) and for the UI text,
// and the to-sun direction itself, for the 17th box that same channel draws
// (FrameConstants::bakedGISunDir).
//
// All of it comes from ONE call to BakedGI::SunFoldWeights, the only owner of
// the kernel, the normalisation and the argmax. That is what keeps the white
// box, the fan's drawn direction, the orange sun marker and the printed weight
// ranking in agreement.
//
// The transport table is indexed by TO-SUN directions and the light stores the
// direction it TRAVELS, so negate it — the same negation the update pass and the
// pending-dump block use. With no sun every weight is 0 and the argmax is 0
// with weight 0, which is the "no sun" report, not a winner.
void Application::FillSunFoldConstants()
{
    float w[BakedGI::kDirections];
    float wMax = 0.0f;
    uint32_t dominant = 0u;

    // Kept in scope after the branch: the sun marker needs the direction, not
    // only the weights. Left at zero when there is no light.
    DirectX::XMFLOAT3 sunToDir = { 0.0f, 0.0f, 0.0f };

    const auto& lights = m_Scene.GetLights();
    if (!lights.empty())
    {
        sunToDir = { -lights[0].direction.x,
                     -lights[0].direction.y,
                     -lights[0].direction.z };
        dominant = BakedGI::SunFoldWeights(sunToDir, w, &wMax);
    }
    else
    {
        // No light at all: no fold, so every weight is 0. SunFoldWeights zeroes
        // its array itself, but this path never calls it, so clear here too.
        for (uint32_t j = 0; j < BakedGI::kDirections; ++j)
            w[j] = 0.0f;
    }

    for (uint32_t j = 0; j < BakedGI::kDirections; ++j)
        m_FrameConstants.bakedGISunWeight[j].x = w[j]; // .x: the field is a float4 array

    // The REAL sun, for the Ray Fan's DeltaDirections channel (debug view 7).
    // That channel draws the 16 bin boxes and, from the same cell and at the
    // same length, this one orange box: the angle between them is the fold's
    // angular error, and it is now visible instead of readable only in the Sun
    // Fold Bins table.
    //
    // w = 1.0 only when the fold is active for this sun. The weight test is the
    // bin panel's own (BakedGI::BuildSunBinPanel sets haveSun from "a direction
    // exists and wMax > 0"), so the marker and the panel's "fold active" label
    // cannot disagree. A sun below the horizon zeroes every kernel weight, so
    // wMax is 0 there too and the marker does not draw.
    //
    // The length test guards the divide AND the NaN case: a NaN passes every
    // comparison as false, so it lands in the zero branch instead of reaching a
    // vertex position. When the marker does not draw, xyz is zeroed as well, so
    // no stale direction survives a removed or a set sun.
    const float sunLen = std::sqrt(sunToDir.x * sunToDir.x + sunToDir.y * sunToDir.y
                                   + sunToDir.z * sunToDir.z);
    if (!lights.empty() && (sunLen > 1e-6f) && (wMax > 0.0f))
        m_FrameConstants.bakedGISunDir = { sunToDir.x / sunLen, sunToDir.y / sunLen,
                                           sunToDir.z / sunLen, 1.0f };
    else
        m_FrameConstants.bakedGISunDir = { 0.0f, 0.0f, 0.0f, 0.0f };

    m_BakedGISunDir    = (int)dominant;
    m_BakedGISunWeight = wMax; // 0.0 when there is no sun
    // Method 3 has ONE direction and no argmax: the dump writes group 0, so the
    // fan must read group 0. Leaving the binned argmax here would point the fan at
    // bin 14, which carries no weight in that method's fold.
    m_FrameConstants.bakedGIRayDebugDir = SingleDirectionSun() ? 0u : dominant;
}

void Application::Render()
{
    // Baked GI probe bake — runs on a synced command list of
    // its own before the frame starts recording. Triggered by the scene-load
    // auto-bake or the "Rebuild GI Probes" button.
    if (m_PendingGIProbeBake)
    {
        m_PendingGIProbeBake = false;
        // Method 3 additionally bakes the sun's own response toward the REAL sun,
        // so the transport needs the to-sun direction at bake time. Everything
        // else about the bake is unchanged.
        const DirectX::XMFLOAT3 bakeSunToDir = SunToDirNow();
        m_Renderer.BakeGIProbes(&m_Model, m_BakedGISpacing, SingleDirectionSun(), bakeSunToDir,
                                SunBakeSeedDir(bakeSunToDir));
        // The fold is baked under method 3, so it must run once. It CANNOT run
        // here: the sky cubemap it samples is baked later in this same frame, so
        // folding now would use the previous frame's sky. Mark it pending and
        // service it right after ExecuteSky below.
        if (SingleDirectionSun())
            m_PendingProbeFold = true;
        // The bake replaced the per-term capture, so the SRV indices the "Efb
        // Contribution" view reads changed. Update published the PRE-bake ones,
        // and Render uploads the constants again after these pending blocks, so
        // refresh them here or that upload would carry the sentinel for one frame.
        FillEfbBallConstants();
    }

    // DDGI grid setup — same pattern: a synced command list of its own before
    // the frame starts recording. Triggered by the scene-load auto-fit or the
    // "Rebuild DDGI Grid" button.
    if (m_PendingDDGISetup)
    {
        m_PendingDDGISetup = false;
        m_Renderer.SetupDDGI(&m_Model, m_DDGISpacing);
    }

    // Ray-debug dump — same synced-list pattern, debug only. Runs after the
    // bake so it reads a complete table. Logs the direction ranking, the ray
    // ranking, the self-check, and the CPU kernel-weight mirror.
    if (m_PendingRayDebug)
    {
        m_PendingRayDebug = false;
        // The table is indexed by TO-SUN directions; the light stores the travel
        // direction. Same negation the update pass uses. Both the ray-debug
        // readout and the kernel-weight log use this one direction.
        const bool haveSunToDir = HasSunLight();
        const DirectX::XMFLOAT3 sunToDir = SunToDirNow();
        // The SAME seed function the bake used. A different value here traces
        // other rays and the self-check reports MISMATCH — a true report.
        m_Renderer.RayDebugBakedGI(&m_Model, (uint32_t)std::max(0, m_RayDebugProbeIndex),
                                   haveSunToDir ? &sunToDir : nullptr, SingleDirectionSun(),
                                   SunBakeSeedDir(sunToDir));
        if (haveSunToDir)
            BakedGI::LogKernelWeights(sunToDir);
        // The dump replaced the ray-debug buffer and cached a new verdict. Fill
        // the dump-derived constants again so the fan reads the new buffer, probe
        // and verdict on THIS frame. Without this the fan would read the pre-dump
        // SRV index, whose first value is 0 — a descriptor that is not the
        // ray-debug buffer.
        FillRayDebugConstants();
    }

    // Sun-bin panel read-back — the same synced-list pattern, and for the same
    // reason: the copy must finish before the Map, and resetting the command
    // list mid-frame would discard the frame in flight.
    if (m_PendingProbeTableReadback)
    {
        m_PendingProbeTableReadback = false;
        const uint32_t want = (uint32_t)std::max(0, m_RayDebugProbeIndex);
        m_Renderer.ReadbackBakedGIProbeTable(want); // no bake, no data; the panel says so
    }

    // Begin frame rendering
    m_Renderer.BeginFrame();

    MICROPROFILE_SCOPEI("Render", "FrameCpu", MP_RED);
    MICROPROFILE_SCOPEGPUI("FrameGpu", MP_RED);

    // Apply deferred resolution change (must happen after BeginFrame but before any rendering)
    if (m_PendingResolutionChange)
    {
        m_PendingResolutionChange = false;
        m_InternalWidth = m_PendingInternalWidth;
        m_InternalHeight = m_PendingInternalHeight;
        m_Renderer.CreateInternalResolutionResources(m_InternalWidth, m_InternalHeight);
        m_Renderer.CreateTaaResources(m_OutputWidth, m_OutputHeight, m_InternalWidth, m_InternalHeight);

        // Update frame constants with new internal resolution
        m_FrameConstants.screenWidth = m_InternalWidth;
        m_FrameConstants.screenHeight = m_InternalHeight;
        m_FrameConstants.internalWidth = m_InternalWidth;
        m_FrameConstants.internalHeight = m_InternalHeight;
        m_Renderer.UpdateFrameCB(m_FrameConstants);
    }

    auto cmdList = m_Renderer.GetCommandList();
    auto& gbuffer = m_Renderer.GetGBuffer();
    const bool usePathTracingFrame = m_UsePathTracer && m_Renderer.IsRayTracingSupported();

    // Reset viewport and scissor for main pass (internal resolution for rendering)
    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(m_InternalWidth), static_cast<float>(m_InternalHeight));
    D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, m_InternalWidth, m_InternalHeight);
    cmdList->RSSetViewports(1, &viewport);
    cmdList->RSSetScissorRects(1, &scissorRect);

    // Set camera viewProj to root param 0
    m_Renderer.UpdateFrameCB(m_FrameConstants);
    cmdList->SetGraphicsRootConstantBufferView(0, m_Renderer.GetFrameGPUAddress());

    // Sky system : bake the Hosek-Wilkie cubemap + project SH9.
    // Must run before any path tracer / GI / lighting pass that reads the sky.
    {
        MICROPROFILE_SCOPEI("Render", "Sky", MP_CYAN);
        MICROPROFILE_SCOPEGPUI("Sky", MP_CYAN);
        GPU_MARKER(cmdList, L"Sky Bake + SH9 Project");
        if (!m_Scene.GetLights().empty())
            m_Renderer.ExecuteSky(m_FrameConstants, m_Scene.GetLights()[0]);

        // Replace directional-light colour/intensity with the Hosek-Wilkie
        // sun irradiance computed by Sky::ComputeSunIrradiance().
        if (!m_Scene.GetLights().empty() && m_Scene.GetLights()[0].direction.w < 0.5f)
        {
            LightConstants& sun = m_Scene.GetLights()[0];
            DirectX::XMFLOAT3 solarIrradiance = m_Renderer.GetSunIrradiance();
            sun.color     = { solarIrradiance.x, solarIrradiance.y, solarIrradiance.z, 1.0f };
            sun.intensity = 1.0f;
            m_Renderer.UpdateLightsBuffer(m_Scene.GetLights());
        }

        // Baked GI — fold the live sun/sky state into the lit probe grid
        // (probe mode only; ReSTIR GI is the exclusive alternative and its
        // dispatch early-outs via enableRestirGI = 0).
        // Under method 3 the fold is BAKED: it runs once per bake, not per frame.
        // It is serviced here, after ExecuteSky and after the sun-irradiance
        // replacement above, so sunE and the cubemap are both current. Running it
        // inside BakeGIProbes instead would read the previous frame's sky.
        const bool foldWanted = SingleDirectionSun() ? m_PendingProbeFold : true;
        if (m_FrameConstants.bakedGIMode != 0 && m_FrameConstants.bakedGIValid != 0
            && !m_Scene.GetLights().empty() && foldWanted)
        {
            GPU_MARKER(cmdList, L"BakedGI Update");
            m_Renderer.DispatchBakedGIUpdate(m_FrameConstants, m_Scene.GetLights()[0],
                                             SingleDirectionSun());
            m_PendingProbeFold = false;
            // ONE line, once per bake, never per frame. This is the observation
            // that the method-3 update pass is idle: if this prints again on the
            // next frame, the gate is broken and the 17.95 -> 0 GB/s claim is
            // wrong. Gated on SingleDirectionSun() so the binned method, which
            // folds every frame by design, does not spam the log.
            if (SingleDirectionSun())
                std::cout << "[BakedGI] fold baked once for "
                          << m_Renderer.GetBakedGI().GetProbeCount()
                          << " probes (single-direction sun); the update pass is idle"
                          << std::endl;
        }

        // DDGI — trace the probe rays and resolve the two atlases. It runs after
        // the sky bake (a miss samples the cubemap) and the sun-colour replacement
        // above, so the probe rays see the live sun. The deferred pass consumes
        // the atlas later in the frame.
        if (m_FrameConstants.ddgiMode != 0 && m_FrameConstants.ddgiValid != 0
            && !m_Scene.GetLights().empty())
        {
            GPU_MARKER(cmdList, L"DDGI Update");
            m_Renderer.RecordDDGIFrame(&m_Model, m_FrameConstants, m_Scene.GetLights()[0]);
        }
    }

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

        {
            MICROPROFILE_SCOPEI("Render", "PathTrace", MP_BLUE);
            MICROPROFILE_SCOPEGPUI("PathTrace", MP_BLUE);
            GPU_MARKER(cmdList, L"Path Trace");
            m_Renderer.DispatchRays(&m_Model, m_FrameConstants, m_Scene.GetLights()[0]);
        }

        if (m_AntiAliasingMode == AA_MODE_TAA && m_Renderer.IsTaaEnabled())
        {
            // Generate motion vectors from depth + viewProj matrices
            {
                MICROPROFILE_SCOPEI("Render", "MotionVectors", MP_GREEN);
                MICROPROFILE_SCOPEGPUI("MotionVectors", MP_GREEN);
                GPU_MARKER(cmdList, L"Motion Vectors");
                m_Renderer.GenerateMotionVectors(m_FrameConstants);
            }
            // Run TAA on the HDR path tracer output, then copy TAA output to back buffer
            {
                MICROPROFILE_SCOPEI("Render", "TAA", MP_YELLOW);
                MICROPROFILE_SCOPEGPUI("TAA", MP_YELLOW);
                GPU_MARKER(cmdList, L"TAA");
                m_Renderer.DispatchNaiveTsr(m_FrameConstants, m_Renderer.GetPathTracerHdrOutput());
            }
            {
                MICROPROFILE_SCOPEI("Render", "CopyToBackBuffer", MP_WHITE);
                GPU_MARKER(cmdList, L"Copy To BackBuffer");
                m_Renderer.CopyTextureToBackBuffer(m_Renderer.GetTaaOutputTex());
            }
        }
        else
        {
            {
                MICROPROFILE_SCOPEI("Render", "PTPresent", MP_WHITE);
                MICROPROFILE_SCOPEGPUI("PTPresent", MP_WHITE);
                GPU_MARKER(cmdList, L"Path Trace Present");
                m_Renderer.CopyTextureToBackBuffer(m_Renderer.GetPathTracerOutput());
            }
        }

        // Draw path visualization lines on top of the PT output (non-RTXDI ReSTIR only)
        if (m_PathVizEverCaptured && m_FrameConstants.enableRestir && !m_FrameConstants.useRTXDI)
        {
            MICROPROFILE_SCOPEI("Render", "PathViz", MP_ORANGE);
            MICROPROFILE_SCOPEGPUI("PathViz", MP_ORANGE);
            GPU_MARKER(cmdList, L"Path Visualization");
            m_Renderer.DrawPathVizLines(m_FrameConstants);
        }

        // Setup viewport and RTV for ImGui rendering on top of PT output
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_Renderer.GetCurrentBackBufferRTV();
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
    }
    else
    {
        // Helper: rasterize the meshlets.
        // m_UseVisibilityBuffer=true  -> visibility-only pass (writes R32_UINT token +
        //     depth, 1 RTV); GBuffer is reconstructed afterward by the "Visibility
        //     GBuffer Resolve" full-screen pass below.
        // m_UseVisibilityBuffer=false -> direct-to-GBuffer pass (MeshletRasterizeGBufferMS.hlsl,
        //     4 RTVs: albedo/normal/material/visToken) — classic per-fragment shading,
        //     no resolve pass needed.
        // clearTargets=true for Phase 1 / frustum-only (clear RTVs + depth),
        // clearTargets=false for Phase 2 — preserve depth/RTVs so Phase 2's
        // newly-revealed meshlets composite on top of Phase 1's output.
        // clearTargets doubles as "is this Phase 1" — Phase 1 clears (and is TWO_PASS_PHASE_FIRST),
        // Phase 2 preserves (and is TWO_PASS_PHASE_SECOND). The phase is threaded through to the
        // mesh shader so it can offset its VisibleMeshlets[] read by Phase 1's final count —
        auto doMeshletRasterize = [&](bool clearTargets)
        {
            GraphicsHelper::TransitionResource(cmdList, gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetVisibilityBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET);
            D3D12_CPU_DESCRIPTOR_HANDLE dsv = gbuffer.depth.dsvHandle;

            if (m_UseVisibilityBuffer)
            {
                D3D12_CPU_DESCRIPTOR_HANDLE rtvs[1] = {
                    m_Renderer.GetVisibilityBuffer().rtvHandle
                };
                cmdList->OMSetRenderTargets(1, rtvs, FALSE, &dsv);

                if (clearTargets)
                {
                    const float clearVis[] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    cmdList->ClearRenderTargetView(rtvs[0], clearVis, 0, nullptr);
                    cmdList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);
                }
            }
            else
            {
                GraphicsHelper::TransitionResource(cmdList, gbuffer.albedo, D3D12_RESOURCE_STATE_RENDER_TARGET);
                GraphicsHelper::TransitionResource(cmdList, gbuffer.normal, D3D12_RESOURCE_STATE_RENDER_TARGET);
                GraphicsHelper::TransitionResource(cmdList, gbuffer.material, D3D12_RESOURCE_STATE_RENDER_TARGET);

                D3D12_CPU_DESCRIPTOR_HANDLE rtvs[4] = {
                    gbuffer.albedo.rtvHandle,
                    gbuffer.normal.rtvHandle,
                    gbuffer.material.rtvHandle,
                    m_Renderer.GetVisibilityBuffer().rtvHandle
                };
                cmdList->OMSetRenderTargets(4, rtvs, FALSE, &dsv);

                if (clearTargets)
                {
                    const float clearCol[] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    const float clearVis[] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    cmdList->ClearRenderTargetView(rtvs[0], clearCol, 0, nullptr);
                    cmdList->ClearRenderTargetView(rtvs[1], clearCol, 0, nullptr);
                    cmdList->ClearRenderTargetView(rtvs[2], clearCol, 0, nullptr);
                    cmdList->ClearRenderTargetView(rtvs[3], clearVis, 0, nullptr);
                    cmdList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);
                }
            }

            uint32_t phase = clearTargets ? TWO_PASS_PHASE_FIRST : TWO_PASS_PHASE_SECOND;
            m_Renderer.DispatchMeshletRasterize(&m_Model, m_UseVisibilityBuffer, phase);
        };

        // Helper: full-screen Visibility Buffer resolve — reconstructs GBuffer
        // (albedo/normal/material) from the visibility token written by the
        // rasterize pass(es) above. Discards sky/background pixels, so the GBuffer
        // targets must be cleared to zero first.
        auto doVisibilityGBufferResolve = [&]()
        {
            MICROPROFILE_SCOPEI("Render", "VisibilityGBuffer", MP_BLUE);
            MICROPROFILE_SCOPEGPUI("VisibilityGBuffer", MP_BLUE);
            GPU_MARKER(cmdList, L"Visibility GBuffer Resolve");

            GraphicsHelper::TransitionResource(cmdList, gbuffer.albedo, D3D12_RESOURCE_STATE_RENDER_TARGET);
            GraphicsHelper::TransitionResource(cmdList, gbuffer.normal, D3D12_RESOURCE_STATE_RENDER_TARGET);
            GraphicsHelper::TransitionResource(cmdList, gbuffer.material, D3D12_RESOURCE_STATE_RENDER_TARGET);
            GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetVisibilityBuffer(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

            const float clearCol[] = { 0.0f, 0.0f, 0.0f, 0.0f };
            cmdList->ClearRenderTargetView(gbuffer.albedo.rtvHandle, clearCol, 0, nullptr);
            cmdList->ClearRenderTargetView(gbuffer.normal.rtvHandle, clearCol, 0, nullptr);
            cmdList->ClearRenderTargetView(gbuffer.material.rtvHandle, clearCol, 0, nullptr);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[3] = {
                gbuffer.albedo.rtvHandle,
                gbuffer.normal.rtvHandle,
                gbuffer.material.rtvHandle
            };
            cmdList->OMSetRenderTargets(_countof(rtvs), rtvs, FALSE, nullptr);

            m_Renderer.DispatchVisibilityGBufferResolve(&m_Model);
        };

        // Debug overlay for meshlet visibility buffer — runs once per frame, after
        // the final rasterize pass (Phase 2 or frustum-only).
        auto runDebugOverlay = [&]() {
            if (m_Renderer.GetMeshletDebugMode() > 0)
            {
                MICROPROFILE_SCOPEGPUI("MeshletDebug", MP_PURPLE);
                GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetVisibilityBuffer(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                // Mip-tint sideband : cull writes it as UAV, overlay reads as SRV
                GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetVisibleMeshletMipsBuffer(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetFullScreenDebugTex(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cmdList->SetComputeRootSignature(m_Renderer.GetRootSignature());
                cmdList->SetPipelineState(m_Renderer.GetMeshletDebugViewPSO());
                cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
                cmdList->SetComputeRootConstantBufferView(0, m_Renderer.GetFrameGPUAddress());
                // MeshletDebugParams carries the bindless SRV indices for the meshlet stream
                // buffers — no root descriptor table needed, so compute
                // and graphics root state can no longer disagree about space3.
                MeshletDebugParams debugParams = {};
                debugParams.Mode             = (uint32_t)m_Renderer.GetMeshletDebugMode();
                debugParams.VisBufSRVIdx     = m_Renderer.GetVisibilityBuffer().srvIndex;
                debugParams.CandidatesSRVIdx = (uint32_t)m_Renderer.GetVisibleMeshletsSRVIndex();
                debugParams.OutputUAVIdx     = m_Renderer.GetFullScreenDebugTex().uavIndex;
                debugParams.Width            = m_InternalWidth;
                debugParams.Height           = m_InternalHeight;
                debugParams.MipsSRVIdx       = (uint32_t)m_Renderer.GetVisibleMeshletMipsSRVIndex();
                debugParams.HZBMipCount      = m_Renderer.GetHZBMips();
                debugParams.GlobalPositionsSRVIdx        = (uint32_t)m_Model.GetGlobalPositionsSRVIndex();
                debugParams.GlobalNormalsSRVIdx          = (uint32_t)m_Model.GetGlobalNormalsSRVIndex();
                debugParams.GlobalUVsSRVIdx              = (uint32_t)m_Model.GetGlobalUVsSRVIndex();
                debugParams.GlobalMeshletsSRVIdx         = (uint32_t)m_Model.GetGlobalMeshletsSRVIndex();
                debugParams.GlobalMeshletVerticesSRVIdx  = (uint32_t)m_Model.GetGlobalMeshletVerticesSRVIndex();
                debugParams.GlobalMeshletTrianglesSRVIdx = (uint32_t)m_Model.GetGlobalMeshletTrianglesSRVIndex();
                debugParams.GlobalMeshletBoundsSRVIdx    = (uint32_t)m_Model.GetGlobalMeshletBoundsSRVIndex();
                debugParams.MeshDataSRVIdx               = (uint32_t)m_Model.GetMeshDataSRVIndex();
                debugParams.InstanceDataSRVIdx           = (uint32_t)m_Model.GetInstanceDataSRVIndex();
                cmdList->SetComputeRoot32BitConstants(12, sizeof(debugParams) / 4, &debugParams, 0);
                cmdList->Dispatch(
                    (m_InternalWidth  + 7) / 8,
                    (m_InternalHeight + 7) / 8, 1);
                D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(
                    m_Renderer.GetFullScreenDebugTex().resource.Get());
                cmdList->ResourceBarrier(1, &uavBarrier);
            }
        };

        if (m_UseMeshlet && m_Model.IsMeshletReady())
        {
            // =====================================================
            // SINGLE-PASS OCCLUSION CULLING — one CullInstances →
            // CullMeshlets pass (occlusion ON, Phase 0 only).
            // Reads the previous-frame HZB and writes Binning-visible
            // meshlets directly. Skips the Phase 2 retest, so instances
            // hidden behind this frame's Phase-1 geometry remain culled
            // until next frame (cheaper, slightly fewer draw calls).
            // =====================================================

            // --- Phase 1: cull vs previous-frame HZB (all instances) ---
            {
                MICROPROFILE_SCOPEI("Render", "Phase1_CullInstances", MP_GREEN);
                MICROPROFILE_SCOPEGPUI("Phase1_CullInstances", MP_GREEN);
                GPU_MARKER(cmdList, L"Phase 1 - CullInstances (vs prev HZB)");
                m_Renderer.DispatchMeshletTwoPassCull(&m_Model, m_FrameConstants, m_EnableOcclusionCulling, 0, m_FreezeCulling);
            }
            {
                MICROPROFILE_SCOPEI("Render", "Phase1_BuildDispatchArgs", MP_YELLOW);
                MICROPROFILE_SCOPEGPUI("Phase1_BuildDispatchArgs", MP_YELLOW);
                GPU_MARKER(cmdList, L"Phase 1 - BuildDispatchMeshArgs");
                m_Renderer.DispatchMeshletBuildDispatchArgs(TWO_PASS_PHASE_FIRST);
            }
            {
                MICROPROFILE_SCOPEI("Render", "Phase1_Rasterize", MP_BLUE);
                MICROPROFILE_SCOPEGPUI("Phase1_Rasterize", MP_BLUE);
                GPU_MARKER(cmdList, L"Phase 1 - Rasterize (Clear)");
                doMeshletRasterize(true); // Clear depth
            }
            {
                MICROPROFILE_SCOPEI("Render", "Phase1_BuildHZB", MP_ORANGE);
                MICROPROFILE_SCOPEGPUI("Phase1_BuildHZB", MP_ORANGE);
                GPU_MARKER(cmdList, L"Phase 1 - BuildHZB (fresh, for Phase 2)");
                // Keep the HZB consistent with the frozen cull view (mirrors D3D12_Research)
                if (!m_FreezeCulling)
                    m_Renderer.DispatchBuildHZB();
            }

            if (m_EnableTwoPassCulling && m_EnableOcclusionCulling)
            {
                // =====================================================
                // TWO-PHASE OCCLUSION CULLING
                //
                // Phase 1 tests ALL instances against the previous frame's
                // HZB and rasterizes the survivors. That partial depth is
                // then used to rebuild a FRESH HZB — this is the entire
                // point of two-pass occlusion culling: Phase 2 gets to
                // retest previously-occluded instances against depth that
                // reflects what Phase 1 already drew THIS frame, revealing
                // instances that are only hidden behind Phase-1 geometry
                // (not stale last-frame geometry).
                //
                // Phase 2 rasterizes its own survivors on top (preserving
                // depth/color from Phase 1), then a final HZB rebuild
                // captures the complete depth for next frame's Phase 1.
                //
                // Gated on m_EnableOcclusionCulling too (matches D3D12_Research's
                // MeshletRasterizer::Render): with occlusion off, CullInstancesCS never
                // defers anything to OccludedInstances (frustum-only has no occlusion
                // verdict to defer from), so Phase 2 would just be a wasted dispatch —
                // skip it explicitly rather than relying on an incidentally-empty list.
                // =====================================================

                // --- Phase 2: retest occluded instances vs fresh HZB ---
                {
                    MICROPROFILE_SCOPEI("Render", "Phase2_CullInstances", MP_GREEN);
                    MICROPROFILE_SCOPEGPUI("Phase2_CullInstances", MP_GREEN);
                    GPU_MARKER(cmdList, L"Phase 2 - CullInstances (vs fresh HZB)");
                    m_Renderer.DispatchMeshletTwoPassCull(&m_Model, m_FrameConstants, m_EnableOcclusionCulling, 1, m_FreezeCulling);
                }
                {
                    MICROPROFILE_SCOPEI("Render", "Phase2_BuildDispatchArgs", MP_YELLOW);
                    MICROPROFILE_SCOPEGPUI("Phase2_BuildDispatchArgs", MP_YELLOW);
                    GPU_MARKER(cmdList, L"Phase 2 - BuildDispatchMeshArgs");
                    m_Renderer.DispatchMeshletBuildDispatchArgs(TWO_PASS_PHASE_SECOND);
                }
                {
                    MICROPROFILE_SCOPEI("Render", "Phase2_Rasterize", MP_BLUE);
                    MICROPROFILE_SCOPEGPUI("Phase2_Rasterize", MP_BLUE);
                    GPU_MARKER(cmdList, L"Phase 2 - Rasterize (Preserve depth)");
                    doMeshletRasterize(false); // Preserve depth/color from Phase 1
                }
                {
                    MICROPROFILE_SCOPEI("Render", "Phase2_BuildHZB", MP_ORANGE);
                    MICROPROFILE_SCOPEGPUI("Phase2_BuildHZB", MP_ORANGE);
                    GPU_MARKER(cmdList, L"Phase 2 - BuildHZB (complete, for next frame)");
                    if (!m_FreezeCulling)
                        m_Renderer.DispatchBuildHZB();
                }
            }

            // Visibility Buffer resolve — runs once after the final rasterize pass
            // (Phase 2 if two-pass culling is on, otherwise Phase 1/frustum-only),
            // reconstructing GBuffer (albedo/normal/material) from the visibility token.
            // Skipped when m_UseVisibilityBuffer is off — MeshletRasterizeGBufferMS.hlsl
            // already wrote the GBuffer directly during rasterize.
            if (m_UseVisibilityBuffer)
                doVisibilityGBufferResolve();

            // Debug overlay — runs once after final rasterize
            runDebugOverlay();
        }
        else
        {
            m_Renderer.ExecuteGBufferPass(&m_Model, frustum, m_EnableDepthPrePass);
        } // closes non-meshlet else branch (DepthPrePass + GBuffer)
        // The meshlet path (Cull + Binning + MeshShader) also writes albedo/normal/material/depth
        // into the same GBuffer targets, so all passes below run identically for both paths.

        // -----------------------------------------------------------------------
        // UNION PATH — runs after both meshlet and non-meshlet geometry branches.
        // GBuffer (albedo, normal, material, depth) is now populated by either path.
        // HZB build is handled inside each branch above (two-phase calls it twice,
        // traditional/frustum-only calls it once) — not duplicated here.
        // -----------------------------------------------------------------------

        // ReSTIR DI — direct illumination from local lights, reads GBuffer depth + material
        {
            MICROPROFILE_SCOPEI("Render", "ReSTIR_DI", MP_RED);
            MICROPROFILE_SCOPEGPUI("ReSTIR_DI", MP_RED);
            GPU_MARKER(cmdList, L"ReSTIR DI");
            m_Renderer.DispatchRestirDI(&m_Model, m_FrameConstants);
        }

        // ReSTIR GI — indirect illumination, reads GBuffer albedo + normal + depth
        {
            MICROPROFILE_SCOPEI("Render", "ReSTIR_GI", MP_PURPLE);
            MICROPROFILE_SCOPEGPUI("ReSTIR_GI", MP_PURPLE);
            GPU_MARKER(cmdList, L"ReSTIR GI");
            m_Renderer.DispatchRestirGI(&m_Model, m_FrameConstants);
        }

        // HZB mip viewer  — visualize the selected HZB mip as grayscale.
        // Runs after all HZB builds, so it shows this frame's freshest chain; the
        // FullScreenDebug pass then blits FullScreenDebugTex in place of the lighting pass.
        if (m_UseMeshlet && m_HZBDebugMip >= 0)
        {
            GPU_MARKER(cmdList, L"HZB Mip Viewer");
            GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetFullScreenDebugTex(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_Renderer.DispatchHZBDebugView(m_Renderer.GetFullScreenDebugTex().uavIndex,
                                            m_InternalWidth, m_InternalHeight, m_HZBDebugMip);
            D3D12_RESOURCE_BARRIER hzbViewBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_Renderer.GetFullScreenDebugTex().resource.Get());
            cmdList->ResourceBarrier(1, &hzbViewBarrier);
            GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetFullScreenDebugTex(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        // Occluded-rect overlay — draw NDC rects of HZB-rejected
        // instances/meshlets over a dimmed scene-albedo background.
        if (m_UseMeshlet && m_OccludedRectDebug)
        {
            GPU_MARKER(cmdList, L"Occluded Rects Debug");
            GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetFullScreenDebugTex(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_Renderer.DispatchOccludedRectsDebug(m_Renderer.GetFrameGPUAddress(), m_Renderer.GetFullScreenDebugTex(),
                                                  m_InternalWidth, m_InternalHeight);
            GraphicsHelper::TransitionResource(cmdList, m_Renderer.GetFullScreenDebugTex(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        // Depth-duel labels — producer for the on-screen debug text
        // pass that runs at end of frame. Emits into the shared text buffer.
        if (m_UseMeshlet && m_DebugScreenText && m_DebugDepthLabels)
        {
            GPU_MARKER(cmdList, L"Depth Readout");
            m_Renderer.DispatchDepthReadout(m_OutputWidth, m_OutputHeight);
        }

        // Cull stats overlay — per-phase culling counters on-screen table
        if (m_UseMeshlet && m_DebugScreenText && m_CullStatsOverlay)
        {
            GPU_MARKER(cmdList, L"Cull Stats");
            m_Renderer.DispatchEmitCullStats(
                static_cast<uint32_t>(m_Model.GetInstanceCount()),
                static_cast<uint32_t>(m_Model.GetTotalMeshletCount()),
                m_OutputWidth, m_OutputHeight);
        }

        const bool rasterTaaActive = (m_AntiAliasingMode == AA_MODE_TAA) && m_Renderer.IsTaaEnabled() && !m_DebugShadowMap;

        // Determine if any full-screen debug mode is active.
        // When true, FullScreenDebug.hlsl replaces Lighting.hlsl entirely.
        const bool debugActive =
            (m_Renderer.GetMeshletDebugMode() > 0) ||
            (m_UseMeshlet && m_HZBDebugMip >= 0) ||
            (m_UseMeshlet && m_OccludedRectDebug) ||
            (m_FrameConstants.sharcDebug != 0) ||
            (m_FrameConstants.restirReservoirDebugMode != RESTIR_RESERVOIR_DEBUG_OFF) ||
            (m_FrameConstants.restirDIDebugMode != RESTIR_DI_DEBUG_OFF);

        // Lighting Pass (or FullScreenDebug Pass when debug is active)
        // Reads GBuffer SRVs + FinalDiffuse/FinalSpecular from ReSTIR passes above.
        // When TAA is active: renders to internal-res RasterHdrOutputTex (no tonemapping).
        // When TAA is off:    renders directly to output-res back buffer (with tonemapping).
        m_Renderer.ExecuteLightingPass(&m_Model, m_FrameConstants, rasterTaaActive, debugActive,
                                        m_DebugShadowMap, m_OutputWidth, m_OutputHeight);

        // Transparency Pass (Forward)
        // Runs BEFORE TAA so that transparent geometry is temporally accumulated.
        // In the TAA path: renders into RasterHdrOutputTex at internal resolution,
        //   depth-tested against the G-buffer depth (also at internal resolution).
        // In the non-TAA path: renders directly to the back buffer at output resolution.
        m_Renderer.ExecuteTransparencyPass(&m_Model, frustum, rasterTaaActive, m_OutputWidth, m_OutputHeight);

        // TAA post-processing (shared by both meshlet and non-meshlet paths)
        if (rasterTaaActive)
        {
            // Generate motion vectors from depth + viewProj matrices
            {
                MICROPROFILE_SCOPEI("Render", "MotionVectors", MP_GREEN);
                MICROPROFILE_SCOPEGPUI("MotionVectors", MP_GREEN);
                GPU_MARKER(cmdList, L"Motion Vectors");
                m_Renderer.GenerateMotionVectors(m_FrameConstants);
            }
            // Run TAA on the HDR output (which now includes transparency),
            // then copy TAA output to back buffer
            {
                MICROPROFILE_SCOPEI("Render", "TAA", MP_YELLOW);
                MICROPROFILE_SCOPEGPUI("TAA", MP_YELLOW);
                GPU_MARKER(cmdList, L"TAA");
                m_Renderer.DispatchNaiveTsr(m_FrameConstants, m_Renderer.GetRasterHdrOutputTex());
            }
            {
                MICROPROFILE_SCOPEI("Render", "CopyToBackBuffer", MP_WHITE);
                GPU_MARKER(cmdList, L"Copy To BackBuffer");
                m_Renderer.CopyTextureToBackBuffer(m_Renderer.GetTaaOutputTex());
            }
        }
        else
        {
            // Non-TAA: copy HDR output directly to back buffer.
            MICROPROFILE_SCOPEI("Render", "CopyToBackBuffer", MP_WHITE);
            GPU_MARKER(cmdList, L"Copy To BackBuffer");
            m_Renderer.CopyTextureToBackBuffer(m_Renderer.GetRasterHdrOutputTex());
        }
    } // closes raster (non-path-tracer) branch


    // Prepare back buffer for ImGui rendering at full output resolution.
    // Rebind the back-buffer RTV *without* a DSV so that the stale G-buffer
    // depth (which is at internal resolution) does not clip ImGui draws.
    {
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_Renderer.GetCurrentBackBufferRTV();
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

        D3D12_VIEWPORT outputViewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(m_OutputWidth), static_cast<float>(m_OutputHeight));
        D3D12_RECT outputScissor = CD3DX12_RECT(0, 0, m_OutputWidth, m_OutputHeight);
        cmdList->RSSetViewports(1, &outputViewport);
        cmdList->RSSetScissorRects(1, &outputScissor);
    }

    // Baked GI probe debug overlay — post-composite LDR (after TAA / tonemap,
    // before on-screen text): display-referred colors, no taaEnabled branching,
    // manual reverse-Z occlusion against the GBuffer depth. Includes the
    // fullscreen Backface Leak Map (drawn first, under the probe views).
    if (!usePathTracingFrame && m_FrameConstants.bakedGIMode != 0 &&
        m_FrameConstants.bakedGIValid != 0 &&
        (m_BakedGIDebugView != 0 || m_FrameConstants.bakedGILeakDebug != 0))
        m_Renderer.DrawBakedGIProbeDebug(m_FrameConstants, m_OutputWidth, m_OutputHeight);

    // DDGI debug overlay — same post-composite slot as the probe overlay above.
    if (!usePathTracingFrame && m_FrameConstants.ddgiMode != 0 &&
        m_FrameConstants.ddgiValid != 0 && m_DDGIDebugView != 0)
        m_Renderer.DrawDDGIDebug(m_FrameConstants, m_OutputWidth, m_OutputHeight);

    // GPU on-screen debug text/lines — draw on top of the final image, under ImGui
    if (m_DebugScreenText)
    {
        GPU_MARKER(cmdList, L"Debug Text Overlay");
        m_Renderer.RenderDebugTextOverlay(m_Renderer.GetFrameGPUAddress(),
                                          m_Renderer.GetCurrentBackBufferRTV(),
                                          m_OutputWidth, m_OutputHeight);
    }

    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplSDL2_NewFrame();

    ImGui::NewFrame();

    // Render ImGui UI
    {
        MICROPROFILE_SCOPEI("UI", "ImGui", MP_YELLOW);
        RenderImGui();
    }

    // Render ImGui draw data
    ImGui::Render();
    ID3D12DescriptorHeap* descriptorHeaps[] = { m_ImGuiDescriptorHeap.Get() };
    m_Renderer.GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_Renderer.GetCommandList());

    // End frame rendering (closes cmdList, executes on m_CommandQueue, presents, waits for GPU)
    m_Renderer.EndFrame();

    // MicroProfileFlip submits its own internal ResolveQueryData command list to the
    // same m_CommandQueue. It must run AFTER EndFrame() has already submitted (and the
    // GPU has finished) the main command list that contains all the EndQuery calls.
    // Passing nullptr because MicroProfile D3D12 uses its own internal command list.
    MicroProfileFlip(nullptr);
}

// Method 3 traces ONE direction, so the help text of the views below describes a
// fold it does not run. Printed as a line above the widget rather than folded into
// the tooltip: the tooltip literals stay untouched, so Probe (Baked) output is
// byte-identical by construction, not by careful editing.
static const char* kMethod3WordingCaveat =
    "Method 3 traces ONE direction -- the real sun. The help text below describes "
    "Probe (Baked)'s 16-direction fold and does not apply to this method.";

// The to-sun direction, negated from the light's travel direction exactly as the
// update pass does. One definition for every reader: the bake, the fold, the
// ray-debug readout, the kernel-weight log and the bin panel all used to negate
// it inline, and the mode-swap check below compares directions, so a divergence
// between two copies would silently disable that check.
DirectX::XMFLOAT3 Application::SunToDirNow()
{
    if (!HasSunLight())
        return DirectX::XMFLOAT3(0.0f, 1.0f, 0.0f);
    const LightConstants& sun = m_Scene.GetLights()[0];
    return DirectX::XMFLOAT3(-sun.direction.x, -sun.direction.y, -sun.direction.z);
}

// The seed index for the single-sun bake and for its ray-debug replay. One
// function for both, because the replay reproduces the bake's RNG stream and its
// self-check is the only proof that the two received the same value.
//
// The bin is DERIVED from the live sun, never written as a literal. It is 14 for
// the captured sun and it stays correct when the sun moves.
uint32_t Application::SunBakeSeedDir(const DirectX::XMFLOAT3& sunToDir) const
{
    if (!SingleDirectionSun() || !m_GISunSeedPairDominantBin)
        return BakedGI::kSunSeedOwnDir;
    return BakedGI::DominantSunDirection(sunToDir);
}

void Application::RenderImGui()
{
    // Create a simple debug window
    ImGui::Begin("Renderer Debug");

    // RGB Color Picker for background
    ImGui::ColorEdit3("Background Color", m_Renderer.m_BackgroundColor);

    ImGui::Checkbox("Enable Depth Pre-Pass", &m_EnableDepthPrePass);

    ImGui::Checkbox("Debug Shadow Map", &m_DebugShadowMap);

    ImGui::Separator();
    ImGui::Checkbox("Use Meshlet Rendering", &m_UseMeshlet);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Forward render using meshlet pipeline (GPU frustum culling + vertex pulling).\nOff = standard vertex-buffer deferred path.");

    if (m_UseMeshlet)
    {
        ImGui::Checkbox("Use Visibility Buffer", &m_UseVisibilityBuffer);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("On = MeshletRasterizeMS writes only a visibility token; GBuffer is\nreconstructed by the VisibilityGBuffer full-screen resolve pass.\nOff = MeshletRasterizeGBufferMS shades GBuffer directly per-fragment\n(classic path, no resolve pass) — useful for A/B comparison.");

        ImGui::Checkbox("Two-Pass Occlusion (Phase 1+2)", &m_EnableTwoPassCulling);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Two-phase HZB occlusion: Phase 1 vs prev-frame HZB → rasterize → rebuild HZB → Phase 2 retest.\nOff = single-pass (Phase 1 only, cheaper, fewer draw calls).");

        ImGui::Checkbox("Occlusion Culling (HZB)", &m_EnableOcclusionCulling);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Use HZB for occlusion culling.");

        ImGui::Checkbox("Freeze Culling", &m_FreezeCulling);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Freeze the culling frustum/HZB at the current view while the render camera keeps moving.\nMeshlets culled by the frozen view pop in/out as you fly away — verifies instance/meshlet culling works.");

        float moveSpeed = m_Camera.GetMoveSpeed();
        if (ImGui::SliderFloat("Camera Speed", &moveSpeed, 0.5f, 50.0f, "%.1f"))
            m_Camera.SetMoveSpeed(moveSpeed);

        ImGui::Separator();
        ImGui::Checkbox("FPS Limiter", &m_FPSLimitEnabled);
        ImGui::SameLine();
        ImGui::SliderFloat("##TargetFPS", &m_TargetFPS, 10.0f, 240.0f, "%.0f FPS");
    }

    if (m_UseMeshlet && m_Model.IsMeshletReady())
    {
        const char* debugModes[] = { "Off", "InstanceID", "MeshletID", "PrimitiveID", "HZB Mip Tint" };
        int debugMode = m_Renderer.GetMeshletDebugMode();
        if (ImGui::Combo("Meshlet Debug View", &debugMode, debugModes, IM_ARRAYSIZE(debugModes)))
        {
            m_Renderer.SetMeshletDebugMode(debugMode);
            // Mip-tint sideband — the cull shader writes one mip per visible
            // meshlet slot; the flag must be forwarded to GPUCulling before dispatch.
            m_Renderer.SetDebugRecordMip(debugMode == 4);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Overlay meshlet debug visualization.\nInstanceID = random color per instance\nMeshletID = random color per meshlet\nPrimitiveID = random color per triangle (wireframe)\nHZB Mip Tint = color by the HZB mip each meshlet's occlusion test used\n  (blue = fine mip / small object, red = coarse mip / large object,\n   gray = no occlusion test ran — enable Occlusion Culling for this mode)");

        int hzbMipMax = (int)m_Renderer.GetHZBMips() - 1;
        ImGui::SliderInt("HZB Mip Viewer", &m_HZBDebugMip, -1, hzbMipMax, m_HZBDebugMip < 0 ? "Off" : "Mip %d");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Visualize one HZB mip as grayscale (reverse-Z: white = near, black = far/sky).\nDisplayed via the FullScreenDebug pass (replaces lighting).\nOverrides the Meshlet Debug View overlay when both are active.");

        ImGui::Checkbox("Occluded Rects Overlay", &m_OccludedRectDebug);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Draws the screen-space rects of instances/meshlets rejected by HZBCull,\nover a dimmed scene background (via the FullScreenDebug pass).\nMeshlet: red = occluded phase 1, orange = phase 2.\nInstance: purple = phase 1, cyan = phase 2.");

        ImGui::Checkbox("On-Screen Debug Text", &m_DebugScreenText);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Rasterize GPU-recorded debug text/lines onto the final image (under ImGui).\nMaster switch — pair it with the options below that emit text.");
        ImGui::Checkbox("Occluded Depth Labels", &m_DebugDepthLabels);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Print the HZB depth duel (nearestDepth<hzbDepth m<mip>) above each\noccluded instance/meshlet, directly on the scene.\nRequires On-Screen Debug Text + Occlusion Culling (Phase 1+2).");
        ImGui::Checkbox("Cull Stats Overlay", &m_CullStatsOverlay);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("On-screen table: total instance/meshlet counts, per-phase occluded\ninstance/meshlet counts, and derived visible totals.\nRequires On-Screen Debug Text + Occlusion Culling (Phase 1+2).");
    }
    else if (m_Renderer.GetMeshletDebugMode() != 0 || m_HZBDebugMip >= 0 || m_OccludedRectDebug
             || m_DebugScreenText || m_DebugDepthLabels || m_CullStatsOverlay)
    {
        m_Renderer.SetMeshletDebugMode(0); // Reset debug when meshlet is disabled
        m_HZBDebugMip = -1;
        m_OccludedRectDebug = false;
        m_DebugScreenText = false;
        m_DebugDepthLabels = false;
        m_CullStatsOverlay = false;
    }

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

        if (selectedLight.direction.w < 0.5f) // Directional
        {
            // Directional-light colour/intensity are now driven by the Hosek-Wilkie
            // solar irradiance (Sky::ComputeSunIrradiance).  The values shown here
            // are read-only and updated each frame.
            DirectX::XMFLOAT3 si = m_Renderer.GetSunIrradiance();
            float irradianceRGB[3] = { si.x, si.y, si.z };
            ImGui::Text("Sun Irradiance (auto)");
            ImGui::InputFloat3("##irradiance", irradianceRGB, "%.3f", ImGuiInputTextFlags_ReadOnly);

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
    
    if (ImGui::DragFloat("Exposure", &m_Exposure, 0.01f, -24.0f, 24.0f))
    {
        // exposure doesn't require reset as it's just post-process
    }

    if (!m_UsePathTracer)
    {
        // Indirect GI — master checkbox, then the exclusive method enum
        // (Probe default). The UI state (enabled, method) is the single
        // source of truth; the engine flags are synced from it every frame.
        const bool wasGIEnabled = m_IndirectGIEnabled;
        ImGui::Checkbox("Enable Indirect GI", &m_IndirectGIEnabled);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Indirect (bounced) lighting for the raster path.\nDirect sun and local lights are always analytic regardless of this switch.");

        if (m_IndirectGIEnabled)
        {
            ImGui::Indent();

            // Index 3 is APPENDED, not inserted: 0/1/2 keep their meaning so no
            // stored setting or external reference is reinterpreted, and Probe
            // stays reachable at index 0 as the fallback.
            const char* giMethods[] = { "Probe (Baked)", "ReSTIR GI", "DDGI",
                                        "Probe (Baked Sun)" };
            const bool methodChanged = ImGui::Combo("Indirect GI Method", &m_IndirectGIMethod, giMethods, IM_ARRAYSIZE(giMethods));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Exclusive methods.\nProbe: direct sun stays runtime-exact; sun/sky bounces + occluded sky come from the baked grid. No local-light indirect, no indirect specular.\nReSTIR GI: spatiotemporal resampling with the SHaRC irradiance cache.\nDDGI: a runtime ray-traced octahedral irradiance probe volume. Re-traces every frame (no bake, no sun gating); sun light enters the probes through frontface hits only, so the probes store indirect light.\nProbe (Baked Sun): the sun is baked as a single response toward the real sun direction instead of being reconstructed from the 16 bins, and the whole fold is baked, so a sun or sky change needs a rebake. Direct sun stays runtime-exact.");

            // Exactly one of the three engine flags is ever set.
            m_FrameConstants.bakedGIMode            = ProbeMethodActive() ? 1u : 0u;
            m_FrameConstants.enableRestirGI = (m_IndirectGIMethod == 1) ? 1u : 0u;
            m_FrameConstants.ddgiMode      = (m_IndirectGIMethod == 2) ? 1u : 0u;

            // -----------------------------------------------------------------
            // A METHOD CHANGE IS AN EVENT, NOT A STATE. Service it here, once,
            // unconditionally.
            //
            // The old code requested work from inside "if (!gi.IsValid())", two
            // blocks further down. That is why the swap back into the
            // single-direction method left the image stuck: a bake already
            // existed, so the branch never ran, nothing armed m_PendingProbeFold,
            // the fold's gate evaluated false, and Lit kept whatever the previous
            // method had written. It also gated the request on
            // "!m_PendingGIProbeBake", which can swallow an event outright. A
            // pending request is idempotent, so that guard bought nothing and
            // risked losing the only retry.
            // -----------------------------------------------------------------
            if (methodChanged || !wasGIEnabled)
            {
                if (SingleDirectionSun())
                {
                    const bool haveSun = HasSunLight();
                    const DirectX::XMFLOAT3 sun = SunToDirNow();
                    BakedGI& giRef = m_Renderer.GetBakedGI();
                    // Arming the fold is free. A bake is 1.6 to 97 s on a UI
                    // action. So reuse the resident sun table when it already
                    // describes THIS sun, and rebake only when the table is
                    // missing or was baked for a different direction. The second
                    // case matters: a stale table would multiply a live sunE by a
                    // response for the wrong direction and look plausible.
                    if (haveSun && giRef.HasSunResponseFor(sun))
                        m_PendingProbeFold = true;
                    else if (haveSun)
                        m_PendingGIProbeBake = true;
                    // No sun: nothing to fold, and the fold gate requires a light.
                }
                else if (m_IndirectGIMethod == kIndirectMethodProbe)
                {
                    // Probe folds every frame from a sun-independent table, so an
                    // existing bake needs nothing at all.
                    if (!m_Renderer.GetBakedGI().IsValid())
                        m_PendingGIProbeBake = true;
                }
                // DDGI is deliberately NOT re-fitted here. Its grid depends on the
                // scene bounds, not on the method, so a valid grid is already
                // correct and re-fitting would cost a rebuild on every swap. Its
                // guard further down is structurally identical to the probe one
                // but cannot leave method-dependent stale data behind.
            }

            ImGui::Checkbox("Indirect Only View", &m_GIIndirectOnly);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Draw only the indirect-GI contribution and nothing else: the direct sun,\n"
                                  "the local lights, the reservoir composition and indirect specular are all\n"
                                  "skipped, and sky pixels render black. Use it to see what the indirect path\n"
                                  "actually adds to the scene.\n"
                                  "Covers the two sources that keep a separate indirect field: the baked probe\n"
                                  "grid and DDGI. Under ReSTIR GI the interchange textures mix direct and\n"
                                  "indirect, so the view is inert there and the normal image is drawn.");

            if (m_GIIndirectOnly)
            {
                ImGui::Indent();
                // Both probe methods write the sky/sun split buffers.
                const bool splitAvailable = ProbeMethodActive();
                ImGui::BeginDisabled(!splitAvailable);
                const char* indirectSources[] = { "Full (Sky + Sun)", "Sky Only", "Sun Only" };
                ImGui::SetNextItemWidth(180.f);
                ImGui::Combo("Indirect Source", &m_GIIndirectOnlySource, indirectSources, IM_ARRAYSIZE(indirectSources));
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Which half of the indirect light to show.\n"
                                      "The update pass bakes the total, the sky-sourced and the sun-sourced SH9\n"
                                      "tables separately, and total = sky + sun exactly, so the three views\n"
                                      "are directly comparable.\n"
                                      "Sky Only: sky bounces + occluded first-bounce sky (V_i gated).\n"
                                      "Sun Only: sun bounces only — the direct sun never enters the probes.\n"
                                      "The split exists only for Probe (Baked); DDGI stores one unsplit field.");

                const char* indirectAlbedos[] = { "Surface Albedo", "All White" };
                ImGui::SetNextItemWidth(180.f);
                ImGui::Combo("Indirect Albedo", &m_GIIndirectOnlyAlbedo, indirectAlbedos, IM_ARRAYSIZE(indirectAlbedos));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Albedo applied by the indirect-only view.\n"
                                      "Surface Albedo: the material's albedo texture, as in the normal image.\n"
                                      "All White: the albedo is forced to 1 in every channel, so the image is the\n"
                                      "indirect light field alone — no texture detail modulates it, which makes a\n"
                                      "lighting change easy to tell apart from an albedo change.\n"
                                      "Both modes keep the Lambert 1/PI, so they differ only by the albedo factor.");
                ImGui::Unindent();
            }

            if (ProbeMethodActive())
            {
                ImGui::SliderFloat("Probe Spacing", &m_BakedGISpacing, 0.5f, 8.0f, "%.1f m");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Grid spacing used by the next \"Rebuild GI Probes\" (auto-grows to fit the probe cap).");

                if (ImGui::Button("Rebuild GI Probes"))
                    m_PendingGIProbeBake = true;

                // --- Diagnostic: pair the sun's RNG seed with the dominant bin ---
                // The bake seeds from the direction index, so the sun pass (dir 0)
                // gathers a DIFFERENT set of 256 directions than bin 14. A
                // bin-versus-sun difference then mixes an angular effect with a
                // sampling effect, and the two cannot be told apart. This makes
                // both trace the same directions, so only wDelta differs.
                //
                // It rewrites the baked sun table, so a toggle must request a
                // rebake. Without that the control does nothing visible and the
                // next comparison silently reads the old table. Method 3 only:
                // the binned method has no single-sun pass to pair.
                ImGui::BeginDisabled(!SingleDirectionSun());
                if (ImGui::Checkbox("Pair sun seed with dominant bin (diagnostic)",
                                    &m_GISunSeedPairDominantBin))
                    m_PendingGIProbeBake = true;
                ImGui::EndDisabled();
                // AllowWhenDisabled: the tooltip is the only place that says WHY
                // the control is greyed out, so it must still appear.
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Makes the single-sun pass trace the SAME 256 gather directions as\n"
                                      "the dominant fibonacci bin (the sun fold's argmax, 14 for the captured\n"
                                      "sun), instead of its own stream. Both methods then re-test the same hit\n"
                                      "surfaces, so the difference between them is angular and not sampling\n"
                                      "noise. Diagnostic only.\n"
                                      "It CHANGES the baked sun table, so it requests a rebake. Wait for the bake\n"
                                      "before you compare anything.\n"
                                      "OFF by default, so the shipped path is unchanged.\n"
                                      "Method \"Probe (Baked Sun)\" only.");

                // Ray-debug tool: re-trace ONE probe and print the evidence.
                ImGui::InputInt("Ray Debug Probe", &m_RayDebugProbeIndex);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Probe index for the ray-debug dump.\n"
                                      "Index = x + y*dimX + z*dimX*dimY, clamped to the grid.\n"
                                      "Do not derive this by hand: read it off \"Probes Near Camera\"\n"
                                      "below, whose rows set this field and dump in one click.\n"
                                      "The dump re-traces the 64 gather rays of this probe and prints:\n"
                                      "the 16-direction ranking, the 64-ray ranking, and a self-check\n"
                                      "that compares the re-trace with the stored table.\n"
                                      "The Ray Fan debug view draws the same probe, so the picture and\n"
                                      "the text always describe one probe.");
                if (ImGui::Button("Dump Probe Ray Debug"))
                    m_PendingRayDebug = true;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Re-traces one probe's gather rays and prints the readout to the console.\n"
                                      "Debug only, on demand — no per-frame cost. See the log section\n"
                                      "[BakedGI RayDebug].");

                // --- Sun fold bins ---
                // The answer to "which bins interpolate this probe, and by how
                // much". Two DIFFERENT quantities side by side:
                //   weight       GLOBAL     w_j, the same for every probe
                //   contribution PER PROBE  w_j * R_j[0], this probe's table
                // Weight alone points at the wrong bin, because R_j varies up to
                // 4x between bins. Do not show one without the other.
                ImGui::Separator();
                ImGui::Text("Sun Fold Bins");
                // Method 3 does not run this fold. The weights below are still the
                // RIGHT weights for the sun direction, but nothing consumes them:
                // the sun column comes from one baked term instead. Say so here,
                // because a panel that silently shows live weights for a fold that
                // is not running is worse than no panel.
                if (SingleDirectionSun())
                {
                    ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                                       "FOLD IS BAKED under \"Probe (Baked Sun)\".");
                    ImGui::TextWrapped("The weights and angles below are the kernel's, "
                                       "but this method does not use the kernel: the sun "
                                       "column is one response baked toward the real sun. "
                                       "The contribution column reads the 16-bin table, "
                                       "which is still baked for the sky quadrature and "
                                       "for these debug views.");
                }
                {
                    BakedGI& binGI = m_Renderer.GetBakedGI();

                    // The to-sun direction, negated from the light's travel
                    // direction exactly as the update pass and FillSunFoldConstants
                    // negate it. One negation rule, three readers.
                    const bool haveSun = HasSunLight();
                    const DirectX::XMFLOAT3 sunToDir = SunToDirNow();

                    if (ImGui::Button("Refresh Bin Contributions"))
                    {
                        m_PendingProbeTableReadback = true;
                        m_ProbeTableRequestedProbe  = (uint32_t)std::max(0, m_RayDebugProbeIndex);
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(reads %u's table)", (uint32_t)std::max(0, m_RayDebugProbeIndex));
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Reads ONE probe's response table (2560 B) and waits for the GPU.\n"
                                          "On demand only — never per frame. The weight and angle columns\n"
                                          "below are pure CPU and update as you drag the sun.");

                    // A probe change re-requests once. A FAILED request does not
                    // retry every frame: each retry costs a GPU wait, so the
                    // Refresh button is the way back.
                    const uint32_t wantProbe = (uint32_t)std::max(0, m_RayDebugProbeIndex);
                    if (binGI.IsValid() && !m_PendingProbeTableReadback
                        && m_ProbeTableRequestedProbe != wantProbe)
                    {
                        m_PendingProbeTableReadback = true;
                        m_ProbeTableRequestedProbe  = wantProbe;
                    }

                    BakedGI::SunBinPanel panel;
                    binGI.BuildSunBinPanel(haveSun ? &sunToDir : nullptr, wantProbe, panel);

                    if (panel.haveDirection)
                    {
                        ImGui::Text("Sun (to-sun)  elev %.1f deg   azim %.1f deg"
                                    "   (%s)", panel.elevationDeg, panel.azimuthDeg,
                                    panel.haveSun ? "fold active" : "below horizon, or no sun: all weights 0");
                    }
                    if (panel.haveSun)
                    {
                        ImGui::Text("Weighted mean angle %.2f deg", panel.meanAngleDeg);
                        ImGui::SameLine();
                        ImGui::TextColored((panel.weightBeyond15 > 0.5f)
                                               ? ImVec4(0.95f, 0.55f, 0.20f, 1.0f)
                                               : ImVec4(0.75f, 0.75f, 0.75f, 1.0f),
                                           "| weight beyond 15 deg %.1f%%", panel.weightBeyond15 * 100.0f);
                        ImGui::SameLine();
                        // Name the THRESHOLD, not "nonzero". Only 1 of the 16
                        // weights is exactly 0 (a bin whose direction faces
                        // away from the sun); 15 are nonzero and all 15 enter
                        // the fold, because BakedGI_Update.hlsl:84 skips only
                        // bw <= 0. So "nonzero" would assert 16 when the count
                        // below is 11. The threshold is the honest label.
                        ImGui::Text("| bins with weight > 1e-6: %u / %u",
                                    panel.nonzeroBins, BakedGI::kDirections);
                    }

                    // Legend. The two markers are different picks and the panel
                    // never calls either one "dominant": a bare dominant label
                    // already caused a defect in this codebase.
                    ImGui::TextDisabled("w*  = fold-weight argmax  = bin %u  (the Ray Fan draws this bin)",
                                        panel.weightArgmax);
                    if (panel.haveResponse)
                        ImGui::TextDisabled("dc* = highest stored DC luminance = bin %u  (NOT the same pick)",
                                            panel.dcArgmax);

                    if (!panel.haveResponse)
                    {
                        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                                           "No table read back for probe %u, so the contribution column is empty.",
                                           wantProbe);
                        if (!binGI.IsValid())
                            ImGui::TextDisabled("(no bake yet — press \"Rebuild GI Probes\")");
                        else
                            ImGui::TextDisabled("(press \"Refresh Bin Contributions\"; sorted by weight until then)");
                    }

                    if (ImGui::BeginTable("SunBinTable", 5,
                                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg
                                              | ImGuiTableFlags_SizingStretchProp))
                    {
                        ImGui::TableSetupColumn("Bin",  ImGuiTableColumnFlags_WidthFixed, 62.0f);
                        ImGui::TableSetupColumn("Weight % (GLOBAL)");
                        ImGui::TableSetupColumn("Angle deg");
                        ImGui::TableSetupColumn("Contribution (PER PROBE)");
                        ImGui::TableSetupColumn("% of probe");
                        ImGui::TableHeadersRow();

                        for (uint32_t r = 0; r < BakedGI::kDirections; ++r)
                        {
                            const BakedGI::SunBinRow& row = panel.rows[r];
                            // A zero-weight bin contributes nothing and cannot be
                            // the answer to "which bins interpolate". Hide it, but
                            // keep the count honest in the header above.
                            if (row.weight <= 1e-6f && !panel.haveResponse)
                                continue;

                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            {
                                const char* mark = (row.bin == panel.weightArgmax) ? " w*"
                                                 : (panel.haveResponse && row.bin == panel.dcArgmax) ? " dc*"
                                                 : "";
                                ImGui::Text("%2u%s", row.bin, mark);
                            }
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%6.2f %%", row.weight * 100.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%6.2f", row.angleDeg);
                            ImGui::TableSetColumnIndex(3);
                            if (panel.haveResponse)
                            {
                                // Tint by the contribution's own hue, so a bin that
                                // carries red reads red. The number is luminance;
                                // the tooltip carries the channels.
                                const DirectX::XMFLOAT3& c = row.contribRGB;
                                const float mx = std::max(c.x, std::max(c.y, c.z));
                                const ImVec4 tint = (mx > 1e-12f)
                                    ? ImVec4(0.25f + 0.75f * (c.x / mx), 0.25f + 0.75f * (c.y / mx),
                                             0.25f + 0.75f * (c.z / mx), 1.0f)
                                    : ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
                                ImGui::TextColored(tint, "%.5f", row.contrib);
                                if (ImGui::IsItemHovered())
                                    ImGui::SetTooltip("weight x R_bin[slot 0], per channel:\n"
                                                      "R %.5f   G %.5f   B %.5f\n"
                                                      "R_bin[0] luminance %.5f",
                                                      row.contribRGB.x, row.contribRGB.y, row.contribRGB.z,
                                                      row.dcLum);
                            }
                            else
                            {
                                ImGui::TextDisabled("--");
                            }
                            ImGui::TableSetColumnIndex(4);
                            if (panel.haveResponse)
                                ImGui::Text("%6.2f %%", row.contribPct * 100.0f);
                            else
                                ImGui::TextDisabled("--");
                        }
                        ImGui::EndTable();
                    }
                }

                // --- Probes near the camera ---
                // The answer to "which probe index am I standing in?". The index
                // is read off a list instead of being derived from coordinates,
                // and a row click selects AND dumps in one step.
                ImGui::Separator();
                ImGui::Text("Probes Near Camera");
                {
                    BakedGI& nearGI = m_Renderer.GetBakedGI();
                    const DirectX::XMFLOAT3 camPos = m_Camera.GetPosition();
                    const float spacing = nearGI.GetSpacing();

                    // A re-bake changes the grid under the list. The revision is a
                    // counter, so this test cannot miss a rebuild the way a float
                    // compare can.
                    if (nearGI.GetGridRevision() != m_NearestProbesGridRev)
                    {
                        m_NearestProbesValid   = false;
                        m_NearestProbesGridRev = nearGI.GetGridRevision();
                    }

                    ImGui::SetNextItemWidth(60.f);
                    if (ImGui::InputInt("Rows", &m_NearestProbeRows, 1, 4))
                        m_NearestProbesValid = false; // the row count changed the query
                    if (m_NearestProbeRows < 1)  m_NearestProbeRows = 1;
                    if (m_NearestProbeRows > 32) m_NearestProbeRows = 32;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("How many of the nearest VALID probes to list.\n"
                                          "Only probes whose bake meta bit says valid are candidates,\n"
                                          "so an invalid (dead) probe is never offered.");

                    ImGui::SameLine();
                    if (ImGui::Button("Refresh"))
                        m_NearestProbesValid = false;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Recompute now. The list also recomputes itself, but only\n"
                                          "after the camera has moved about one probe spacing: a list\n"
                                          "that re-sorts every frame cannot be clicked.");

                    // Rebuild only when stale. The trigger is the plain distance
                    // the camera has moved since the build, not a per-frame
                    // recompute, so the rows hold still long enough to click.
                    const float rebuildDist = (spacing > 0.0f) ? spacing : 1.0f;
                    const float mdx = camPos.x - m_NearestProbesOrigin.x;
                    const float mdy = camPos.y - m_NearestProbesOrigin.y;
                    const float mdz = camPos.z - m_NearestProbesOrigin.z;
                    const float moved2 = mdx * mdx + mdy * mdy + mdz * mdz;
                    if (!m_NearestProbesValid || moved2 > rebuildDist * rebuildDist)
                    {
                        nearGI.FindNearestValidProbes(camPos, (uint32_t)m_NearestProbeRows, m_NearestProbes);
                        m_NearestProbesOrigin = camPos;
                        m_NearestProbesValid  = true;
                    }

                    if (m_NearestProbes.empty())
                    {
                        ImGui::TextDisabled("No valid probe to list: the bake has not run, or no "
                                            "probe is valid.");
                    }
                    else
                    {
                        // Stretch sizing, four columns: the panel is often narrow, and a
                        // six-column fixed table would clip the numbers off the right edge.
                        // The detail that does not fit (centre position, spacings) moves to
                        // the per-row tooltip instead of being dropped.
                        const ImGuiTableFlags nearFlags = ImGuiTableFlags_BordersInnerV |
                                                          ImGuiTableFlags_RowBg |
                                                          ImGuiTableFlags_SizingStretchProp;
                        if (ImGui::BeginTable("##nearestProbes", 4, nearFlags))
                        {
                            ImGui::TableSetupColumn("#",       ImGuiTableColumnFlags_WidthFixed, 24.f);
                            ImGui::TableSetupColumn("Index",   ImGuiTableColumnFlags_WidthFixed, 48.f);
                            ImGui::TableSetupColumn("Cell (x,y,z)");
                            ImGui::TableSetupColumn("Dist (m)", ImGuiTableColumnFlags_WidthFixed, 60.f);
                            ImGui::TableHeadersRow();

                            for (size_t i = 0; i < m_NearestProbes.size(); ++i)
                            {
                                const BakedGI::NearestProbe& np = m_NearestProbes[i];
                                ImGui::TableNextRow();
                                ImGui::TableSetColumnIndex(0);
                                const std::string rowId = std::to_string(i + 1u);
                                // The row matching the current dump index reads as selected, so
                                // the list and the "Ray Debug Probe" field above agree.
                                const bool isCurrent =
                                    (np.index == (uint32_t)std::max(0, m_RayDebugProbeIndex));
                                if (ImGui::Selectable(rowId.c_str(), isCurrent,
                                                      ImGuiSelectableFlags_SpanAllColumns))
                                {
                                    m_RayDebugProbeIndex = (int)np.index;
                                    m_PendingRayDebug    = true; // pick and dump in one step
                                }
                                if (ImGui::IsItemHovered())
                                    ImGui::SetTooltip("Click to set \"Ray Debug Probe\" to %u and dump it now —\n"
                                                      "the same dump the button above runs, in one step.\n"
                                                      "Cell (%u,%u,%u), centre (%.2f, %.2f, %.2f) m.\n"
                                                      "Distance %.2f m = %.2f probe spacings.\n"
                                                      "Index = x + y*dimX + z*dimX*dimY, the same convention as\n"
                                                      "the field above and the probe the Ray Fan draws.",
                                                      np.index, np.cellX, np.cellY, np.cellZ,
                                                      np.center.x, np.center.y, np.center.z,
                                                      np.distance,
                                                      (spacing > 1e-6f) ? np.distance / spacing : 0.0f);
                                ImGui::TableSetColumnIndex(1);
                                ImGui::Text("%u", np.index);
                                ImGui::TableSetColumnIndex(2);
                                ImGui::Text("%u,%u,%u", np.cellX, np.cellY, np.cellZ);
                                ImGui::TableSetColumnIndex(3);
                                ImGui::Text("%.2f", np.distance);
                            }
                            ImGui::EndTable();
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("The nearest valid probes to the camera, ascending by distance.\n"
                                              "Only probes the bake meta marks valid are listed.\n"
                                              "Distances are from the camera position at the last rebuild:\n"
                                              "fly, then press Refresh (or move a full spacing) to re-sort.");
                    }
                }
                ImGui::Separator();

                // "Sun Directions" was folded into the Ray Fan's DeltaDirections
                // channel: the same 16 boxes and the same weight ramp, but at the
                // one cell the fan is already at instead of repeating 16 identical
                // colours at every probe in the window.
                const char* probeViews[] = { "Off", "Placement", "Lit Irradiance", "Sky Visibility", "Visibility Rays",
                                             "Lit Sky Source", "Lit Sun Source", "Ray Fan", "Efb Contribution",
                                             "Cage Trace" };
                if (SingleDirectionSun())
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.75f, 0.30f, 1.0f));
                    ImGui::TextWrapped("%s", kMethod3WordingCaveat);
                    ImGui::PopStyleColor();
                }
                ImGui::SetNextItemWidth(180.f);
                ImGui::Combo("Probe Debug View", &m_BakedGIDebugView, probeViews, IM_ARRAYSIZE(probeViews));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Post-composite overlay, depth-tested against the GBuffer.\n"
                                      "Placement: cube per cell — classification A/B.\n"
                                      "Green = valid under both criteria. Red = bad under both.\n"
                                      "BLUE = newly bad: the old 1cm test said valid, DDGI's rule\n"
                                      "(>= 25% backfaces among 32 stable rays) kills it.\n"
                                      "Yellow = newly valid (old test killed it, DDGI keeps it).\n"
                                      "Lit Irradiance: SH ball per probe — each sphere point evaluates the probe's lit SH9 at its own normal (directional structure of the baked+lit field).\n"
                                      "Lit Sky / Lit Sun Source: same ball reading the update-pass SPLIT buffers —\n"
                                      "sky = sky bounces + occluded sky direct (V_i-gated), sun = sun bounces only (direct sun never enters probes).\n"
                                      "The three lit views always sum: total = sky + sun.\n"
                                      "Sky Visibility: grayscale cube = mean of the 16 baked sky visibilities (sky-open fraction); red = invalid.\n"
                                      "Visibility Rays: per-probe fibonacci directions — green = open sky, red = blocked (near camera only, invalid probes skipped).\n"
                                      "Ray Fan: the 64 gather rays of ONE direction of one probe (the Ray Debug Probe above), as thin boxes.\n"
                                       "  Probe (Baked): the SUN-DOMINANT direction, i.e. the bin the fold's kernel weights most.\n"
                                       "  Probe (Baked Sun): the REAL SUN DIRECTION. That method traces a single direction\n"
                                       "  toward the sun at bake time, so there is no dominant bin and the 16 delta\n"
                                       "  directions shown by the DeltaDirections channel carry NO weight in its fold.\n"
                                      "It draws the records of the LAST dump, so press \"Dump Probe Ray Debug\" first and again after a rebuild.\n"
                                      "Its DeltaDirections channel swaps the geometry: the 16 delta directions of that same probe,\n"
                                      "grayscale by the weight in the sun fold, white = the direction the fan itself draws, plus\n"
                                      "one thicker ORANGE box along the REAL sun, so the fold's angular error is visible. That\n"
                                      "channel reads no record, so it works before any dump.\n"
                                      "Efb Contribution: SH ball per probe of the SERIES-FEEDBACK contribution — the selected\n"
                                      "bake term's response ball minus term 1's (the no-feedback baseline), so it is exactly what\n"
                                      "the Efb term added, and where. The ball's DC reads as the feedback's share of that term's\n"
                                      "transport. Term 1 is black by construction: the bake gates feedback off there. Needs a bake\n"
                                      "from the dispatch path; a Probes/ cache load has no per-term tables and draws nothing.\n"
                                      "Cage Trace: ONE probe (the Ray Debug Probe above). It derives the chain — the sun-dominant\n"
                                      "direction, that direction's largest FEEDBACK-DOMINATED ray (a shaded ray whose Efb term\n"
                                      "beats its NdotD*Vhit term), a line from the probe to that ray's hit, and the 8 trilinear\n"
                                      "cage probes the bake's feedback gather actually sampled at that hit. GREEN = the largest\n"
                                      "share of the gather, AMBER = middle, RED = smallest or a member the backface test culled.\n"
                                      "The dim wireframe is the trilinear cell of the PUSHED read point (hit + n*0.1*spacing);\n"
                                      "a cage face swapped for another cell means the push crossed that box. X-RAY: this view\n"
                                      "skips the depth test. Needs a bake AND a dump — press \"Dump Probe Ray Debug\" first.");

                if (m_BakedGIDebugView == 2 || m_BakedGIDebugView == 5 || m_BakedGIDebugView == 6 ||
                    m_BakedGIDebugView == 8)
                {
                    ImGui::SliderFloat("Lit Gain", &m_BakedGIDebugScale, 0.1f, 10.0f, "%.2f");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Gain on the DC-normalized ball shading: the probe's mean irradiance\n"
                                          "maps to ~1.0 x gain, band structure swings around it. Same normalization\n"
                                          "reference (the total buffer's DC) for the Irradiance / Sky / Sun views.\n"
                                          "The Efb Contribution view normalizes by the SELECTED term's own ball DC\n"
                                          "instead, so its DC reads as the feedback's share of that term's transport.");
                }

                if (m_BakedGIDebugView == 8)
                {
                    // Efb Contribution: which bake term to show. The bake runs
                    // exactly kBakeIterations terms and the capture stores a ball
                    // for each, so the combo's entry count is that constant — the
                    // static_assert in the constants block above is what keeps the
                    // two in step. The label is 1-based and the member 0-based:
                    // the field is called "iteration" in the bake (0 = no
                    // feedback, BakedGI_Bake.hlsl:236), and a "Term 0" label on a
                    // view that cannot draw anything would invite the reader to
                    // think it is broken.
                    const char* efbIters[] = { "1", "2", "3" };
                    ImGui::SetNextItemWidth(180.f);
                    ImGui::Combo("Iteration", &m_BakedGIEfbIter, efbIters, IM_ARRAYSIZE(efbIters));
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Which bake term's series-feedback (Efb) contribution to show.\n"
                                          "The bake runs 3 terms (%u = kBakeIterations, 3 series bounces).\n"
                                          "1: feedback is GATED OFF, so this view is black by construction —\n"
                                          "   a check that the gate works, not an empty view.\n"
                                          "2: the first feedback term, and 3: the last (the converged one).\n"
                                          "The baseline is always term 1, so the ball is what the feedback added.",
                                          (unsigned)BakedGI::kBakeIterations);

                    // Which delta direction the ball covers. Entry 0 is "All" and
                    // maps to the member's -1, so the combo index IS the direction
                    // + 1 — one binding, no translation table. The list is
                    // BakedGI::kDirections + 1 entries; the fill helper clamps, so
                    // a mismatch can never publish a direction that does not exist.
                    const char* efbDirs[] = { "All (16-dir fold)", "0", "1", "2", "3", "4", "5", "6", "7",
                                              "8", "9", "10", "11", "12", "13", "14", "15" };
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(180.f);
                    ImGui::Combo("Direction", &m_BakedGIEfbDir, efbDirs, IM_ARRAYSIZE(efbDirs));
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Which delta direction the ball covers.\n"
                                          "\"All\" folds all 16 into the ball — the summed view.\n"
                                          "A single direction reads that ONE direction of the two transport\n"
                                          "tables instead, so its own colour is not averaged against 15\n"
                                          "others. Use it when a per-direction tint (a red bounce, say) is\n"
                                          "swamped by the fold: red on 1 of 16 directions stays invisible.\n"
                                          "Both modes normalise by the selected term's own DC (that\n"
                                          "direction's DC in single mode), so the ball keeps one reading:\n"
                                          "DC(diff)/DC(term) cannot exceed 1, the contribution being\n"
                                          "non-negative and additive.");

                    const BakedGI& efbGI = m_Renderer.GetBakedGI();
                    if (!efbGI.HasEfbBalls())
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.62f, 0.15f, 1.0f));
                        ImGui::TextWrapped("No per-term capture. Press \"Rebuild GI Probes\": the response "
                                           "balls are folded on the bake's dispatch path, and a Probes/ cache "
                                           "load restores only the final table.");
                        ImGui::PopStyleColor();
                    }
                    else if (m_BakedGIEfbIter == 0)
                    {
                        ImGui::TextDisabled("Term 1 has no feedback term (the bake gates it off), so every ball "
                                            "here is black by construction.");
                    }
                }

                if (m_BakedGIDebugView == 7)
                {
                    // Ray Fan: same probe as the text dump, so the two reads agree.
                    // The last entry must stay last and must stay at index
                    // BakedGI::kRayFanChannelDeltaDirs: the renderer reads the
                    // channel to choose the instance count, and the shader reads
                    // it to choose the geometry. Three places, one value.
                    const char* rayFanChannels[] = { "Status", "Albedo", "NdotD", "Vhit", "Efb",
                                                     "Hit Distance", "Direct vs Efb", "DeltaDirections" };
                    if (SingleDirectionSun())
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.75f, 0.30f, 1.0f));
                        ImGui::TextWrapped("%s", kMethod3WordingCaveat);
                        ImGui::PopStyleColor();
                    }
                    ImGui::SetNextItemWidth(180.f);
                    ImGui::Combo("Ray Fan Channel", &m_BakedGIRayDebugChannel, rayFanChannels, IM_ARRAYSIZE(rayFanChannels));
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Colour of each box. Channels 0..6 colour the 64 gather rays of the\n"
                                          "sun-dominant direction (see below).\n"
                                          "Status: grey = sky miss, blue = backface reject, orange = shaded.\n"
                                          "NdotD: cos(hit normal, the sun-dominant direction) drawn straight — 1 is white, 0 is black.\n"
                                          "Albedo / Vhit / Efb: the record fields.\n"
                                          "Hit Distance: log ramp, sky miss near black.\n"
                                          "Direct vs Efb: green = NdotD*Vhit dominates, red = Efb dominates.\n"
                                          "DeltaDirections: swaps the geometry — the 16 delta directions of the\n"
                                          "same probe, one box each, drawn with the NdotD channel's straight\n"
                                          "grayscale on the sun-fold weight divided by the largest of the 16.\n"
                                          "WHITE is 1.0 = the most weighted direction, which is the direction\n"
                                          "the fan draws. One more box, thicker and ORANGE, points along the\n"
                                          "REAL sun: the angle between it and the bin boxes is the fold's\n"
                                          "angular error. It does not draw with no sun, or with the sun below\n"
                                          "the horizon. It reads no record, so it needs no dump.");

                    const bool deltaDirs =
                        ((uint32_t)m_BakedGIRayDebugChannel == BakedGI::kRayFanChannelDeltaDirs);

                    // Which direction the fan draws, and how strongly the sun
                    // fold weights it. A weight of 0 means no sun (the sun is
                    // below the horizon, or the direction is zero), so direction
                    // 0 is then not a real winner — say so, do not imply a pick.
                    // Method 3 has ONE direction, traced toward the real sun. There
                    // is no "dominant" bin there, and saying so would be false: the
                    // fold does not touch the 16 bins at all.
                    if (SingleDirectionSun())
                        ImGui::Text("Ray Fan: the SUN direction (method 3 traces one direction, not 16)");
                    else if (m_BakedGISunWeight > 0.0f)
                        ImGui::Text("Ray Fan: sun-dominant direction %d of 16, share %.3f of the fold",
                                    m_BakedGISunDir, m_BakedGISunWeight);
                    else
                        ImGui::Text("Ray Fan: no sun. Direction %d of 16, weight 0.000",
                                    m_BakedGISunDir);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("The sun fold weights each of the 16 delta directions by\n"
                                          "pow(max(0, dot(direction, toSun)), 8), then normalises.\n"
                                          "The fan draws the 64 gather rays of the largest one, so every\n"
                                          "drawn ray is one the sun actually weights. The dump's kernel\n"
                                          "weight line marks the same direction with an arrow.");

                    // Both widgets below describe the RECORD channels only.
                    // DeltaDirections reads no record and draws a fixed-length box,
                    // so neither applies. Offering a ray-length slider that changes
                    // nothing is the kind of dead control this panel has been
                    // shedding, and it would also suggest the 16 boxes are rays.
                    if (!deltaDirs)
                    {
                        // The dump zeroes an invalid probe's stored table, so the
                        // channels that read that table are degenerate. The fan dims
                        // them (BAKED_GI_RAYFAN_INVALID_DIM). Say which ones here too.
                        const BakedGI& fanGI = m_Renderer.GetBakedGI();
                        if (fanGI.HasRayDebugBuffer() && fanGI.GetRayDebugKnownInvalid())
                        {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.62f, 0.15f, 1.0f));
                            ImGui::TextWrapped("Warning: the dumped probe is INVALID. Channels 4 "
                                               "(Efb) and 6 (Direct vs Efb) are degenerate and drawn "
                                               "dimmed.");
                            ImGui::PopStyleColor();
                        }

                        ImGui::SliderFloat("Ray Length (spacings)", &m_BakedGIRayDebugLength, 0.5f, 32.0f, "%.1f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Ray length cap in probe spacings. A hit shorter than the cap draws\n"
                                              "at its true distance; a longer hit and a sky miss draw at the cap.\n"
                                              "The leak the fan is chasing spans about 6 spacings.");
                    }
                }

                if (m_BakedGIDebugView == 9)
                {
                    // Cage Trace. The panel says WHICH probe and WHICH ray will be
                    // traced. The probe is the dump's; the ray is either the
                    // shader's own argmax or a row of the pool below.
                    //
                    // The pool is the dump's OWN readback (BakedGI::GetRayCandidates),
                    // so its numbers are the trace's numbers rather than a second
                    // computation, and it is kept for all 16 directions — hence the
                    // lookup by the CURRENT sun-dominant direction: a sun move after
                    // the dump re-lists instead of describing another direction.
                    const BakedGI& cageGI = m_Renderer.GetBakedGI();
                    if (!cageGI.IsValid())
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.62f, 0.15f, 1.0f));
                        ImGui::TextWrapped("No bake. The trace reads the baked depth moments for the "
                                           "Chebyshev weight — press \"Rebuild GI Probes\".");
                        ImGui::PopStyleColor();
                    }
                    else if (!cageGI.HasRayDebugBuffer())
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.62f, 0.15f, 1.0f));
                        ImGui::TextWrapped("No ray dump. Press \"Dump Probe Ray Debug\" above — the trace "
                                           "follows the dumped probe's records.");
                        ImGui::PopStyleColor();
                    }
                    else if (SingleDirectionSun())
                    {
                        // No argmax exists under method 3: the dump wrote one
                        // direction, toward the real sun. Naming a bin here would
                        // point at a direction that method's fold never reads.
                        ImGui::Text("Cage Trace: probe %u, the SUN direction (method 3) — no dominant bin",
                                    cageGI.GetRayDebugProbe());
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("The probe is the dumped one. The direction is the REAL SUN\n"
                                              "DIRECTION, which is the only direction this method traces.\n"
                                              "There is no argmax and no 16-bin fold to name a direction from.");
                    }
                    else
                    {
                        ImGui::Text("Cage Trace: probe %u, sun-dominant direction %d of 16",
                                    cageGI.GetRayDebugProbe(), m_BakedGISunDir);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("The probe is the dumped one. The direction is the sun fold's argmax,\n"
                                              "the same pick the Ray Fan draws. What the trace draws:\n"
                                              "  the probe -> hit segment, then the 8 trilinear cage probes the\n"
                                              "  bake's feedback gather actually sampled at that hit, then the\n"
                                              "  trilinear CELL of the pushed read point (hit + n*0.1*spacing) as a\n"
                                              "  dim wireframe — a cage face falling into another cell means that\n"
                                              "  push crossed the box.\n"
                                              "GREEN = the largest share of the 8-probe gather, AMBER = the middle,\n"
                                              "RED = the smallest or a backface-culled member (share 0). The ramp\n"
                                              "is scaled by the LARGEST share, so a collapsed cage still reads —\n"
                                              "the absolute level is the dump's readout, not this colour.\n"
                                              "X-RAY: the view skips the depth test.");

                        if (cageGI.GetRayCandidateDirs() == 0u)
                        {
                            // A ray-debug buffer exists, so a dump happened. The pool is
                            // filled by that dump's readout, so an empty one means the
                            // readout did not complete — not that the rule found nothing.
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.62f, 0.15f, 1.0f));
                            ImGui::TextWrapped("No picker pool: the last dump's readout did not complete "
                                               "(the readback could not be mapped), so the ray cannot be "
                                               "steered and the trace is the shader's own pick.");
                            ImGui::PopStyleColor();
                        }
                        else
                        {
                            const uint32_t dir =
                                (m_BakedGISunDir >= 0 && m_BakedGISunDir < (int)BakedGI::kDirections)
                                    ? (uint32_t)m_BakedGISunDir : 0u;
                            const auto& pool = cageGI.GetRayCandidates(dir);

                            const BakedGI::RayCandidate* sel = nullptr;
                            for (const BakedGI::RayCandidate& c : pool)
                            {
                                if ((int)c.ray == m_BakedGIRayDebugRay) { sel = &c; break; }
                            }

                            char preview[192];
                            if (m_BakedGIRayDebugRay < 0)
                                snprintf(preview, sizeof(preview),
                                         "Auto - largest carried Efb (%zu in pool)", pool.size());
                            else if (sel)
                                snprintf(preview, sizeof(preview), "ray %u  efb %.3f  direct %.3f",
                                         sel->ray, sel->efbLum, sel->directLum);
                            else
                                snprintf(preview, sizeof(preview), "ray %d  (not in this pool)",
                                         m_BakedGIRayDebugRay);

                            ImGui::SetNextItemWidth(300.f);
                            if (ImGui::BeginCombo("Cage Ray", preview))
                            {
                                if (ImGui::Selectable("Auto - largest carried Efb", m_BakedGIRayDebugRay < 0))
                                    m_BakedGIRayDebugRay = -1;
                                for (const BakedGI::RayCandidate& c : pool)
                                {
                                    char label[192];
                                    snprintf(label, sizeof(label),
                                             "ray %2u   score %.3f   efb %.3f   direct %.3f   alb %.3f",
                                             c.ray, c.score, c.efbLum, c.directLum, c.albedoLum);
                                    if (ImGui::Selectable(label, (int)c.ray == m_BakedGIRayDebugRay))
                                        m_BakedGIRayDebugRay = (int)c.ray;
                                }
                                if (pool.empty())
                                    ImGui::TextDisabled("(no feedback-dominated ray here)");
                                ImGui::EndCombo();
                            }
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip("Which gather ray of direction %u to trace. The pool is every SHADED\n"
                                                  "ray of that direction whose Efb term beats its NdotD*Vhit term —\n"
                                                  "\"direct < Efb\" — enumerated from the dump's own readback, so these\n"
                                                  "numbers ARE the record's.\n"
                                                  "score = luminance(diffAlbedo * Efb), the trace's ranking key: the\n"
                                                  "bake's band-0 coefficient is one constant times the ray's radiance,\n"
                                                  "and diffAlbedo stays inside the product because a ray onto a dark or\n"
                                                  "blue surface carries a large Efb and contributes almost nothing.\n"
                                                  "Auto leaves the pick to the shader's argmax, which is this list's\n"
                                                  "largest score. The rows are in RAY order; the printed step-2c table\n"
                                                  "is the score order of the same list.\n"
                                                  "The pick survives a dump on purpose (it keeps tracking one ray index\n"
                                                  "while the probe changes). A pick that is not in the current\n"
                                                  "direction's pool is drawn MAGENTA.", dir);
                        }

                        if (m_BakedGIRayDebugRay >= 0)
                        {
                            const auto& check = cageGI.GetRayCandidates(
                                (m_BakedGISunDir >= 0 && m_BakedGISunDir < (int)BakedGI::kDirections)
                                    ? (uint32_t)m_BakedGISunDir : 0u);
                            bool inPool = false;
                            for (const BakedGI::RayCandidate& c : check)
                                if ((int)c.ray == m_BakedGIRayDebugRay) { inPool = true; break; }

                            if (!inPool)
                            {
                                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.87f, 0.18f, 0.87f, 1.0f));
                                ImGui::TextWrapped("ray %d is not in this direction's pool. The shader still "
                                                   "traces it when it is a shaded ray — the probe->hit segment "
                                                   "is then MAGENTA, marking that its hit is not an Efb "
                                                   "borrow. A sky miss or a backface reject draws nothing.",
                                                   m_BakedGIRayDebugRay);
                                ImGui::PopStyleColor();
                            }
                        }
                    }
                }

                bool leakMap = (m_FrameConstants.bakedGILeakDebug != 0);
                if (ImGui::Checkbox("Backface Leak Map", &leakMap))
                    m_FrameConstants.bakedGILeakDebug = leakMap ? 1u : 0u;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Fullscreen map drawn over the tonemapped scene (sky passes through; probe views draw on top).\n"
                                      "Per surface: fraction of the trilinear probe blend coming from probes BEHIND the\n"
                                      "receiver's tangent plane — the wrong-side-of-wall leak that backface rejection would remove.\n"
                                      "Green = clean cage, red ramp = contamination, blue = no valid probes (dark-leak risk).");

                ImGui::Checkbox("Visibility Check (Chebyshev)", &m_BakedGIVisCheck);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("A/B the step-3 probe-fetch upgrade. Instant per-frame toggle — no re-bake needed.\n"
                                      "ON:  unified self-shadow bias + backface weight + Chebyshev occlusion from\n"
                                      "     the baked per-probe depth moments (wall/roof geometry rejects wrong-side\n"
                                      "     cage probes for that receiver only).\n"
                                      "OFF: plain validity-renormalized trilinear (the step-1 fetch).");

                const BakedGI& gi = m_Renderer.GetBakedGI();
                if (gi.IsValid())
                    ImGui::Text("Grid %ux%ux%u @ %.2fm  (%u probes)",
                                gi.GetDimX(), gi.GetDimY(), gi.GetDimZ(), gi.GetSpacing(), gi.GetProbeCount());
                else
                {
                    ImGui::TextDisabled("No bake yet");
                    // Probe method just became active without data — request
                    // the bake automatically (parity with scene-load auto-bake).
                    // The method-change half of this moved to the event handler
                    // right after the combo. What is left is the per-frame "no
                    // bake yet" net, and a pending request needs no guard because
                    // setting it twice is idempotent.
                    if (!wasGIEnabled)
                        m_PendingGIProbeBake = true;
                }
            }
            else if (m_IndirectGIMethod == 2)
            {
                ImGui::SliderFloat("Probe Spacing", &m_DDGISpacing, 0.5f, 8.0f, "%.1f m");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Grid spacing used by the next \"Rebuild DDGI Grid\" (auto-grows to fit the probe cap).");

                if (ImGui::Button("Rebuild DDGI Grid"))
                    m_PendingDDGISetup = true;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Re-fit the probe grid to the scene bounds and clear the atlas.\n"

                                      "The next frame starts a fresh temporal history.");

                const char* ddgiViews[] = { "Off", "Irradiance", "Probe States", "Depth" };
                ImGui::SetNextItemWidth(180.f);
                ImGui::Combo("DDGI Debug View", &m_DDGIDebugView, ddgiViews, IM_ARRAYSIZE(ddgiViews));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Post-composite overlay, depth-tested against the GBuffer.\n"
                                      "Irradiance: one sphere per probe — each point samples the atlas at its own\n"
                                      "normal, so the directional structure of the field is visible.\n"
                                      "Probe States: sphere per probe; magenta marks an inactive (relocated out) probe.\n"
                                      "Depth: sphere per probe, shaded by the mean hit distance.");

                if (m_DDGIDebugView != 0)
                {
                    ImGui::SliderFloat("Debug Gain", &m_DDGIDebugScale, 0.1f, 10.0f, "%.2f");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Gain on the debug spheres. The irradiance view removes the stored gamma\n"
                                          "and divides by 2*PI before this gain is applied.");
                }

                const DDGI& ddgi = m_Renderer.GetDDGI();
                if (ddgi.IsValid())
                    ImGui::Text("Grid %ux%ux%u @ %.2fm  (%u probes, %u rays)",
                                ddgi.GetDimX(), ddgi.GetDimY(), ddgi.GetDimZ(), ddgi.GetSpacing(),
                                ddgi.GetProbeCount(), (uint32_t)DDGI::kRaysPerProbe);
                else
                {
                    ImGui::TextDisabled("No grid yet");
                    // DDGI just became active without a grid — fit one now
                    // (parity with the probe-method auto-bake above).
                    if (!wasGIEnabled)
                        m_PendingDDGISetup = true;
                }
            }
            else
            {
                ImGui::TextDisabled("SHaRC + reservoir debug: see sections below");
            }

            ImGui::Unindent();
        }
        else
        {
            m_FrameConstants.bakedGIMode            = 0u;
            m_FrameConstants.enableRestirGI = 0u;
            m_FrameConstants.ddgiMode      = 0u;
        }

        bool enableRestirDI = (m_FrameConstants.enableRestirDI != 0);
        if (ImGui::Checkbox("Enable ReSTIR DI (Local Lights)", &enableRestirDI))
        {
            m_FrameConstants.enableRestirDI = enableRestirDI ? 1 : 0;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Spatiotemporal reservoir resampling for local light direct illumination.\nWhen off, falls back to single-frame RIS (4 candidates).");

        if (enableRestirDI)
        {
            ImGui::Indent();
            const char* diDebugModes[] = {
                "Off",
                "Light Index",
                "M Count",
                "Weight (W)",
                "Visibility Age"
            };
            int diDebugMode = static_cast<int>(m_FrameConstants.restirDIDebugMode);
            ImGui::SetNextItemWidth(180.f);
            if (ImGui::Combo("DI Debug Vis", &diDebugMode, diDebugModes, IM_ARRAYSIZE(diDebugModes)))
            {
                m_FrameConstants.restirDIDebugMode = static_cast<uint32_t>(diDebugMode);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Visualize ReSTIR DI reservoir fields.\nOff = normal rendering.\nLight Index = selected light (normalized).\nM Count = history length.\nWeight (W) = unbiased RIS weight.\nVisibility Age = stale visibility counter.");
            ImGui::Unindent();
        }
        else
        {
            // Reset debug mode when DI is disabled
            m_FrameConstants.restirDIDebugMode = RESTIR_DI_DEBUG_OFF;
        }
        // NRD RELAX is available when either ReSTIR GI or ReSTIR DI is enabled
        const bool anyRestirActive = (m_FrameConstants.enableRestirGI != 0) || (m_FrameConstants.enableRestirDI != 0);
        if (anyRestirActive)
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
        }
        if (m_FrameConstants.enableRestirGI != 0)
        {
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
            (!m_UsePathTracer && m_FrameConstants.enableRestirGI);

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

    // Anti-Aliasing / Post-Processing mode
    ImGui::SeparatorText("Anti-Aliasing");
    {
        const char* aaModes[] = { "No AA", "Accumulation", "TAAU" };
        if (ImGui::Combo("AA Mode", &m_AntiAliasingMode, aaModes, IM_ARRAYSIZE(aaModes)))
        {
            m_TaaResetHistory = true;
            m_FrameConstants.frameIndex = 0;

            // Internal resolution must match output resolution when TAA is off
            // (no upscaler available). When TAA is re-enabled, restore the
            // upsampling-factor-derived resolution.
            const bool nowTaa = (m_AntiAliasingMode == AA_MODE_TAA);
            uint32_t newW = nowTaa ? std::max(320u, std::min((uint32_t)(m_OutputWidth  / m_TaaUpsamplingFactor), m_OutputWidth))  : m_OutputWidth;
            uint32_t newH = nowTaa ? std::max(180u, std::min((uint32_t)(m_OutputHeight / m_TaaUpsamplingFactor), m_OutputHeight)) : m_OutputHeight;
            if (newW != m_InternalWidth || newH != m_InternalHeight)
            {
                m_PendingResolutionChange = true;
                m_PendingInternalWidth  = newW;
                m_PendingInternalHeight = newH;
            }
        }
        if (ImGui::IsItemHovered())
        {
            const char* tooltips[] = {
                "No anti-aliasing. Single sample per pixel, no accumulation.",
                "Progressive accumulation. Path tracer converges over time when camera is still.",
                "Temporal Anti-Aliasing Upsample with sub-pixel jitter. TAA handles temporal accumulation."
            };
            ImGui::SetTooltip("%s", tooltips[m_AntiAliasingMode]);
        }

        // TAA-specific options (only enabled when AA mode is TAA)
        const bool isTaaMode = (m_AntiAliasingMode == AA_MODE_TAA);
        if (!isTaaMode)
            ImGui::BeginDisabled();

        float prevFactor = m_TaaUpsamplingFactor;
        if (ImGui::SliderFloat("Upsampling Factor", &m_TaaUpsamplingFactor, 1.0f, 4.0f, "%.1f"))
        {
            // Defer resource recreation to the start of the next frame
            // to avoid destroying GPU resources mid-frame (causes ImGui crash)
            uint32_t newW = std::max(320u, std::min((uint32_t)(m_OutputWidth / m_TaaUpsamplingFactor), m_OutputWidth));
            uint32_t newH = std::max(180u, std::min((uint32_t)(m_OutputHeight / m_TaaUpsamplingFactor), m_OutputHeight));
            if (newW != m_InternalWidth || newH != m_InternalHeight)
            {
                m_PendingResolutionChange = true;
                m_PendingInternalWidth = newW;
                m_PendingInternalHeight = newH;
            }
            m_TaaResetHistory = true;
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Internal render resolution: %ux%u (factor %.1fx)",
                m_InternalWidth, m_InternalHeight, m_TaaUpsamplingFactor);
        }

        ImGui::Text("Internal: %ux%u -> Output: %ux%u [Naive TSR]",
            m_InternalWidth, m_InternalHeight,
            m_OutputWidth, m_OutputHeight);

        if (!isTaaMode)
            ImGui::EndDisabled();

        // Debug info (collapsible, always available)
        if (ImGui::TreeNode("AA Debug"))
        {
            ImGui::Text("Mode: %s", aaModes[m_AntiAliasingMode]);
            ImGui::Text("Jitter: (%.4f, %.4f) px", m_FrameConstants.taaJitter.x, m_FrameConstants.taaJitter.y);
            if (isTaaMode)
            {
                uint32_t N = (uint32_t)std::max(8.0f, std::ceil(m_TaaUpsamplingFactor * m_TaaUpsamplingFactor));
                ImGui::Text("Halton sequence length: %u", N);
                ImGui::Text("TAA history index: %d", m_Renderer.IsTaaEnabled() ? 1 : 0);
            }
            ImGui::Text("Frame index: %u", m_FrameConstants.frameIndex);
            const char* accumState = "N/A";
            if (m_AntiAliasingMode == AA_MODE_NONE)          accumState = "DISABLED (No AA)";
            else if (m_AntiAliasingMode == AA_MODE_ACCUMULATION) accumState = "ENABLED (converging)";
            else if (m_AntiAliasingMode == AA_MODE_TAA)      accumState = "DISABLED (TAA active)";
            ImGui::Text("PT accumulation: %s", accumState);
            ImGui::TreePop();
        }
    }

    ImGui::Separator();

    // --- Profiler section ---
    ImGui::SeparatorText("Profiler");

    static bool profilerEnabled = true;
    if (ImGui::Checkbox("Enable Microprofile", &profilerEnabled))
    {
        MicroProfileSetEnableAllGroups(profilerEnabled);
    }

    if (!profilerEnabled)
    {
        ImGui::TextDisabled("Profiling disabled (zero overhead)");
    }

    // --- RenderDoc GPU capture section ---
    RenderRenderDocSection();

    ImGui::Separator();
    ImGui::Checkbox("Show ImGui Demo Window", &m_ShowDemoWindow);

    ImGui::End();

    // --- Separate MicroProfile Stats window ---
    if (profilerEnabled)
    {
        ImGui::SetNextWindowSize(ImVec2(320, 180), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("MicroProfile Stats", &profilerEnabled))
        {
            float gpuMs = MicroProfileGetTime("GPU", "FrameGpu");
            float cpuMs = MicroProfileGetTime("Render", "FrameCpu");
            float totalMs = gpuMs + cpuMs;

            ImGui::Text("GPU frame: %.2f ms", gpuMs);
            ImGui::Text("CPU frame: %.2f ms", cpuMs);
            ImGui::Text("Total:      %.2f ms", totalMs);
            float fps = totalMs > 0.0f ? 1000.0f / totalMs : 0.0f;
            ImGui::Text("FPS:        %.1f", fps);

            ImGui::Separator();

            float safeTotal = totalMs > 0.0f ? totalMs : 1.0f;
            ImGui::ProgressBar(gpuMs / safeTotal, ImVec2(-1, 0), "GPU");
            ImGui::ProgressBar(cpuMs / safeTotal, ImVec2(-1, 0), "CPU");

            ImGui::Separator();

            if (ImGui::Button("Dump Frame to HTML"))
            {
                MicroProfileDumpFileImmediately("microprofile_dump.html", 32, nullptr, 0, nullptr);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(saves to Bin/)");
        }
        ImGui::End();
    }

    if (m_ShowDemoWindow)
        ImGui::ShowDemoWindow(&m_ShowDemoWindow);
}

void Application::RenderRenderDocSection()
{
    ImGui::SeparatorText("RenderDoc");

    // RenderDoc is absent. A capture is impossible from this process, because
    // RenderDoc must enter the process before the D3D12 device exists. This
    // section only reports the state, and it offers no button.
    if (!m_RenderDoc.IsAvailable())
    {
        ImGui::TextDisabled("RenderDoc is not loaded in this process.");
        ImGui::TextWrapped("RenderDoc must enter the process before the D3D12 device exists. "
                           "Start the program with the argument \"-autoattach\", or start it "
                           "from RenderDoc.");
        return;
    }

    ImGui::Text("Connected: RenderDoc %s", m_RenderDoc.VersionString().c_str());

    // One-frame dump. The click is inside the frame, so this frame is captured.
    if (ImGui::Button("Capture Frame"))
    {
        m_RenderDoc.TriggerCapture();
        m_RenderDocStatus = "Dumped this frame.";
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Dumps the frame that contains this click to a .rdc file.\nRenderDoc writes the file when the frame ends.");

    ImGui::Text("Captures this session: %u", m_RenderDoc.CaptureCount());
    const std::string lastCapture = m_RenderDoc.LastCapturePath();
    if (!lastCapture.empty())
        ImGui::TextWrapped("Last file: %s", lastCapture.c_str());

    // The capture path is fixed at compile time. The line reports the value that
    // RenderDoc holds.
    ImGui::TextDisabled("Capture path: %s (fixed)", m_RenderDoc.CapturePathTemplate().c_str());

    if (ImGui::Button("Open Last Capture"))
    {
        m_RenderDocStatus = m_RenderDoc.LaunchReplayUI()
            ? "Replay UI opened."
            : "The replay UI did not start.";
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Opens the RenderDoc replay user interface for the last capture.");

    if (!m_RenderDocStatus.empty())
        ImGui::TextWrapped("%s", m_RenderDocStatus.c_str());
}