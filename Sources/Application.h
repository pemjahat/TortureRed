#pragma once

#include "Utility.h"
#include "Camera.h"
#include "Model.h"
#include "Renderer.h"


#include "Scene.h"

class Application
{
public:
    Application();
    ~Application();
    
    // ...
    
    Scene m_Scene;

    void Run();

private:
    void Initialize();
    void Shutdown();
    void ProcessEvents();
    void Update(float deltaTime);
    void Render();

    void InitializeImGui();
    void RenderImGui();

    bool m_IsRunning;
    bool m_EnableDepthPrePass    = false;
    bool m_DebugShadowMap        = false;
    bool m_UsePathTracer          = false;
    bool m_ShowDemoWindow         = false;
    bool m_UseMeshlet             = false;  // Toggle meshlet vs vertex-buffer rendering
    float m_Exposure = 1.0f;
    SDL_Window* m_Window;

    // Core systems
    Renderer m_Renderer;
    Model m_Model;
    Camera m_Camera;
    DirectX::XMMATRIX m_ViewProj;
    DirectX::XMMATRIX m_LastViewMatrix;
    DirectX::XMFLOAT4X4 m_LastViewProj;
    DirectX::XMFLOAT4X4 m_LastViewInverse;
    DirectX::XMFLOAT4 m_LastCameraPos;
    FrameConstants m_FrameConstants;

    int m_SelectedLightIndex = 0;

    // ImGui
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ImGuiDescriptorHeap;

    // Input state for camera
    bool m_RightMouseButtonHeld;
    int m_LastMouseX, m_LastMouseY;

    // Path visualization state
    bool m_PathVizJustClicked = false;
    bool m_PathVizEverCaptured = false;

    // Anti-aliasing / post-processing mode (AA_MODE_NONE, AA_MODE_ACCUMULATION, AA_MODE_TAA)
    int   m_AntiAliasingMode = AA_MODE_TAA;
    float m_TaaUpsamplingFactor = 1.5f;
    bool  m_TaaResetHistory = false;   // Set true when mode/factor changes
    uint32_t m_TaaFrameCounter = 0;   // Monotonically increasing counter for TAA jitter sequence
    bool  m_PendingResolutionChange = false; // Deferred resolution change
    uint32_t m_PendingInternalWidth = 0;
    uint32_t m_PendingInternalHeight = 0;
    uint32_t m_InternalWidth = 0;
    uint32_t m_InternalHeight = 0;
    uint32_t m_OutputWidth = WINDOW_WIDTH;
    uint32_t m_OutputHeight = WINDOW_HEIGHT;

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};