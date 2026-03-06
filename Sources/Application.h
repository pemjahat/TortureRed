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
    bool m_ShowProbeSpheresDebug  = false;
    bool m_FreezeIrCacheCamera    = false;
    DirectX::XMFLOAT4 m_FrozenIrCacheCameraPos = { 0.f, 0.f, 0.f, 1.f };
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

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};