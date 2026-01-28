#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define STRICT                          // Use strict declarations for Windows types

#define NOMINMAX // Prevent min/max macros

#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_sdl2.h>

#include "Utility.h"
#include "Camera.h"
#include "Model.h"
#include "Renderer.h"


class Application
{
public:
    Application();
    ~Application();

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
    bool m_EnableDepthPrePass = false;
    bool m_DebugShadowMap = false;
    bool m_UsePathTracer = false;
    float m_SunIntensity = 1.0f;
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
    LightConstants m_MainLight;

    // ImGui
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ImGuiDescriptorHeap;

    // Input state for camera
    bool m_RightMouseButtonHeld;
    int m_LastMouseX, m_LastMouseY;

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};