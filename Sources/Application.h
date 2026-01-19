#pragma once

#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_sdl2.h>

#include "Camera.h"
#include "Model.h"
#include "Renderer.h"

// Error checking macro that always asserts in both debug and release
#define CHECK_HR(hr, msg) \
    if (FAILED(hr)) { \
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "DirectX Error: %s (HRESULT: 0x%08X)", msg, hr); \
        assert(false); \
    }

#define CHECK_BOOL(condition, msg) \
    if (!(condition)) { \
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Application Error: %s", msg); \
        assert(false); \
    }


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
    SDL_Window* m_Window;

    // Core systems
    Renderer m_Renderer;
    Model m_Model;
    Camera m_Camera;

    // ImGui
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ImGuiDescriptorHeap;

    // Input state for camera
    bool m_RightMouseButtonHeld;
    int m_LastMouseX, m_LastMouseY;

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};