#pragma once

#include <SDL.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

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

    void InitializeDirectX();
    void GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter);
    void CreateRootSignature();
    void CreatePipelineState();
    void CreateVertexBuffer();
    std::vector<char> LoadShader(const std::string& filename);
    std::vector<char> CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target);
    void WaitForPreviousFrame();

    bool m_IsRunning;
    SDL_Window* m_Window;

    // DirectX 12 objects
    Microsoft::WRL::ComPtr<ID3D12Device> m_Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CommandQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain4> m_SwapChain;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_RTVHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_RenderTargets[2];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_CommandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_CommandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_Fence;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PipelineState;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_RootSignature;

    // Synchronization
    UINT m_FrameIndex;
    HANDLE m_FenceEvent;
    UINT64 m_FenceValue;

    // Vertex buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> m_VertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};