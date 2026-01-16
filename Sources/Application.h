#pragma once

#include <SDL.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <memory>
#include <vector>
#include <string>
#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_sdl2.h>
#include <iostream>

// Forward declarations for cgltf
struct cgltf_data;

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

// GLTF Model structures
struct GLTFVertex
{
    float position[3];
    float normal[3];
    float texCoord[2];
};

struct GLTFMaterial
{
    float baseColorFactor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
};

struct GLTFMesh
{
    std::vector<GLTFVertex> vertices;
    std::vector<uint32_t> indices;
    GLTFMaterial material;
    Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    D3D12_INDEX_BUFFER_VIEW indexBufferView;
};

struct GLTFModel
{
    std::vector<GLTFMesh> meshes;
    cgltf_data* data = nullptr;
};

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
    void InitializeImGui();
    void RenderImGui();
    bool LoadGLTFModel(const std::string& filepath);
    void CreateGLTFResources();
    void RenderGLTFModel();
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

    // ImGui
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ImGuiDescriptorHeap;
    float m_BackgroundColor[3] = { 0.098f, 0.098f, 0.439f }; // Default: Dark blue

    // GLTF Model
    GLTFModel m_GltfModel;

    // Depth Buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> m_DepthStencilBuffer;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DSVHeap;

    // Constant Buffer for view-projection matrix
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ConstantBuffer;
    void* m_ConstantBufferData = nullptr;

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};