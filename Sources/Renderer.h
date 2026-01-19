#pragma once

#include <d3d12.h>
#include <directx/d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <DirectXMath.h>
#include <vector>
#include <string>

#include "Model.h"

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;

class Renderer
{
public:
    Renderer();
    ~Renderer();

    bool Initialize(HWND hwnd);
    void Shutdown();
    void Resize(uint32_t width, uint32_t height);

    // Rendering functions
    void BeginFrame();
    void EndFrame();
    void Present();

    // Resource creation
    void CreateRootSignature();
    void CreatePipelineState();

    // Shader compilation
    std::vector<char> LoadShader(const std::string& filename);
    std::vector<char> CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target);

    // Constant buffer management
    void UpdateViewProjectionMatrix(const DirectX::XMMATRIX& viewMatrix);

    // Getters
    ID3D12Device* GetDevice() const { return m_Device.Get(); }
    ID3D12GraphicsCommandList* GetCommandList() const { return m_CommandList.Get(); }
    ID3D12CommandQueue* GetCommandQueue() const { return m_CommandQueue.Get(); }
    ID3D12CommandAllocator* GetCommandAllocator() const { return m_CommandAllocator.Get(); }
    ID3D12RootSignature* GetRootSignature() const { return m_RootSignature.Get(); }
    ID3D12PipelineState* GetPipelineState() const { return m_PipelineState.Get(); }
    ID3D12DescriptorHeap* GetSRVHeap() const { return m_SRVHeap.Get(); }

    void ExecuteCommandList();

    // Background color
    float m_BackgroundColor[3] = { 0.098f, 0.098f, 0.439f }; // Default: Dark blue

private:
    void GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter);
    void WaitForPreviousFrame();

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

    // Depth Buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> m_DepthStencilBuffer;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DSVHeap;

    // SRV Heap for textures
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_SRVHeap;

    // Constant Buffer for view-projection matrix
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ConstantBuffer;
    void* m_ConstantBufferData = nullptr;

    // Synchronization
    UINT m_FrameIndex;
    HANDLE m_FenceEvent;
    UINT64 m_FenceValue;

    // Prevent copying
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
};