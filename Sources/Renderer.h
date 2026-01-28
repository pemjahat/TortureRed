#pragma once

#include <d3d12.h>
#include <directx/d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <DirectXMath.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "GraphicsTypes.h"

// Forward declarations to avoid circular dependencies
struct GLTFVertex;
struct GLTFPrimitive;

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

    void BuildAccelerationStructures(class Model* model);
    void DispatchRays(class Model* model, const FrameConstants& frame, const LightConstants& light);
    void CopyTextureToBackBuffer(const GPUTexture& texture);

    // Resource creation
    void CreateRootSignature();
    void CreatePipelineState();
    void CreateRayTracingPipeline();
    void CreateShaderBindingTable();

    // GBuffer management
    void CreateGBuffer();

    // Descriptor management
    UINT AllocateDescriptor();

    // Resource helpers
    bool CreateBuffer(GPUBuffer& buffer, UINT64 size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON, bool createSRV = false);
    bool CreateStructuredBuffer(GPUBuffer& buffer, UINT64 elementSize, UINT64 elementCount, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState);
    bool CreateTexture(GPUTexture& texture, UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, const FLOAT* clearColor = nullptr, UINT mipLevels = 1);

    void TransitionResource(GPUTexture& texture, D3D12_RESOURCE_STATES newState);
    void TransitionResource(GPUBuffer& buffer, D3D12_RESOURCE_STATES newState);
    void TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES& currentState, D3D12_RESOURCE_STATES newState);
    void TransitionBackBuffer(D3D12_RESOURCE_STATES newState);

    // Shader compilation
    std::vector<char> LoadShader(const std::string& filename);
    std::vector<char> CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target);

    // Constant buffer management
    void UpdateFrameCB(const FrameConstants& frameConstants);
    void UpdateLightCB(const LightConstants& lightConstants);

    // Getters
    ID3D12Device* GetDevice() const { return m_Device.Get(); }
    ID3D12GraphicsCommandList* GetCommandList() const { return m_CommandList.Get(); }
    ID3D12CommandQueue* GetCommandQueue() const { return m_CommandQueue.Get(); }
    ID3D12CommandAllocator* GetCommandAllocator() const { return m_CommandAllocator.Get(); }
    ID3D12RootSignature* GetRootSignature() const { return m_RootSignature.Get(); }
    ID3D12CommandSignature* GetCommandSignature() const { return m_CommandSignature.Get(); }
    ID3D12PipelineState* GetPipelineState() const { return m_PipelineState.Get(); }
    ID3D12PipelineState* GetDepthPrePassPSO() const { return m_DepthPrePassPSO.Get(); }
    ID3D12PipelineState* GetGBufferPSO() const { return m_GBufferPSO.Get(); }
    ID3D12PipelineState* GetGBufferWritePSO() const { return m_GBufferWritePSO.Get(); }
    ID3D12PipelineState* GetLightingPSO() const { return m_LightingPSO.Get(); }
    ID3D12PipelineState* GetDebugPSO() const { return m_DebugPSO.Get(); }
    ID3D12PipelineState* GetShadowPSO() const { return m_ShadowPSO.Get(); }
    
    // Ray Tracing Getters
    bool IsRayTracingSupported() const { return m_RayTracingSupported; }

    ID3D12DescriptorHeap* GetSRVHeap() const { return m_SRVHeap.Get(); }

    D3D12_GPU_VIRTUAL_ADDRESS GetFrameGPUAddress() const { return m_FrameCB.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetLightGPUAddress() const { return m_LightCB.gpuAddress; }

    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUDescriptorHandle(UINT index) const {
        D3D12_GPU_DESCRIPTOR_HANDLE handle = m_SRVHeap->GetGPUDescriptorHandleForHeapStart();
        handle.ptr += (UINT64)index * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return handle;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GetCPUDescriptorHandle(UINT index) const {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = m_SRVHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += (UINT64)index * m_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return handle;
    }

    void ExecuteCommandList();

    // Pass management
    D3D12_CPU_DESCRIPTOR_HANDLE GetCurrentBackBufferRTV() const;
    ID3D12Resource* GetCurrentBackBuffer() const;

    // Background color
    float m_BackgroundColor[3] = { 0.098f, 0.098f, 0.439f }; // Default: Dark blue

    // GBuffer access
    GBuffer& GetGBuffer() { return m_GBuffer; }
    GPUTexture& GetShadowMap() { return m_ShadowMap; }
    GPUTexture& GetPathTracerOutput() { return m_PathTracerOutput; }

private:
    void GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter);
    void WaitForPreviousFrame();

    // DirectX 12 objects
    Microsoft::WRL::ComPtr<ID3D12Device> m_Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CommandQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain4> m_SwapChain;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_RTVHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_RenderTargets[2];
    D3D12_RESOURCE_STATES m_BackBufferStates[2] = { D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_PRESENT };
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_CommandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_CommandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_Fence;

    // Pipeline States
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PipelineState;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DepthPrePassPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_GBufferPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_GBufferWritePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_LightingPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DebugPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ShadowPSO;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_RootSignature;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_CommandSignature;

    // Ray Tracing
    bool m_RayTracingSupported = false;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PathTracerPSO;
    std::unordered_map<const struct GLTFPrimitive*, GPUBuffer> m_BlasPool;
    GPUBuffer m_TLAS;
    GPUTexture m_PathTracerOutput;
    GPUTexture m_AccumulationBuffer;
    GPUBuffer m_ReservoirBuffer[2]; // ReSTIR Reservoirs (Current and Previous)
    int m_CurrentReservoirIndex = 0;

    // Descriptor Heaps
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DSVHeap;

    // SRV Heap for textures (Global Unified Heap)
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_SRVHeap;
    UINT m_SrvHeapIndex = 0;

    // GBuffer resources
    GBuffer m_GBuffer;
    GPUTexture m_ShadowMap;

    // Constant Buffers
    GPUBuffer m_FrameCB;
    GPUBuffer m_LightCB;

    // Synchronization
    UINT m_FrameIndex;
    HANDLE m_FenceEvent;
    UINT64 m_FenceValue;

    // Prevent copying
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
};