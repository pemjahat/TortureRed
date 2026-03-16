#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <functional>
#include <string>
#include <vector>
#include "d3dx12.h"

struct GPUResource;
struct GPUBuffer;
struct GPUTexture;

struct GraphicsContext {
    ID3D12Device* device = nullptr;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap;
    
    UINT srvHeapIndex = 0;
    UINT rtvHeapIndex = 0;
    UINT dsvHeapIndex = 0;

    UINT srvDescriptorSize = 0;
    UINT rtvDescriptorSize = 0;
    UINT dsvDescriptorSize = 0;
};

class GraphicsHelper {
public:
    static void Initialize(ID3D12Device* device);
    static void Shutdown();
    
    static UINT AllocateSRV();
    static UINT AllocateRTV();
    static UINT AllocateDSV();

    static ID3D12DescriptorHeap* GetSRVHeap() { return s_Context.srvHeap.Get(); }
    static ID3D12DescriptorHeap** GetSRVHeapAddress() { return s_Context.srvHeap.GetAddressOf(); }
    static ID3D12DescriptorHeap* GetRTVHeap() { return s_Context.rtvHeap.Get(); }
    static ID3D12DescriptorHeap* GetDSVHeap() { return s_Context.dsvHeap.Get(); }

    static std::vector<char> CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target);
    static std::vector<char> CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target,
                                           const std::vector<std::pair<std::wstring, std::wstring>>& defines);
    static void InvalidateShaderCache();

    static D3D12_CPU_DESCRIPTOR_HANDLE GetSRVCPUHandle(UINT index);
    static D3D12_GPU_DESCRIPTOR_HANDLE GetSRVGPUHandle(UINT index);
    static D3D12_CPU_DESCRIPTOR_HANDLE GetRTVCPUHandle(UINT index);
    static D3D12_CPU_DESCRIPTOR_HANDLE GetDSVCPUHandle(UINT index);

    static void TransitionResource(ID3D12GraphicsCommandList* cmdList, GPUResource& resource, D3D12_RESOURCE_STATES newState);
    static void TransitionResource(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, D3D12_RESOURCE_STATES& currentState, D3D12_RESOURCE_STATES newState);

    static GraphicsContext& GetContext() { return s_Context; }

private:
    static GraphicsContext s_Context;
};
