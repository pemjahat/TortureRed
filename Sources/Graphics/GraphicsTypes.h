#pragma once

#include "Shared/SharedTypes.h"

struct GPUResource
{
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;

    void Transition(ID3D12GraphicsCommandList* cmdList, D3D12_RESOURCE_STATES newState)
    {
        if (state != newState)
        {
            D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource.Get(), state, newState);
            cmdList->ResourceBarrier(1, &barrier);
            state = newState;
        }
    }
};

struct GPUBuffer : public GPUResource
{
    UINT64 size = 0;
    void* cpuPtr = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
    int srvIndex = -1;
    int uavIndex = -1;       // slot in shader-visible SRV heap (for GPU access)
    int cpuUavIndex = -1;    // slot in CPU-only UAV heap (for ClearUnorderedAccessViewUint)
};

struct GPUTexture : public GPUResource
{
    UINT srvIndex = UINT(-1);
    UINT uavIndex = UINT(-1);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = { 0 };
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = { 0 };
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};

struct GBuffer
{
    GPUTexture albedo;
    GPUTexture normal;
    GPUTexture material;
    GPUTexture depth;
};

// FrameConstants, LightConstants, BindlessIndices, etc. moved to Shared/SharedTypes.h

// debugName (optional): when provided, sets the ID3D12Object name (visible in
// RenderDoc's resource inspector / PIX).
bool CreateBuffer(GPUBuffer& buffer, UINT64 size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON, bool createSRV = false, bool createUAV = false, const char* debugName = nullptr);
bool CreateStructuredBuffer(GPUBuffer& buffer, UINT64 elementSize, UINT64 elementCount, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, const char* debugName = nullptr);
bool CreateTexture(GPUTexture& texture, UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, const FLOAT* clearColor = nullptr, UINT mipLevels = 1, UINT arraySize = 1, const char* debugName = nullptr, bool isCubemap = false);
bool CreateTexture3D(GPUTexture& texture, UINT width, UINT height, UINT depth, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, UINT mipLevels = 1, const char* debugName = nullptr);