#pragma once

#include <d3d12.h>
#include <directx/d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>

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
};

struct GPUTexture : public GPUResource
{
    UINT srvIndex = UINT(-1);
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
