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
    int srvIndex = -1;
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

struct FrameConstants
{
    DirectX::XMFLOAT4X4 viewProj;
    DirectX::XMFLOAT4X4 viewInverse;
    DirectX::XMFLOAT4X4 projectionInverse;
    DirectX::XMFLOAT4 cameraPosition;
    uint32_t frameIndex;
    int32_t albedoIndex;
    int32_t normalIndex;
    int32_t materialIndex;
    int32_t depthIndex;
    int32_t shadowMapIndex;
    uint32_t padding[2];
};

struct LightConstants
{
    DirectX::XMFLOAT4X4 viewProj;
    DirectX::XMFLOAT4 position;
    DirectX::XMFLOAT4 color;
    DirectX::XMFLOAT4 direction;
};

struct PrimitiveData
{
    int vertexBufferIndex;
    int indexBufferIndex;
    uint32_t materialIndex;
    uint32_t padding;
};
