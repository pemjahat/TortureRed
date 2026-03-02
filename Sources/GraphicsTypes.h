#pragma once

struct Reservoir
{
    DirectX::XMFLOAT3 hitPos;
    DirectX::XMFLOAT3 hitNormal;
    DirectX::XMFLOAT3 radiance;
    float w_sum;
    float M;
    float W;
};

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
    int uavIndex = -1;
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
    DirectX::XMFLOAT4X4 viewProjPrevious;
    DirectX::XMFLOAT4X4 viewInversePrevious;
    DirectX::XMFLOAT4 cameraPosition;
    DirectX::XMFLOAT4 prevCameraPosition;
    uint32_t frameIndex;
    int32_t albedoIndex;
    int32_t normalIndex;
    int32_t materialIndex;
    int32_t depthIndex;
    float exposure;
    uint32_t enableRestir;
    uint32_t enableAvoidCaustics;
    uint32_t enableIndirectSpecular;
    uint32_t enableRasterIndirectGI;
    uint32_t useRTXDI; // Kept but maybe unused
    uint32_t numLights;
    uint32_t lightSamplingMode; // 0=uniform, 1=importance sampling
    uint32_t lightLUTBufferIndex; // Index into the light LUT buffer for O(1) importance sampling
    uint32_t debugIrCache;
    uint32_t screenWidth;
    uint32_t screenHeight;
};

struct LightConstants
{
    DirectX::XMFLOAT4X4 viewProj;
    DirectX::XMFLOAT4 position;
    DirectX::XMFLOAT4 color;
    DirectX::XMFLOAT4 direction;
    float intensity;
    float selectionPDF;
    uint32_t padding[2];
};

struct BindlessIndices 
{
    uint32_t InputIdx0; // raster rister require 2 input: current IrCache and previous Restir reservoir
    uint32_t InputIdx1;
    uint32_t OutputIdx0; // Restir output accumulation and path trace output    
    uint32_t OutputIdx1;
};