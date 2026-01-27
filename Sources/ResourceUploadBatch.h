#pragma once

#include <d3d12.h>
#include <wrl.h>
#include <vector>
#include "GraphicsTypes.h"

class Renderer;

class ResourceUploadBatch
{
public:
    ResourceUploadBatch(Renderer* renderer);
    ~ResourceUploadBatch();

    void Begin();
    void Upload(GPUBuffer& dest, const void* data, UINT64 size);
    void Transition(GPUResource& resource, D3D12_RESOURCE_STATES newState);
    
    void End();

private:
    Renderer* m_Renderer;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_CommandList;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_Allocator;
    std::vector<GPUBuffer> m_StagingBuffers;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_StagingResources;
};
