#include "ResourceUploadBatch.h"
#include "Renderer.h"
#include "Utility.h"
#include <iostream>

ResourceUploadBatch::ResourceUploadBatch(Renderer* renderer)
    : m_Renderer(renderer)
{
    auto device = m_Renderer->GetDevice();
    CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_Allocator)), "Failed to create upload allocator");
    CHECK_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_Allocator.Get(), nullptr, IID_PPV_ARGS(&m_CommandList)), "Failed to create upload command list");
    m_CommandList->Close();
}

ResourceUploadBatch::~ResourceUploadBatch()
{
}

void ResourceUploadBatch::Begin()
{
    CHECK_HR(m_Allocator->Reset(), "Failed to reset upload allocator");
    CHECK_HR(m_CommandList->Reset(m_Allocator.Get(), nullptr), "Failed to reset upload command list");
    m_StagingBuffers.clear();
}

void ResourceUploadBatch::Upload(GPUBuffer& dest, const void* data, UINT64 size)
{
    GPUBuffer staging;
    if (!m_Renderer->CreateBuffer(staging, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
    {
        std::cerr << "ResourceUploadBatch: Failed to create staging buffer" << std::endl;
        return;
    }
    
    if (data)
    {
        memcpy(staging.cpuPtr, data, size);
    }
    
    dest.Transition(m_CommandList.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    m_CommandList->CopyBufferRegion(dest.resource.Get(), 0, staging.resource.Get(), 0, size);
    
    m_StagingBuffers.push_back(staging);
}

void ResourceUploadBatch::Transition(GPUResource& resource, D3D12_RESOURCE_STATES newState)
{
    resource.Transition(m_CommandList.Get(), newState);
}

void ResourceUploadBatch::End()
{
    CHECK_HR(m_CommandList->Close(), "Failed to close upload command list");
    
    ID3D12CommandList* ppCommandLists[] = { m_CommandList.Get() };
    m_Renderer->GetCommandQueue()->ExecuteCommandLists(1, ppCommandLists);
    
    // Quick synchronization (make sure CopyBuffer + Resource is done before releasing staging buffers)
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    CHECK_HR(m_Renderer->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create fence failed");
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    m_Renderer->GetCommandQueue()->Signal(fence.Get(), 1);
    fence->SetEventOnCompletion(1, event);
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    
    m_StagingBuffers.clear();
}
