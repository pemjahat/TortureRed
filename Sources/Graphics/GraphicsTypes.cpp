#include "GraphicsTypes.h"
#include "GraphicsHelper.h"
#include "Core/Utility.h"
#include "d3dx12.h"

bool CreateBuffer(GPUBuffer& buffer, UINT64 size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, bool createSRV, bool createUAV, const char* debugName)
{
    auto& ctx = GraphicsHelper::GetContext();
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(heapType);
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size);

    if (initialState & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE))
    {
        desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    CHECK_HR(ctx.device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&buffer.resource)), "CreateCommittedResource for Buffer failed");

    buffer.size = size;
    buffer.state = initialState;
    buffer.gpuAddress = buffer.resource->GetGPUVirtualAddress();

    GraphicsHelper::SetObjectName(buffer.resource.Get(), debugName);

    if (heapType == D3D12_HEAP_TYPE_UPLOAD)
    {
        buffer.resource->Map(0, nullptr, &buffer.cpuPtr);
    }

    if (createSRV)
    {
        // Reuse existing descriptor slot if available, otherwise allocate a new one
        if (buffer.srvIndex < 0)
            buffer.srvIndex = (int)GraphicsHelper::AllocateSRV();
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GraphicsHelper::GetSRVCPUHandle((UINT)buffer.srvIndex);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = (UINT)(size / 4);
        srvDesc.Buffer.StructureByteStride = 0;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;

        ctx.device->CreateShaderResourceView(buffer.resource.Get(), &srvDesc, srvHandle);
    }

    if (createUAV && (initialState & D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
    {
        if (buffer.uavIndex < 0)
            buffer.uavIndex = (int)GraphicsHelper::AllocateSRV();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = GraphicsHelper::GetSRVCPUHandle((UINT)buffer.uavIndex);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = (UINT)(size / 4);
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

        ctx.device->CreateUnorderedAccessView(buffer.resource.Get(), nullptr, &uavDesc, uavHandle);

        // Also create in the CPU-only (non-shader-visible) heap for ClearUnorderedAccessViewUint.
        // D3D12 requires ViewCPUHandle to come from a non-shader-visible heap.
        if (buffer.cpuUavIndex < 0)
            buffer.cpuUavIndex = (int)GraphicsHelper::AllocateCpuUAV();
        D3D12_CPU_DESCRIPTOR_HANDLE cpuUavHandle = GraphicsHelper::GetCpuUAVHandle((UINT)buffer.cpuUavIndex);
        ctx.device->CreateUnorderedAccessView(buffer.resource.Get(), nullptr, &uavDesc, cpuUavHandle);
    }

    return true;
}

bool CreateStructuredBuffer(GPUBuffer& buffer, UINT64 elementSize, UINT64 elementCount, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, const char* debugName)
{
    UINT64 size = elementSize * elementCount;
    if (!CreateBuffer(buffer, size, heapType, initialState, false, false, debugName)) return false;

    auto& ctx = GraphicsHelper::GetContext();
    // Reuse existing descriptor slot if available, otherwise allocate a new one
    if (buffer.srvIndex < 0)
        buffer.srvIndex = (int)GraphicsHelper::AllocateSRV();
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GraphicsHelper::GetSRVCPUHandle((UINT)buffer.srvIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = (UINT)elementCount;
    srvDesc.Buffer.StructureByteStride = (UINT)elementSize;
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    ctx.device->CreateShaderResourceView(buffer.resource.Get(), &srvDesc, srvHandle);

    if (initialState & D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    {
        if (buffer.uavIndex < 0)
            buffer.uavIndex = (int)GraphicsHelper::AllocateSRV();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = GraphicsHelper::GetSRVCPUHandle((UINT)buffer.uavIndex);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = (UINT)elementCount;
        uavDesc.Buffer.StructureByteStride = (UINT)elementSize;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

        ctx.device->CreateUnorderedAccessView(buffer.resource.Get(), nullptr, &uavDesc, uavHandle);
    }
    return true;
}

bool CreateTexture(GPUTexture& texture, UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, const FLOAT* clearColor, UINT mipLevels, UINT arraySize, const char* debugName)
{
    auto& ctx = GraphicsHelper::GetContext();
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = static_cast<UINT16>(arraySize);
    desc.MipLevels = static_cast<UINT16>(mipLevels);
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;

    D3D12_CLEAR_VALUE clearVal = {};
    clearVal.Format = format;
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL && format == DXGI_FORMAT_R32_TYPELESS)
    {
        clearVal.Format = DXGI_FORMAT_D32_FLOAT;
    }
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
    {
        if (clearColor) memcpy(clearVal.Color, clearColor, sizeof(float) * 4);
    }
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        clearVal.DepthStencil.Depth = 1.0f;
    }

    CHECK_HR(ctx.device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        (flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) ? &clearVal : nullptr,
        IID_PPV_ARGS(&texture.resource)), "CreateCommittedResource for Texture failed");

    texture.state = initialState;
    texture.format = format;

    GraphicsHelper::SetObjectName(texture.resource.Get(), debugName);

    // Create SRV (reuse existing descriptor slot if available)
    if (!(flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
    {
        if (texture.srvIndex == UINT(-1))
            texture.srvIndex = GraphicsHelper::AllocateSRV();
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GraphicsHelper::GetSRVCPUHandle(texture.srvIndex);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS)
            srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
        else
            srvDesc.Format = format;
        
        if (arraySize > 1)
        {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srvDesc.Texture2DArray.MipLevels = mipLevels;
            srvDesc.Texture2DArray.ArraySize = arraySize;
            srvDesc.Texture2DArray.FirstArraySlice = 0;
            srvDesc.Texture2DArray.MostDetailedMip = 0;
        }
        else
        {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MipLevels = mipLevels;
        }

        ctx.device->CreateShaderResourceView(texture.resource.Get(), &srvDesc, srvHandle);
    }

    // Create UAV (reuse existing descriptor slot if available)
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        if (texture.uavIndex == UINT(-1))
            texture.uavIndex = GraphicsHelper::AllocateSRV();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = GraphicsHelper::GetSRVCPUHandle(texture.uavIndex);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = (format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS) ? DXGI_FORMAT_R32_FLOAT : format;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = 0;
        uavDesc.Texture2D.PlaneSlice = 0;

        ctx.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr, &uavDesc, uavHandle);
    }

    // Create RTV or DSV (reuse existing descriptor slot if available)
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
    {
        if (texture.rtvHandle.ptr == 0)
            texture.rtvHandle = GraphicsHelper::GetRTVCPUHandle(GraphicsHelper::AllocateRTV());
        ctx.device->CreateRenderTargetView(texture.resource.Get(), nullptr, texture.rtvHandle);
    }
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        if (texture.dsvHandle.ptr == 0)
            texture.dsvHandle = GraphicsHelper::GetDSVCPUHandle(GraphicsHelper::AllocateDSV());
        
        D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
        dsvDesc.Format = (format == DXGI_FORMAT_R32_TYPELESS) ? DXGI_FORMAT_D32_FLOAT : format;
        dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        
        ctx.device->CreateDepthStencilView(texture.resource.Get(), &dsvDesc, texture.dsvHandle);
    }

    return true;
}

bool CreateTexture3D(GPUTexture& texture, UINT width, UINT height, UINT depth, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState, UINT mipLevels, const char* debugName)
{
    auto& ctx = GraphicsHelper::GetContext();
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = static_cast<UINT16>(depth);
    desc.MipLevels = static_cast<UINT16>(mipLevels);
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;

    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    HRESULT hr = ctx.device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&texture.resource)
    );

    if (FAILED(hr)) return false;

    texture.state = initialState;
    texture.format = format;

    GraphicsHelper::SetObjectName(texture.resource.Get(), debugName);

    // Create SRV if not a depth stencil
    if (!(flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        srvDesc.Texture3D.MipLevels = mipLevels;
        srvDesc.Texture3D.MostDetailedMip = 0;

        texture.srvIndex = GraphicsHelper::AllocateSRV();
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = GraphicsHelper::GetSRVCPUHandle(texture.srvIndex);
        ctx.device->CreateShaderResourceView(texture.resource.Get(), &srvDesc, srvHandle);
    }

    // Create UAV if requested
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = format;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uavDesc.Texture3D.MipSlice = 0;
        uavDesc.Texture3D.FirstWSlice = 0;
        uavDesc.Texture3D.WSize = depth;

        texture.uavIndex = GraphicsHelper::AllocateSRV();
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = GraphicsHelper::GetSRVCPUHandle(texture.uavIndex);
        ctx.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr, &uavDesc, uavHandle);
    }

    return true;
}
