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
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap;      // shader-visible CBV/SRV/UAV
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap;
    // CPU-only (non-shader-visible) UAV heap required by ClearUnorderedAccessViewUint.
    // D3D12 spec: the ViewCPUHandle argument must NOT come from a shader-visible heap.
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> cpuUavHeap;

    UINT srvHeapIndex = 0;
    UINT rtvHeapIndex = 0;
    UINT dsvHeapIndex = 0;
    UINT cpuUavHeapIndex = 0;

    UINT srvDescriptorSize = 0;
    UINT rtvDescriptorSize = 0;
    UINT dsvDescriptorSize = 0;
};

// -----------------------------------------------------------------------------
// ScopedGpuEvent — RAII wrapper around ID3D12GraphicsCommandList::BeginEvent/
// EndEvent so RenderDoc (and PIX) shows named regions in the capture timeline
// (e.g. "GBuffer Pass", "Depth Pre-Pass", "Lighting") instead of the generic
// "Compute Pass #N" / "Colour Pass #N" RenderDoc falls back to when no event
// markers are present.
//
// This intentionally does NOT depend on WinPixEventRuntime: it encodes the
// event name as a plain null-terminated wide string with Metadata=0, which is
// the same fallback format RenderDoc's D3D12 capture layer recognizes without
// needing the PIX blob encoding. PIX for Windows itself does not decode this
// format, but this project targets RenderDoc as its capture/debug tool.
// -----------------------------------------------------------------------------
class ScopedGpuEvent
{
public:
    ScopedGpuEvent(ID3D12GraphicsCommandList* cmdList, const wchar_t* name)
        : m_CmdList(cmdList)
    {
        if (!m_CmdList) return;
        const size_t byteSize = (wcslen(name) + 1) * sizeof(wchar_t);
        m_CmdList->BeginEvent(0, name, static_cast<UINT>(byteSize));
    }

    ~ScopedGpuEvent()
    {
        if (m_CmdList) m_CmdList->EndEvent();
    }

    ScopedGpuEvent(const ScopedGpuEvent&) = delete;
    ScopedGpuEvent& operator=(const ScopedGpuEvent&) = delete;

private:
    ID3D12GraphicsCommandList* m_CmdList = nullptr;
};

// GPU_MARKER(cmdList, "Name") — scoped for the rest of the enclosing {} block.
#define GPU_MARKER_CONCAT_INNER(a, b) a##b
#define GPU_MARKER_CONCAT(a, b) GPU_MARKER_CONCAT_INNER(a, b)
#define GPU_MARKER(cmdList, name) ScopedGpuEvent GPU_MARKER_CONCAT(_gpuMarker_, __LINE__)((cmdList), (name))

class GraphicsHelper {
public:
    static void Initialize(ID3D12Device* device);
    static void Shutdown();
    
    static UINT AllocateSRV();
    static UINT AllocateRTV();
    static UINT AllocateDSV();
    // Allocates a slot in the CPU-only UAV heap (for ClearUnorderedAccessViewUint)
    static UINT AllocateCpuUAV();

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
    // CPU-only UAV handle (non-shader-visible) — use as ViewCPUHandle in ClearUnorderedAccessViewUint
    static D3D12_CPU_DESCRIPTOR_HANDLE GetCpuUAVHandle(UINT index);

    static void TransitionResource(ID3D12GraphicsCommandList* cmdList, GPUResource& resource, D3D12_RESOURCE_STATES newState);
    static void TransitionResource(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, D3D12_RESOURCE_STATES& currentState, D3D12_RESOURCE_STATES newState);

    static GraphicsContext& GetContext() { return s_Context; }

    // ---------------------------------------------------------------------
    // Debug naming
    // ---------------------------------------------------------------------
    // Sets the ID3D12Object debug name (visible in RenderDoc's resource
    // inspector / PIX object browser) — thin wrapper around SetName().
    static void SetObjectName(ID3D12Resource* resource, const char* name);
private:
    static GraphicsContext s_Context;
};
