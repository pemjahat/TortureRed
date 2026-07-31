#include "pch.h"
#include <fstream>
#include <iostream>
#include <dxcapi.h>
#include "GraphicsHelper.h"
#include "GraphicsTypes.h"
#include "d3dx12.h"
#include "Core/Utility.h"

GraphicsContext GraphicsHelper::s_Context;

void GraphicsHelper::Initialize(ID3D12Device* device) {
    s_Context.device = device;

    // Create RTV heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
        rtvHeapDesc.NumDescriptors = 16;
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        CHECK_HR(device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&s_Context.rtvHeap)), "CreateDescriptorHeap for RTV failed");
        s_Context.rtvHeap->SetName(L"Heap_RTV");
    }

    // Create DSV heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
        dsvHeapDesc.NumDescriptors = 4;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        CHECK_HR(device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&s_Context.dsvHeap)), "CreateDescriptorHeap for DSV failed");
        s_Context.dsvHeap->SetName(L"Heap_DSV");
    }

    // Create SRV heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
        srvHeapDesc.NumDescriptors = 4096;
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK_HR(device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&s_Context.srvHeap)), "CreateDescriptorHeap for SRV failed");
        s_Context.srvHeap->SetName(L"Heap_Bindless_CBV_SRV_UAV");
    }

    // Create CPU-only (non-shader-visible) UAV heap for ClearUnorderedAccessViewUint.
    // D3D12 requires the ViewCPUHandle to come from a non-shader-visible heap.
    {
        D3D12_DESCRIPTOR_HEAP_DESC cpuUavHeapDesc = {};
        cpuUavHeapDesc.NumDescriptors = 512;
        cpuUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        cpuUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE; // NOT shader-visible
        CHECK_HR(device->CreateDescriptorHeap(&cpuUavHeapDesc, IID_PPV_ARGS(&s_Context.cpuUavHeap)), "CreateDescriptorHeap for CPU UAV failed");
        s_Context.cpuUavHeap->SetName(L"Heap_CPU_UAV");
    }

    s_Context.srvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s_Context.rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    s_Context.dsvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    s_Context.srvHeapIndex = 0;
    s_Context.rtvHeapIndex = 0;
    s_Context.dsvHeapIndex = 0;
}

void GraphicsHelper::Shutdown() {
    s_Context.srvHeap.Reset();
    s_Context.rtvHeap.Reset();
    s_Context.dsvHeap.Reset();
    s_Context.cpuUavHeap.Reset();
    s_Context.device = nullptr;
}

UINT GraphicsHelper::AllocateSRV() {
    return s_Context.srvHeapIndex++;
}

UINT GraphicsHelper::AllocateRTV() {
    return s_Context.rtvHeapIndex++;
}

UINT GraphicsHelper::AllocateDSV() {
    return s_Context.dsvHeapIndex++;
}

UINT GraphicsHelper::AllocateCpuUAV() {
    return s_Context.cpuUavHeapIndex++;
}

D3D12_CPU_DESCRIPTOR_HANDLE GraphicsHelper::GetSRVCPUHandle(UINT index) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_Context.srvHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += (SIZE_T)index * s_Context.srvDescriptorSize;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE GraphicsHelper::GetSRVGPUHandle(UINT index) {
    D3D12_GPU_DESCRIPTOR_HANDLE handle = s_Context.srvHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += (UINT64)index * s_Context.srvDescriptorSize;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE GraphicsHelper::GetRTVCPUHandle(UINT index) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_Context.rtvHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += (SIZE_T)index * s_Context.rtvDescriptorSize;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE GraphicsHelper::GetDSVCPUHandle(UINT index) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_Context.dsvHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += (SIZE_T)index * s_Context.dsvDescriptorSize;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE GraphicsHelper::GetCpuUAVHandle(UINT index) {
    // CPU-only (non-shader-visible) heap — use as ViewCPUHandle in ClearUnorderedAccessViewUint
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_Context.cpuUavHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += (SIZE_T)index * s_Context.srvDescriptorSize;
    return handle;
}

void GraphicsHelper::TransitionResource(ID3D12GraphicsCommandList* cmdList, GPUResource& resource, D3D12_RESOURCE_STATES newState) {
    if (resource.state != newState) {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource.resource.Get(), resource.state, newState);
        cmdList->ResourceBarrier(1, &barrier);
        resource.state = newState;
    }
}

void GraphicsHelper::TransitionResource(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, D3D12_RESOURCE_STATES& currentState, D3D12_RESOURCE_STATES newState) {
    if (currentState != newState) {
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, currentState, newState);
        cmdList->ResourceBarrier(1, &barrier);
        currentState = newState;
    }
}

void GraphicsHelper::SetObjectName(ID3D12Resource* resource, const char* name) {
    if (!resource || !name) return;
    // ID3D12Object::SetName takes a wide string. This name shows up in
    // RenderDoc's resource inspector and PIX's object browser (and in D3D12
    // debug-layer validation messages, which is a nice side benefit).
    int len = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
    if (len <= 0) return;
    std::wstring wname(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wname.data(), len);
    resource->SetName(wname.c_str());
}

std::vector<char> GraphicsHelper::CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target)
{
    return CompileShader(filename, entryPoint, target, {});
}

std::vector<char> GraphicsHelper::CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target,
    const std::vector<std::pair<std::wstring, std::wstring>>& defines)
{
    // Resolve "Shaders/foo.hlsl" → SHADER_SOURCE_DIR "/foo.hlsl" so hot-reload
    // always reads from the source tree, not the post-build copy in Bin/.
    std::string resolvedFilename = filename;
#ifdef SHADER_SOURCE_DIR
    const std::string shadersPrefix = "Shaders/";
    if (resolvedFilename.size() >= shadersPrefix.size() &&
        resolvedFilename.compare(0, shadersPrefix.size(), shadersPrefix) == 0)
    {
        resolvedFilename = SHADER_SOURCE_DIR "/" + resolvedFilename.substr(shadersPrefix.size());
    }
#endif

    // --- Shader disk cache ---
    // Cache file: <exeDir>/ShaderCache/<stem>.<entry>.<target>[.<definesHash>].dxil
    // Load from cache if it exists and is newer than the source file.
    namespace fs = std::filesystem;
    auto GetExeDir = []() -> fs::path {
        wchar_t buf[MAX_PATH];
        GetModuleFileNameW(nullptr, buf, MAX_PATH);
        return fs::path(buf).parent_path();
    };
    fs::path cacheDir = GetExeDir() / L"ShaderCache";
    std::string cacheFileStem = fs::path(resolvedFilename).stem().string()
                                + "." + entryPoint + "." + target;
    if (!defines.empty())
    {
        std::string defStr;
        for (auto& [n, v] : defines)
            defStr += std::string(n.begin(), n.end()) + "=" + std::string(v.begin(), v.end()) + ";";
        cacheFileStem += "." + std::to_string(std::hash<std::string>{}(defStr));
    }
    fs::path cachePath = cacheDir / (cacheFileStem + ".dxil");

    {
        std::error_code ecSrc, ecCache;
        auto srcTime   = fs::last_write_time(resolvedFilename, ecSrc);
        auto cacheTime = fs::last_write_time(cachePath, ecCache);
        if (!ecSrc && !ecCache && cacheTime >= srcTime)
        {
            std::ifstream cacheIn(cachePath, std::ios::binary);
            if (cacheIn.is_open())
            {
                std::vector<char> blob((std::istreambuf_iterator<char>(cacheIn)), {});
                if (!blob.empty())
                    return blob;
            }
        }
    }

    // Load HLSL source
    std::ifstream file(resolvedFilename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open HLSL file: " << resolvedFilename << std::endl;
        return std::vector<char>();
    }

    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Create DXC compiler and utils
    Microsoft::WRL::ComPtr<IDxcUtils> dxcUtils;
    Microsoft::WRL::ComPtr<IDxcCompiler3> dxcCompiler;

    CHECK_HR(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils)), "DxcCreateInstance for DxcUtils failed");
    CHECK_HR(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler)), "DxcCreateInstance for DxcCompiler failed");

    // Create blob from source
    Microsoft::WRL::ComPtr<IDxcBlobEncoding> sourceBlob;
    CHECK_HR(dxcUtils->CreateBlob(source.c_str(), static_cast<UINT32>(source.size()), CP_UTF8, &sourceBlob), "CreateBlob failed");

    // Compile shader
    std::wstring entryPointW(entryPoint.begin(), entryPoint.end());
    std::wstring targetW(target.begin(), target.end());

    std::vector<LPCWSTR> arguments;
    arguments.push_back(L"-E");
    arguments.push_back(entryPointW.c_str());
    arguments.push_back(L"-T");
    arguments.push_back(targetW.c_str());
    arguments.push_back(L"-HV");
    arguments.push_back(L"2021");
    arguments.push_back(L"-enable-16bit-types");
#ifdef SHADER_SOURCE_DIR
    static const std::wstring shaderSourceDirW = []{
        std::string s = SHADER_SOURCE_DIR;
        return std::wstring(s.begin(), s.end());
    }();
    arguments.push_back(L"-I");
    arguments.push_back(shaderSourceDirW.c_str());
#else
    arguments.push_back(L"-I");
    arguments.push_back(L"Shaders");
#endif

#ifdef SHADER_SOURCE_DIR
    // Also expose the parent (Sources/) so "#include "Shared/SharedTypes.h"" resolves.
    static const std::wstring shaderRootDirW = []{
        namespace fs = std::filesystem;
        std::string s = fs::path(SHADER_SOURCE_DIR).parent_path().string();
        return std::wstring(s.begin(), s.end());
    }();
    arguments.push_back(L"-I");
    arguments.push_back(shaderRootDirW.c_str());
#endif

#ifdef RTXDI_INCLUDE_DIR
    std::string rtxdiInclude = RTXDI_INCLUDE_DIR;
    std::wstring rtxdiIncludeW(rtxdiInclude.begin(), rtxdiInclude.end());
    arguments.push_back(L"-I");
    arguments.push_back(rtxdiIncludeW.c_str());
#endif

#ifdef SHARC_INCLUDE_DIR
    static const std::wstring sharcDirW = []{
        std::string s = SHARC_INCLUDE_DIR;
        return std::wstring(s.begin(), s.end());
    }();
    arguments.push_back(L"-I");
    arguments.push_back(sharcDirW.c_str());
#endif

#ifdef SPD_INCLUDE_DIR
    static const std::wstring spdDirW = []{
        std::string s = SPD_INCLUDE_DIR;
        return std::wstring(s.begin(), s.end());
    }();
    arguments.push_back(L"-I");
    arguments.push_back(spdDirW.c_str());
#endif

#ifdef NRD_SHADER_INCLUDE_DIR
    static const std::wstring nrdShaderDirW = []{
        std::string s = NRD_SHADER_INCLUDE_DIR;
        return std::wstring(s.begin(), s.end());
    }();
    arguments.push_back(L"-I");
    arguments.push_back(nrdShaderDirW.c_str());
#endif

    // Per-shader compile defines (e.g. SHARC_UPDATE=1, SHARC_PROPAGATION_DEPTH=4)
    std::vector<std::wstring> defineArgs;
    for (auto& [name, value] : defines)
    {
        std::wstring arg = L"-D" + name;
        if (!value.empty()) arg += L"=" + value;
        defineArgs.push_back(arg);
    }
    for (auto& arg : defineArgs)
        arguments.push_back(arg.c_str());

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = CP_UTF8;

    Microsoft::WRL::ComPtr<IDxcIncludeHandler> includeHandler;
    dxcUtils->CreateDefaultIncludeHandler(&includeHandler);

    Microsoft::WRL::ComPtr<IDxcResult> result;
    CHECK_HR(dxcCompiler->Compile(&sourceBuffer, arguments.data(), (UINT32)arguments.size(), includeHandler.Get(), IID_PPV_ARGS(&result)), "Compile failed");

    // Check compilation result
    HRESULT statusHr;
    CHECK_HR(result->GetStatus(&statusHr), "GetStatus failed");

    if (!SUCCEEDED(statusHr))
    {
        // Get error messages
        Microsoft::WRL::ComPtr<IDxcBlobUtf8> errors;
        if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr)) && errors && errors->GetStringLength() > 0)
        {
            std::cerr << "DXC Shader Compilation Errors for " << filename << " (" << entryPoint << " -> " << target << "):" << std::endl;
            std::cerr << errors->GetStringPointer() << std::endl;
        }
        else
        {
            std::cerr << "Shader compilation failed for " << filename << " (" << entryPoint << " -> " << target << ") but no error details available." << std::endl;
        }
        return std::vector<char>();
    }

    // Get compiled shader
    Microsoft::WRL::ComPtr<IDxcBlob> shaderBlob;
    CHECK_HR(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr), "GetOutput failed");

    std::vector<char> compiledShader(shaderBlob->GetBufferSize());
    memcpy(compiledShader.data(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());

    // Write compiled blob to disk cache
    {
        std::error_code ecDir;
        fs::create_directories(cacheDir, ecDir);
        if (!ecDir)
        {
            std::ofstream cacheOut(cachePath, std::ios::binary);
            if (cacheOut.is_open())
                cacheOut.write(compiledShader.data(), static_cast<std::streamsize>(compiledShader.size()));
        }
    }

    return compiledShader;
}

void GraphicsHelper::InvalidateShaderCache()
{
    namespace fs = std::filesystem;
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    fs::path cacheDir = fs::path(buf).parent_path() / L"ShaderCache";
    std::error_code ec;
    fs::remove_all(cacheDir, ec);
    std::cout << "[ShaderCache] Cache invalidated." << std::endl;
}