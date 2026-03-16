#include "pch.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <unordered_set>
#include <dxcapi.h>
#include "GraphicsHelper.h"
#include "GraphicsTypes.h"
#include "d3dx12.h"
#include "Utility.h"

GraphicsContext GraphicsHelper::s_Context;

namespace
{
    namespace fs = std::filesystem;

    struct ShaderCompileRequest
    {
        std::string filename;
        std::string entryPoint;
        std::string target;
        GraphicsHelper::ShaderDefines defines;
    };

    struct TrackedShader
    {
        ShaderCompileRequest request;
        std::vector<fs::path> dependencies;
        std::vector<fs::file_time_type> timestamps;
    };

    std::vector<TrackedShader> g_TrackedShaders;
    Microsoft::WRL::ComPtr<IDxcUtils> g_DxcUtils;
    Microsoft::WRL::ComPtr<IDxcCompiler3> g_DxcCompiler;

    fs::path NormalizePath(const fs::path& path)
    {
        std::error_code ec;
        const fs::path absolutePath = fs::absolute(path, ec);
        return ec ? path.lexically_normal() : absolutePath.lexically_normal();
    }

    fs::path FindProjectRoot()
    {
        fs::path current = NormalizePath(fs::current_path());
        while (!current.empty())
        {
            std::error_code ec;
            const bool hasCMake = fs::exists(current / "CMakeLists.txt", ec) && ec.value() == 0;
            ec.clear();
            const bool hasSources = fs::exists(current / "Sources", ec) && ec.value() == 0;
            if (hasCMake && hasSources)
                return current;

            fs::path parent = current.parent_path();
            if (parent == current)
                break;

            current = parent;
        }

        return NormalizePath(fs::current_path());
    }

    fs::path SourceRoot()
    {
        return NormalizePath(FindProjectRoot() / "Sources");
    }

    fs::path SourceShadersRoot()
    {
        return NormalizePath(SourceRoot() / "Shaders");
    }

    fs::path ResolveShaderFilePath(const std::string& filename)
    {
        const fs::path input(filename);
        if (input.is_absolute())
            return NormalizePath(input);

        const std::vector<fs::path> candidates =
        {
            NormalizePath(SourceRoot() / input),
            NormalizePath(FindProjectRoot() / input),
            NormalizePath(fs::current_path() / input),
        };

        for (const fs::path& candidate : candidates)
        {
            std::error_code ec;
            if (fs::exists(candidate, ec) && ec.value() == 0)
                return candidate;
        }

        return NormalizePath(SourceRoot() / input);
    }

    std::vector<fs::path> ShaderIncludeRoots()
    {
        std::vector<fs::path> roots;
        roots.push_back(SourceShadersRoot());
        roots.push_back(SourceRoot());
        roots.push_back(NormalizePath(fs::current_path() / "Shaders"));
        roots.push_back(NormalizePath(fs::current_path() / "Sources"));

#ifdef RTXDI_INCLUDE_DIR
        roots.push_back(NormalizePath(fs::path(RTXDI_INCLUDE_DIR)));
#endif

        return roots;
    }

    std::string ReadTextFile(const fs::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            return std::string();

        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

    std::string TrimCopy(const std::string& value)
    {
        size_t start = 0;
        while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0)
            ++start;

        size_t end = value.size();
        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
            --end;

        return value.substr(start, end - start);
    }

    fs::path ResolveIncludePath(const fs::path& includingFile, const std::string& includePath, bool systemInclude)
    {
        std::vector<fs::path> searchRoots;
        if (systemInclude == false)
            searchRoots.push_back(includingFile.parent_path());

        const std::vector<fs::path> includeRoots = ShaderIncludeRoots();
        searchRoots.insert(searchRoots.end(), includeRoots.begin(), includeRoots.end());

        for (const fs::path& root : searchRoots)
        {
            const fs::path candidate = NormalizePath(root / includePath);
            std::error_code ec;
            if (fs::exists(candidate, ec) && ec.value() == 0)
                return candidate;
        }

        return fs::path();
    }

    std::string ExpandShaderSource(const fs::path& path, std::vector<fs::path>& dependencies, std::unordered_set<std::string>& visited)
    {
        const fs::path normalizedPath = NormalizePath(path);
        const std::string visitedKey = normalizedPath.generic_string();
        if (visited.find(visitedKey) != visited.end())
            return std::string();

        visited.insert(visitedKey);
        dependencies.push_back(normalizedPath);

        std::string source = ReadTextFile(normalizedPath);
        if (source.empty() && fs::exists(normalizedPath) == false)
            return std::string();

        std::string expanded;
        size_t lineStart = 0;
        while (lineStart <= source.size())
        {
            size_t lineEnd = source.find('\n', lineStart);
            const bool hasNewline = lineEnd != std::string::npos;
            if (lineEnd == std::string::npos)
                lineEnd = source.size();

            std::string line = source.substr(lineStart, lineEnd - lineStart);
            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            const std::string trimmedLine = TrimCopy(line);
            if (trimmedLine.rfind("#include", 0) == 0)
            {
                const size_t openQuote = trimmedLine.find('"');
                const size_t openAngle = trimmedLine.find('<');
                const bool systemInclude = openQuote == std::string::npos && openAngle != std::string::npos;
                const size_t openPos = systemInclude ? openAngle : openQuote;
                const size_t closePos = systemInclude ? trimmedLine.find('>', openPos + 1) : trimmedLine.find('"', openPos + 1);
                if (openPos != std::string::npos && closePos != std::string::npos && closePos > openPos + 1)
                {
                    const std::string includePath = trimmedLine.substr(openPos + 1, closePos - openPos - 1);
                    const fs::path resolvedPath = ResolveIncludePath(normalizedPath, includePath, systemInclude);
                    if (!resolvedPath.empty())
                    {
                        expanded += ExpandShaderSource(resolvedPath, dependencies, visited);
                    }
                }
            }

            expanded += line;
            if (hasNewline)
                expanded += '\n';

            if (hasNewline == false)
                break;

            lineStart = lineEnd + 1;
        }

        return expanded;
    }

    std::string DefinesToString(const GraphicsHelper::ShaderDefines& defines)
    {
        std::ostringstream stream;
        for (const auto& define : defines)
            stream << std::string(define.first.begin(), define.first.end()) << "=" << std::string(define.second.begin(), define.second.end()) << ";";
        return stream.str();
    }

    uint64_t HashBytes(const void* data, size_t size, uint64_t seed = 1469598103934665603ull)
    {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        uint64_t hash = seed;
        for (size_t index = 0; index < size; ++index)
        {
            hash ^= bytes[index];
            hash *= 1099511628211ull;
        }

        return hash;
    }

    uint64_t HashString(const std::string& value, uint64_t seed = 1469598103934665603ull)
    {
        return HashBytes(value.data(), value.size(), seed);
    }

    std::string ToHexString(uint64_t value)
    {
        std::ostringstream stream;
        stream << std::hex << value;
        return stream.str();
    }

    std::string CompilerIdentityString()
    {
        HMODULE compilerModule = GetModuleHandleW(L"dxcompiler.dll");
        if (compilerModule == nullptr)
            compilerModule = LoadLibraryW(L"dxcompiler.dll");

        wchar_t modulePath[MAX_PATH] = {};
        if (compilerModule != nullptr)
            GetModuleFileNameW(compilerModule, modulePath, MAX_PATH);

        std::error_code ec;
        fs::file_time_type timestamp = {};
        if (modulePath[0] != 0)
            timestamp = fs::last_write_time(modulePath, ec);

        std::ostringstream stream;
        stream << "dxcompiler:" << fs::path(modulePath).string();
        if (ec.value() == 0)
            stream << ":" << timestamp.time_since_epoch().count();
        return stream.str();
    }

    fs::path CacheDirectory()
    {
        fs::path cacheDirectory = NormalizePath(fs::current_path() / "ShaderCache");
#if defined(_DEBUG)
        cacheDirectory /= "Debug";
#else
        cacheDirectory /= "Release";
#endif
        return cacheDirectory;
    }

    fs::path CachePathForRequest(const ShaderCompileRequest& request, const std::string& expandedSource)
    {
        uint64_t hash = HashString(expandedSource);
        hash = HashString(request.entryPoint, hash);
        hash = HashString(request.target, hash);
        hash = HashString(DefinesToString(request.defines), hash);
        hash = HashString(CompilerIdentityString(), hash);
        return CacheDirectory() / (ToHexString(hash) + ".dxil");
    }

    std::vector<char> ReadBinaryFile(const fs::path& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
            return std::vector<char>();

        const std::streamsize size = file.tellg();
        if (size <= 0)
            return std::vector<char>();

        file.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(size));
        file.read(bytes.data(), size);
        return bytes;
    }

    void WriteBinaryFile(const fs::path& path, const std::vector<char>& bytes)
    {
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);

        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file.is_open())
            return;

        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    void UpdateTrackedShader(const ShaderCompileRequest& request, const std::vector<fs::path>& dependencies)
    {
        auto sameRequest = [&](const TrackedShader& trackedShader)
        {
            return trackedShader.request.filename == request.filename &&
                   trackedShader.request.entryPoint == request.entryPoint &&
                   trackedShader.request.target == request.target &&
                   trackedShader.request.defines == request.defines;
        };

        auto trackedShaderIt = std::find_if(g_TrackedShaders.begin(), g_TrackedShaders.end(), sameRequest);
        if (trackedShaderIt == g_TrackedShaders.end())
        {
            g_TrackedShaders.push_back({request, {}, {}});
            trackedShaderIt = std::prev(g_TrackedShaders.end());
        }

        trackedShaderIt->dependencies = dependencies;
        trackedShaderIt->timestamps.clear();
        trackedShaderIt->timestamps.reserve(dependencies.size());

        for (const fs::path& dependency : dependencies)
        {
            std::error_code ec;
            trackedShaderIt->timestamps.push_back(fs::last_write_time(dependency, ec));
            if (ec.value() != 0)
                trackedShaderIt->timestamps.back() = fs::file_time_type::min();
        }
    }

    bool EnsureCompilerObjects()
    {
        if (g_DxcUtils == nullptr)
            CHECK_HR(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&g_DxcUtils)), "DxcCreateInstance for DxcUtils failed");

        if (g_DxcCompiler == nullptr)
            CHECK_HR(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&g_DxcCompiler)), "DxcCreateInstance for DxcCompiler failed");

        return g_DxcUtils != nullptr && g_DxcCompiler != nullptr;
    }
}

void GraphicsHelper::Initialize(ID3D12Device* device) {
    s_Context.device = device;

    // Create RTV heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
        rtvHeapDesc.NumDescriptors = 16;
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        CHECK_HR(device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&s_Context.rtvHeap)), "CreateDescriptorHeap for RTV failed");
    }

    // Create DSV heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
        dsvHeapDesc.NumDescriptors = 4;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        CHECK_HR(device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&s_Context.dsvHeap)), "CreateDescriptorHeap for DSV failed");
    }

    // Create SRV heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
        srvHeapDesc.NumDescriptors = 4096;
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK_HR(device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&s_Context.srvHeap)), "CreateDescriptorHeap for SRV failed");
    }

    s_Context.srvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s_Context.rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    s_Context.dsvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    s_Context.srvHeapIndex = 0;
    s_Context.rtvHeapIndex = 0;
    s_Context.dsvHeapIndex = 0;
}

void GraphicsHelper::Shutdown() {
    g_TrackedShaders.clear();
    g_DxcUtils.Reset();
    g_DxcCompiler.Reset();
    s_Context.srvHeap.Reset();
    s_Context.rtvHeap.Reset();
    s_Context.dsvHeap.Reset();
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

std::vector<char> GraphicsHelper::CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target)
{
    return CompileShader(filename, entryPoint, target, {});
}

std::vector<char> GraphicsHelper::CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& target,
    const ShaderDefines& defines)
{
    const fs::path resolvedShaderPath = ResolveShaderFilePath(filename);
    const ShaderCompileRequest request = { resolvedShaderPath.generic_string(), entryPoint, target, defines };
    std::vector<fs::path> dependencies;
    std::unordered_set<std::string> visited;
    const fs::path normalizedShaderPath = resolvedShaderPath;
    const std::string expandedSource = ExpandShaderSource(normalizedShaderPath, dependencies, visited);
    if (expandedSource.empty() && fs::exists(normalizedShaderPath) == false)
    {
        std::cerr << "Failed to open HLSL file: " << filename << std::endl;
        return std::vector<char>();
    }

    UpdateTrackedShader(request, dependencies);

    const fs::path cachePath = CachePathForRequest(request, expandedSource);
    std::vector<char> cachedShader = ReadBinaryFile(cachePath);
    if (!cachedShader.empty())
    {
        std::cout << "Shader cache hit: " << filename << " (" << entryPoint << " -> " << target << ")" << std::endl;
        return cachedShader;
    }

    if (!EnsureCompilerObjects())
        return std::vector<char>();

    const std::string source = ReadTextFile(normalizedShaderPath);

    // Create blob from source
    Microsoft::WRL::ComPtr<IDxcBlobEncoding> sourceBlob;
    CHECK_HR(g_DxcUtils->CreateBlob(source.c_str(), static_cast<UINT32>(source.size()), CP_UTF8, &sourceBlob), "CreateBlob failed");

    // Compile shader
    std::wstring entryPointW(entryPoint.begin(), entryPoint.end());
    std::wstring targetW(target.begin(), target.end());

    std::vector<LPCWSTR> arguments;
    std::vector<std::wstring> dynamicArguments;
    arguments.push_back(L"-E");
    arguments.push_back(entryPointW.c_str());
    arguments.push_back(L"-T");
    arguments.push_back(targetW.c_str());
    arguments.push_back(L"-HV");
    arguments.push_back(L"2021");
    arguments.push_back(L"-enable-16bit-types");

    const std::vector<fs::path> includeRoots = ShaderIncludeRoots();
    dynamicArguments.reserve(includeRoots.size() * 2 + defines.size());
    for (const fs::path& includeRoot : includeRoots)
    {
        dynamicArguments.push_back(L"-I");
        dynamicArguments.push_back(includeRoot.wstring());
    }

    // Per-shader compile defines (e.g. SHARC_UPDATE=1, SHARC_PROPAGATION_DEPTH=4)
    for (auto& [name, value] : defines)
    {
        std::wstring arg = L"-D" + name;
        if (!value.empty())
            arg += L"=" + value;
        dynamicArguments.push_back(std::move(arg));
    }

    for (const std::wstring& arg : dynamicArguments)
        arguments.push_back(arg.c_str());

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = CP_UTF8;

    Microsoft::WRL::ComPtr<IDxcIncludeHandler> includeHandler;
    g_DxcUtils->CreateDefaultIncludeHandler(&includeHandler);

    Microsoft::WRL::ComPtr<IDxcResult> result;
    CHECK_HR(g_DxcCompiler->Compile(&sourceBuffer, arguments.data(), (UINT32)arguments.size(), includeHandler.Get(), IID_PPV_ARGS(&result)), "Compile failed");

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
    WriteBinaryFile(cachePath, compiledShader);
    std::cout << "Shader cache miss: " << filename << " (" << entryPoint << " -> " << target << ")" << std::endl;

    return compiledShader;
}

bool GraphicsHelper::UpdateShaders(bool forceReloadAll)
{
    bool changed = false;

    for (TrackedShader& trackedShader : g_TrackedShaders)
    {
        bool shaderChanged = forceReloadAll;
        for (size_t dependencyIndex = 0; dependencyIndex < trackedShader.dependencies.size() && shaderChanged == false; ++dependencyIndex)
        {
            std::error_code ec;
            fs::file_time_type currentTimestamp = fs::last_write_time(trackedShader.dependencies[dependencyIndex], ec);
            if (ec.value() != 0)
                currentTimestamp = fs::file_time_type::min();

            if (dependencyIndex >= trackedShader.timestamps.size() || trackedShader.timestamps[dependencyIndex] != currentTimestamp)
                shaderChanged = true;
        }

        if (shaderChanged)
        {
            changed = true;
            std::cout << "Detected shader change: " << trackedShader.request.filename << " (" << trackedShader.request.entryPoint << " -> " << trackedShader.request.target << ")" << std::endl;

            std::vector<fs::path> dependencies;
            std::unordered_set<std::string> visited;
            ExpandShaderSource(NormalizePath(trackedShader.request.filename), dependencies, visited);
            UpdateTrackedShader(trackedShader.request, dependencies);
        }
    }

    return changed;
}

uint64_t GraphicsHelper::TrackedShaderCount()
{
    return static_cast<uint64_t>(g_TrackedShaders.size());
}

uint64_t GraphicsHelper::TrackedDependencyCount()
{
    std::unordered_set<std::string> uniquePaths;
    for (const TrackedShader& trackedShader : g_TrackedShaders)
    {
        for (const fs::path& dependency : trackedShader.dependencies)
            uniquePaths.insert(dependency.generic_string());
    }

    return static_cast<uint64_t>(uniquePaths.size());
}
