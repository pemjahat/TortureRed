#include "pch.h"

#include "RenderDocCapture.h"

#include <windows.h>

// The public RenderDoc header. RenderDoc expects an application to copy this
// header into its own tree. The file is vendored in Externals/renderdoc.
#include "renderdoc_app.h"

namespace
{
    // API 1.0.0 has every function this module needs, and every RenderDoc
    // release supports it. The module uses no function of a later version.
    const RENDERDOC_Version kRequestedVersion = eRENDERDOC_API_Version_1_0_0;

    RENDERDOC_API_1_0_0* AsApi(void* api)
    {
        return static_cast<RENDERDOC_API_1_0_0*>(api);
    }

    // The file name of the RenderDoc capture layer.
    const char* kRenderDocLibraryName = "renderdoc.dll";

    // Returns the RenderDoc module when RenderDoc is already in this process.
    // This is a passive check, and it never loads the library.
    HMODULE LoadedRenderDocModule()
    {
        return GetModuleHandleA(kRenderDocLibraryName);
    }

    // The command-line flags that ask for automatic attachment. One or two
    // dashes are accepted.
    bool IsAutoAttachFlag(const char* argument)
    {
        if (argument == nullptr)
            return false;

        return std::strcmp(argument, "-autoattach") == 0 ||
               std::strcmp(argument, "--autoattach") == 0;
    }

    // The install folder of RenderDoc, from the Windows registry. RenderDoc
    // registers the file class "RenderDoc.RDCCapture.1", and the icon value of
    // that class holds the path of the main executable. Returns an empty string
    // when the registry holds no such value.
    std::string RenderDocFolderFromRegistry()
    {
        HKEY key = nullptr;
        const LSTATUS opened = RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            "SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon",
            0, KEY_READ, &key);
        if (opened != ERROR_SUCCESS)
            return std::string();

        // The default value of the key has no name. RegQueryValueExA reads it
        // when the value name is null.
        char value[MAX_PATH] = {};
        DWORD size = sizeof(value);
        DWORD type = 0;
        const LSTATUS read = RegQueryValueExA(key, nullptr, nullptr, &type,
                                              reinterpret_cast<LPBYTE>(value), &size);
        RegCloseKey(key);

        if (read != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ))
            return std::string();

        std::string path(value);

        // The value can hold an icon index after a comma, and quotation marks.
        const size_t comma = path.find_last_of(',');
        if (comma != std::string::npos)
            path.resize(comma);
        if (path.size() >= 2 && path.front() == '"' && path.back() == '"')
            path = path.substr(1, path.size() - 2);

        if (path.empty())
            return std::string();

        return std::filesystem::path(path).parent_path().string();
    }
}

RenderDocCapture::~RenderDocCapture()
{
    // The DLL stays loaded. RenderDoc keeps hooks in the process, so an unload
    // near shutdown is not safe.
}

bool RenderDocCapture::Initialize(const std::string& capturePathTemplate)
{
    HMODULE module = LoadedRenderDocModule();
    if (module == nullptr)
        return false;

    auto getApi = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(module, "RENDERDOC_GetAPI"));
    if (getApi == nullptr)
        return false;

    void* api = nullptr;
    if (getApi(kRequestedVersion, &api) != 1 || api == nullptr)
        return false;

    m_Module = module;
    m_API = api;

    AsApi(m_API)->GetAPIVersion(&m_VersionMajor, &m_VersionMinor, &m_VersionPatch);

    // The overlay keeps the RenderDoc default set of bits, so it is visible.
    // The mask is deliberately not touched here.

    // The capture path is fixed. No user interface changes it.
    if (!capturePathTemplate.empty())
        AsApi(m_API)->SetCaptureFilePathTemplate(capturePathTemplate.c_str());

    return true;
}

std::string RenderDocCapture::FindRenderDocLibraryPath()
{
    // 1) An explicit install folder from the environment.
    char envPath[MAX_PATH] = {};
    const DWORD length = GetEnvironmentVariableA("RENDERDOC_PATH", envPath, MAX_PATH);
    if (length > 0 && length < MAX_PATH)
    {
        const std::filesystem::path candidate =
            std::filesystem::path(envPath) / kRenderDocLibraryName;
        if (std::filesystem::exists(candidate))
            return candidate.string();
    }

    // 2) The folder of an installed RenderDoc. The registry holds it.
    const std::string registryFolder = RenderDocFolderFromRegistry();
    if (!registryFolder.empty())
    {
        const std::filesystem::path candidate =
            std::filesystem::path(registryFolder) / kRenderDocLibraryName;
        if (std::filesystem::exists(candidate))
            return candidate.string();
    }

    // 3) The default folder of the RenderDoc installer.
    const std::filesystem::path installerDefault =
        std::filesystem::path("C:/Program Files/RenderDoc") / kRenderDocLibraryName;
    if (std::filesystem::exists(installerDefault))
        return installerDefault.string();

    return std::string();
}

bool RenderDocCapture::LoadCaptureLayer()
{
    if (LoadedRenderDocModule() != nullptr)
        return true;

    // RenderDoc must enter the process before the D3D12 device is created. A
    // load at this point is therefore early enough for the capture layer to
    // wrap the device.
    const std::string path = FindRenderDocLibraryPath();
    HMODULE module = path.empty() ? nullptr : LoadLibraryA(path.c_str());

    if (module == nullptr)
    {
        // The library search path is the last option. It covers a renderdoc.dll
        // that sits in the working directory, or in a folder on the path.
        module = LoadLibraryA(kRenderDocLibraryName);
    }

    return module != nullptr;
}

RenderDocCapture::AutoAttachResult RenderDocCapture::AutoAttach(int argc, char* argv[])
{
    bool requested = false;
    for (int index = 1; index < argc; ++index)
    {
        if (IsAutoAttachFlag(argv[index]))
        {
            requested = true;
            break;
        }
    }

    if (!requested)
        return AutoAttachResult::NotRequested;

    // A launcher started this process, and the capture layer is already here.
    if (LoadedRenderDocModule() != nullptr)
        return AutoAttachResult::AlreadyLoaded;

    if (!LoadCaptureLayer())
        return AutoAttachResult::Failed;

    return AutoAttachResult::Loaded;
}

std::string RenderDocCapture::VersionString() const
{
    if (m_API == nullptr)
        return std::string();

    return std::to_string(m_VersionMajor) + "." +
           std::to_string(m_VersionMinor) + "." +
           std::to_string(m_VersionPatch);
}

void RenderDocCapture::TriggerCapture()
{
    if (m_API == nullptr)
        return;
    AsApi(m_API)->TriggerCapture();
}

uint32_t RenderDocCapture::CaptureCount() const
{
    if (m_API == nullptr)
        return 0;
    return AsApi(m_API)->GetNumCaptures();
}

std::string RenderDocCapture::LastCapturePath() const
{
    if (m_API == nullptr)
        return std::string();

    RENDERDOC_API_1_0_0* api = AsApi(m_API);
    const uint32_t count = api->GetNumCaptures();
    if (count == 0)
        return std::string();

    // GetCapture writes the length in bytes, without the terminator. One more
    // byte holds the terminator.
    uint32_t length = 0;
    if (api->GetCapture(count - 1, nullptr, &length, nullptr) == 0 || length == 0)
        return std::string();

    std::string path(static_cast<size_t>(length) + 1, '\0');
    if (api->GetCapture(count - 1, path.data(), &length, nullptr) == 0)
        return std::string();

    path.resize(std::strlen(path.c_str()));
    return path;
}

std::string RenderDocCapture::CapturePathTemplate() const
{
    if (m_API == nullptr)
        return std::string();

    const char* pathTemplate = AsApi(m_API)->GetCaptureFilePathTemplate();
    return pathTemplate != nullptr ? std::string(pathTemplate) : std::string();
}

bool RenderDocCapture::LaunchReplayUI() const
{
    if (m_API == nullptr)
        return false;

    // Connect the replay user interface to this process.
    return AsApi(m_API)->LaunchReplayUI(1, nullptr) != 0;
}
