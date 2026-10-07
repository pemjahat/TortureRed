#pragma once

#include <cstdint>
#include <string>

// In-application RenderDoc capture control.
//
// RenderDoc must enter the process before the D3D12 device is created. Then the
// capture layer wraps the device. RenderDoc cannot attach to a device that
// already exists, so the process start is the only moment for the load.
//
// RenderDoc enters the process by one of two routes:
//
// 1. The RenderDoc launcher injects it. The module finds it with a passive
//    check of the loaded modules.
// 2. This program loads it. AutoAttach does this work at the start of main when
//    the command line holds "-autoattach". The Unreal Engine 5 RenderDoc plugin
//    uses the same method for its setting "renderdoc.AutoAttach".
//
// Every method is safe to call when RenderDoc is absent. Then the method does
// nothing and reports failure.
class RenderDocCapture
{
public:
    RenderDocCapture() = default;
    ~RenderDocCapture();

    // Folder and file prefix of the capture files. RenderDoc appends
    // "_frame<N>.rdc" to this template. The path is relative to the working
    // directory and is fixed at compile time.
    static constexpr const char* kDefaultCapturePathTemplate = "Captures/rendercore";

    // Result of a request for automatic attachment at startup.
    enum class AutoAttachResult
    {
        NotRequested,  // The command line has no auto-attach flag.
        AlreadyLoaded, // The flag is set, and RenderDoc is in this process.
        Loaded,        // The flag is set, and the capture layer is now in this process.
        Failed,        // The flag is set, and the capture layer was not found.
    };

    // Checks the command line for "-autoattach" and brings the RenderDoc capture
    // layer into this process. Call this function as the first statement of
    // main, before the D3D12 device is created.
    //
    // A load at this moment is the only method for automatic attachment.
    // RenderDoc wraps the device at creation, so a device that exists before
    // RenderDoc enters the process stays unwrapped. For the same reason,
    // RenderDoc cannot attach to a process that already runs.
    static AutoAttachResult AutoAttach(int argc, char* argv[]);

    // Brings the RenderDoc capture layer into this process. Returns true when
    // the module is in the process after the call. A false result means that
    // the layer was not found, or that the load failed.
    static bool LoadCaptureLayer();

    // Full path of renderdoc.dll, or an empty string when the module is absent.
    // The search uses the environment, the Windows registry, and the default
    // folder of the RenderDoc installer, in that order.
    static std::string FindRenderDocLibraryPath();

    // Finds the RenderDoc API and applies the capture settings. Call this once,
    // before the D3D12 device is created. Returns true when the API is ready.
    bool Initialize(const std::string& capturePathTemplate);

    bool IsAvailable() const { return m_API != nullptr; }

    // Version of the RenderDoc API that the host returned. Example: "1.6.0".
    std::string VersionString() const;

    // Dumps one frame. RenderDoc captures the next present after this call.
    // Call the function during frame recording to capture the current frame.
    void TriggerCapture();

    // Number of captures in this session, and the path of the last capture.
    uint32_t CaptureCount() const;
    std::string LastCapturePath() const;

    // Folder and file prefix of the captures. RenderDoc adds "_frame<N>.rdc".
    std::string CapturePathTemplate() const;

    // Opens the RenderDoc replay user interface for the last capture.
    bool LaunchReplayUI() const;

private:
    void* m_API = nullptr;    // RENDERDOC_API_1_0_0 pointer
    void* m_Module = nullptr; // the loaded renderdoc.dll

    int m_VersionMajor = 0;
    int m_VersionMinor = 0;
    int m_VersionPatch = 0;
};
