#include "pch.h"

#include "Core/Application.h"
#include "Core/RenderDocCapture.h"
#include <windows.h>

// Agility SDK Exports
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 618; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\"; }

int main(int argc, char* argv[])
{
    // The argument "-autoattach" brings the RenderDoc capture layer into this
    // process. RenderDoc must enter the process before the D3D12 device is
    // created, so this step is first. RenderDoc cannot attach to a device that
    // already exists.
    const RenderDocCapture::AutoAttachResult attachResult =
        RenderDocCapture::AutoAttach(argc, argv);

    if (attachResult == RenderDocCapture::AutoAttachResult::Failed)
    {
        // A capture tool must never stop the program.
        std::cout << "RenderDoc auto-attach failed: renderdoc.dll was not found.\n";
        std::cout << "Set RENDERDOC_PATH to the RenderDoc folder, or start the program\n";
        std::cout << "from RenderDoc. The program continues without a capture layer.\n";
    }

    Application app;
    app.Run();
    return 0;
}
