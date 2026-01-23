#include <iostream>
#include "Application.h"
#include <windows.h>

// Agility SDK Exports
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 618; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\"; }

int main(int argc, char* argv[])
{
    Application app;
    app.Run();
    return 0;
}