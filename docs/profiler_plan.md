# Microprofile Profiler Integration Plan

_Integration plan for embedding [jonasmr/microprofile](https://github.com/jonasmr/microprofile) into TortureRed — June 2026_

---

## 📋 Table of contents

- [Overview](#-overview)
- [Why microprofile](#-why-microprofile)
- [Integration steps](#-integration-steps)
- [Instrumentation: where to place profiling scopes](#-instrumentation-where-to-place-profiling-scopes)
- [ImGui toggle menu](#-imgui-toggle-menu)
- [Usage workflow](#-usage-workflow)
- [File plan](#-file-plan)

---

## 🎯 Overview

[Microprofile](https://github.com/jonasmr/microprofile) is an embeddable, cross-platform CPU + GPU profiler that ships as a handful of C++ files. It instruments code with simple macros, collects timing data per frame, and serves a live HTML dashboard over a built-in web server — no external tools required.

The goal is to integrate microprofile into TortureRed to provide **continuous, always-available frame-level profiling** of every render pass, with live visual feedback and zero friction to enable/disable via the existing ImGui debug panel.

---

## ❓ Why microprofile

| Concern | Microprofile answer |
|---------|---------------------|
| **Dependency weight** | 4 files (`microprofile.h`, `microprofile.cpp`, `microprofile_html.h`, `patch_win32.asm`). No external libraries. |
| **GPU support** | First-class D3D11/D3D12 GPU timer support via `MICROPROFILE_SCOPEGPU` macros. |
| **Viewer** | Built-in HTML/JS web server — open `localhost:1338` in a browser. No tool install. |
| **CPU overhead** | Negligible; scopes compile to minimal timestamp writes. Disabled entirely with a single `#define`. |
| **vs. RTXDI Profiler approach** | The RTXDI FullSample uses a custom `nvrhi::TimerQuery`-based Profiler class that maps enumerated sections to double-buffered timer queries. That approach requires defining every section in an enum, allocating query heaps, managing readback, and writing custom UI. Microprofile replaces all of that with macros and a built-in viewer. The RTXDI approach is **fine for a demo** but microprofile is **better for an evolving research engine** where passes are frequently added, renamed, or restructured. |

---

## 🔧 Integration steps

### Step 1: acquire the library

Add microprofile as a CMake `FetchContent` dependency alongside the existing ones in [CMakeLists.txt](d:\TortureRed\CMakeLists.txt):

```cmake
# Microprofile — embeddable CPU+GPU profiler
FetchContent_Declare(
    microprofile
    GIT_REPOSITORY https://github.com/jonasmr/microprofile.git
    GIT_TAG master  # or pin a commit hash for reproducibility
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/microprofile
)
FetchContent_MakeAvailable(microprofile)
```

Microprofile is source-only; no library target is needed. Simply include its directory:

```cmake
include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/microprofile)
```

The `microprofile.h` header discovers the graphics API at compile time via `#if defined(D3D12_H)` guards, so including the D3D12 headers **before** microprofile is critical.

### Step 2: platform configuration

Microprofile is configured entirely through preprocessor defines set **before** `#include "microprofile.h"`. The recommended setup:

```cpp
// ProfilerConfig.h — single configuration point
#pragma once

// --- Enable microprofile ---
// Comment out to completely compile-out all profiling (zero overhead)
#define MICROPROFILE_ENABLED 1

// --- GPU support ---
#define MICROPROFILE_GPU_TIMERS_D3D12 1

// --- Web server ---
#define MICROPROFILE_WEBSERVER 1
#define MICROPROFILE_WEBSERVER_PORT 1338
#define MICROPROFILE_WEBSERVER_MAXFRAMES 512   // retained frame history in the viewer

// --- Context switch tracing (Windows only) ---
#define MICROPROFILE_CONTEXT_SWITCH_TRACE 0    // leave off unless diagnosing scheduling issues

// --- Worker thread names ---
#define MICROPROFILE_THREAD_NAME_LEN 32

// --- Keybindings for the in-app overlay ---
#define MICROPROFILE_KEY_TOGGLE VK_F2          // toggle overlay visibility
#define MICROPROFILE_KEY_MODE VK_F1            // cycle overlay modes
```

Include this header before `microprofile.h` everywhere profiling is used.

### Step 3: initialization and per-frame calls

In [Application.cpp](d:\TortureRed\Sources\Application.cpp), add three integration points:

**A. Startup** — in `Application::Initialize()`, after the D3D12 device is created:

```cpp
// After m_Renderer.Initialize(hwnd) succeeds:
MicroProfileOnThreadCreate("Main");
MicroProfileSetEnableAllGroups(true);
MicroProfileSetForceMetaCounters(true);

// GPU initialization — requires device, command queue, and a command list
MicroProfileGpuInitD3D12(
    m_Renderer.GetDevice(),
    m_Renderer.GetCommandQueue(),
    m_Renderer.GetCommandList()
);
```

**B. Per-frame flip** — at the end of `Application::Render()`, just before `m_Renderer.EndFrame()`:

```cpp
// Advance microprofile to the next frame — this submits GPU timestamps and
// makes the current frame's data available to the web viewer
MicroProfileFlip(m_Renderer.GetCommandList());
```

**C. Shutdown** — in `Application::Shutdown()`:

```cpp
MicroProfileGpuShutdown();
MicroProfileOnThreadExit();
```

### Step 4: precompiled header

Add `#include "microprofile.h"` (behind the config header) to [pch.h](d:\TortureRed\Sources\pch.h) so profiling macros are available everywhere without per-file includes:

```cpp
// At the bottom of pch.h, after all D3D12 includes:
#include "ProfilerConfig.h"     // microprofile configuration
#include "microprofile.h"       // profiling macros
```

---

## 📍 Instrumentation: where to place profiling scopes

The key principle: **wrap each logical render pass in a CPU scope** (which also becomes a GPU scope when the D3D12 queue is active). Microprofile's `MICROPROFILE_SCOPE` macro times the CPU span; `MICROPROFILE_SCOPEGPU` emits a D3D12 timestamp query on the command queue to measure GPU duration.

### Render passes to instrument

Each pass below maps to a section of `Application::Render()` in [Application.cpp](d:\TortureRed\Sources\Application.cpp). The proposed scope names are designed to be readable in the microprofile viewer's hierarchical timeline.

<details>
<summary><b>Rasterizer path scope map</b></summary>

| # | Pass | Scope name | CPU/GPU | Location in `Application::Render()` |
|---|------|-----------|---------|-------------------------------------|
| 1 | Frame boundary | `Frame` | GPU | Top of render function |
| 2 | Depth pre-pass | `DepthPrePass` | Both | Around the depth-only draw block |
| 3 | G-Buffer fill | `GBuffer` | Both | Around the G-Buffer draw block |
| 4 | ReSTIR DI total | `ReSTIR_DI` | Both | Around `DispatchRestirDI()` |
| 4a | DI temporal | `DI_Temporal` | Both | Inside `DispatchRestirDI()` |
| 4b | DI spatial | `DI_Spatial` | Both | Inside `DispatchRestirDI()` |
| 4c | DI split shade | `DI_SplitShade` | Both | Inside `DispatchRestirDI()` |
| 5 | ReSTIR GI total | `ReSTIR_GI` | Both | Around `DispatchRestirGI()` |
| 5a | SHaRC update | `SHaRC_Update` | Both | Inside `DispatchRestirGI()` |
| 5b | SHaRC resolve | `SHaRC_Resolve` | Both | Inside `DispatchRestirGI()` |
| 5c | Diffuse temporal | `GI_Diffuse_Temporal` | Both | Inside `DispatchRestirGI()` |
| 5d | Specular temporal | `GI_Specular_Temporal` | Both | Inside `DispatchRestirGI()` |
| 5e | Diffuse spatial | `GI_Diffuse_Spatial` | Both | Inside `DispatchRestirGI()` |
| 5f | Specular spatial | `GI_Specular_Spatial` | Both | Inside `DispatchRestirGI()` |
| 5g | GI resolve intermediates | `GI_ResolveIntermediates` | Both | Inside `DispatchRestirGI()` |
| 5h | Store shading output | `GI_StoreOutput` | Both | Inside `DispatchRestirGI()` |
| 6 | NRD prepare guides | `NRD_PrepareGuides` | Both | Inside NRD block |
| 7 | NRD pack noise | `NRD_PackNoise` | Both | Inside NRD block |
| 8 | NRD RELAX denoise | `NRD_RELAX` | Both | Inside NRD block |
| 9 | NRD composite | `NRD_Composite` | Both | Inside NRD block |
| 10 | Lighting pass | `Lighting` | Both | Around the fullscreen triangle draw |
| 11 | Motion vectors | `MotionVectors` | Both | Around `GenerateMotionVectors()` |
| 12 | TAA reproject | `TAA_Reproject` | Both | Inside `DispatchNaiveTsr()` |
| 13 | TAA resolve | `TAA_Resolve` | Both | Inside `DispatchNaiveTsr()` |
| 14 | Copy to back buffer | `CopyToBackBuffer` | CPU | Around `CopyTextureToBackBuffer()` |
| 15 | Transparency | `Transparency` | Both | Around the alpha-blend draw block |
| 16 | ImGui overlay | `ImGui` | CPU | Around ImGui render |
| 17 | Present + fence wait | `Present` | GPU | Around `EndFrame()` |

</details>

<details>
<summary><b>Path tracer scope map</b></summary>

| # | Pass | Scope name | CPU/GPU | Location in `Application::Render()` |
|---|------|-----------|---------|-------------------------------------|
| 1 | G-Buffer fill | `GBuffer` | Both | G-Buffer draw block |
| 2 | Path trace dispatch | `PathTrace` | Both | Around `DispatchRays()` |
| 3 | Path trace present | `PTPresent` | Both | Around `CopyTextureToBackBuffer()` for PT output |
| 4 | Path viz lines | `PathViz` | Both | Around `DrawPathVizLines()` |
| 5 | TAA reproject | `TAA_Reproject` | Both | Around TAA dispatch (PT path) |
| 6 | TAA resolve | `TAA_Resolve` | Both | Around TAA resolve (PT path) |
| 7 | ImGui overlay | `ImGui` | CPU | Around ImGui render |
| 8 | Present | `Present` | GPU | Around `EndFrame()` |

</details>

### How to instrument: code patterns

**CPU-only scope** (e.g., ImGui, copy operations):

```cpp
MICROPROFILE_SCOPEI("UI", "ImGui", MP_YELLOW);
// ... ImGui render code ...
```

**GPU scope** (any pass that submits GPU work):

```cpp
{
    MICROPROFILE_SCOPEGPUI("Render", "GBuffer", MP_BLUE);
    // ... G-Buffer draw calls ...
}
```

**Combined CPU+GPU** (for passes where you want to see both the CPU dispatch cost and GPU execution time):

```cpp
{
    MICROPROFILE_SCOPEI("Render", "ReSTIR_DI", MP_RED);
    MICROPROFILE_SCOPEGPUI("Render", "ReSTIR_DI", MP_RED);
    m_Renderer.DispatchRestirDI(&m_Model, m_FrameConstants);
}
```

### Inside Renderer::DispatchRestirGI and DispatchRestirDI

The finer-grained scopes (4a–4c, 5a–5h) are placed **inside** the `Dispatch*` methods in [Renderer.cpp](d:\TortureRed\Sources\Renderer.cpp). Each dispatch block gets its own scope:

```cpp
// Example inside DispatchRestirGI:
void Renderer::DispatchRestirGI(Model* model, const FrameConstants& frame)
{
    // SHaRC update
    if (frame.enableRasterIndirectGI)
    {
        MICROPROFILE_SCOPEGPUI("GI", "SHaRC_Update", MP_CYAN);
        // ... dispatch SHaRC update CS ...
    }

    // SHaRC resolve
    {
        MICROPROFILE_SCOPEGPUI("GI", "SHaRC_Resolve", MP_CYAN);
        // ... dispatch SHaRC resolve CS ...
    }

    // Diffuse temporal
    {
        MICROPROFILE_SCOPEGPUI("GI", "Diffuse_Temporal", MP_PURPLE);
        // ... dispatch diffuse temporal CS ...
    }
    // ... etc.
}
```

---

## 🖥️ ImGui toggle menu

Add a **"Profiler"** section to the existing `RenderImGui()` function in [Application.cpp](d:\TortureRed\Sources\Application.cpp), placed after the FPS counter block.

### UI design

```cpp
ImGui::SeparatorText("🔬 Profiler");

// Master toggle
static bool profilerEnabled = true;
if (ImGui::Checkbox("Enable Microprofile", &profilerEnabled))
{
    MicroProfileSetEnableAllGroups(profilerEnabled);
}

// Web server info
if (profilerEnabled)
{
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Server: http://localhost:1338");
    if (ImGui::SmallButton("Open in Browser"))
    {
        ShellExecuteA(nullptr, "open", "http://localhost:1338", nullptr, nullptr, SW_SHOWNORMAL);
    }

    ImGui::Separator();

    // Overlay control
    static bool showOverlay = false;
    if (ImGui::Checkbox("Show In-App Overlay", &showOverlay))
    {
        MicroProfileSetDisplayMode(showOverlay
            ? MicroProfileOnScreenMode::Bars
            : MicroProfileOnScreenMode::Off);
    }

    // GPU timing aggregation
    static bool aggregateGPUTimers = true;
    if (ImGui::Checkbox("Aggregate GPU Timers", &aggregateGPUTimers))
    {
        MicroProfileGpuSetAggregate(aggregateGPUTimers);
    }

    // Dump to file
    if (ImGui::Button("Dump Frame to HTML"))
    {
        MicroProfileDumpFileImmediately("microprofile_dump.html", nullptr, 0);
    }

    ImGui::SameLine();
    ImGui::TextDisabled("(saves to Bin/microprofile_dump.html)");

    ImGui::Separator();

    // Quick live stats
    float frameTimeMs = (float)MicroProfileGetFloat("Frame", MicroProfileCounterType::Gpu);
    ImGui::Text("GPU frame time: %.2f ms", frameTimeMs);
}
else
{
    ImGui::TextDisabled("Profiling disabled (zero overhead)");
}
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Separate "Profiler" section** | Keeps profiling controls visually distinct from rendering toggles. Users scan for the emoji. |
| **Checkbox, not ImGui::MenuItem** | The existing "Renderer Debug" window uses checkboxes for toggles; consistency matters. |
| **Open in Browser button** | Removes friction. Developers shouldn't need to remember port 1338. |
| **Live GPU frame time** | Quick sanity check without opening the browser. |
| **Dump to HTML** | Useful for sharing profiling data offline or comparing runs. |
| **No per-scope toggles in ImGui** | The web viewer already provides per-scope filtering. ImGui stays simple. |

---

## 📖 Usage workflow

### Day-to-day profiling

1. Launch TortureRed — profiling starts automatically (if `MICROPROFILE_ENABLED` is 1)
2. Open `http://localhost:1338` in a browser
3. Observe the live timeline of all render passes
4. Use the ImGui **Profiler** section to toggle the in-app overlay for at-a-glance GPU times
5. Toggle `Enable Microprofile` off in ImGui when measuring non-instrumented performance (or ship with `MICROPROFILE_ENABLED 0`)

### Investigating a frame spike

1. With the browser dashboard open, reproduce the spike
2. Click any scope bar in the timeline — microprofile shows min/max/avg across retained frames
3. Expand the hierarchical view to see which sub-pass is the bottleneck
4. Check the **CPU vs GPU** breakdown per scope: if GPU time > CPU time, the pass is GPU-bound; if CPU time is high with low GPU time, look for driver/API overhead

### Comparing two configurations

1. Run config A, click **Pause** in the web viewer to freeze the frame data
2. Run config B (in a separate launch or after changing settings)
3. Compare side-by-side in two browser tabs

### Release builds

Set `MICROPROFILE_ENABLED 0` in `ProfilerConfig.h`. All macros compile to nothing. Zero runtime cost.

---

## 📁 File plan

| File | Action | Description |
|------|--------|-------------|
| `ThirdParty/microprofile/` | **New directory** | FetchContent downloads microprofile here |
| `Sources/ProfilerConfig.h` | **New file** | Microprofile compile-time configuration (defines, port, keybinds) |
| `Sources/pch.h` | **Modify** | Add `#include "ProfilerConfig.h"` and `#include "microprofile.h"` at bottom |
| `CMakeLists.txt` | **Modify** | Add `FetchContent_Declare(microprofile ...)` and `include_directories` |
| `Sources/Application.cpp` | **Modify** | Add init/shutdown/flip calls; add ImGui Profiler section; add top-level GPU scopes |
| `Sources/Renderer.cpp` | **Modify** | Add sub-pass GPU scopes inside `DispatchRestirGI()`, `DispatchRestirDI()`, `NRDDenoise()`, `DispatchNaiveTsr()` |
| `Sources/Renderer.h` | **No changes** | Scopes are added inline; no API changes needed |

### Files NOT modified

The following files need **no changes** because scopes are injected inline and microprofile is header-only:

- `Sources/Application.h` — no new members (profiler state is static/global in microprofile)
- `Sources/Shared/SharedTypes.h` — no new frame constants
- `Sources/Model.*` — scopes go in the render loop, not inside model code
- All shader files — profiling is host-side only

---

## 🔗 References

- Microprofile repository: [jonasmr/microprofile](https://github.com/jonasmr/microprofile)
- TortureRed render pipeline: [RenderPipeline.md](d:\TortureRed\docs\RenderPipeline.md)
- TortureRed build system: [CMakeLists.txt](d:\TortureRed\CMakeLists.txt)
- RTXDI FullSample Profiler (alternative approach): [Profiler.h](d:\RTXDI\Samples\FullSample\Source\Profiler.h)
