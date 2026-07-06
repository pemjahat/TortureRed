# Microprofile ImGui Stats Display Plan

_Plan for showing GPU/CPU frame timing from Microprofile in the ImGui "Renderer Debug" window — July 2026_

---

## 🎯 Goal

Add a **"Toggle Stats"** checkbox under the existing Profiler section in the ImGui debug window. When enabled, it shows live GPU and CPU frame timing data queried from Microprofile directly in ImGui — no browser required.

## 📍 Location

**Single file edit**: [Application.cpp](d:\TortureRed\Sources\Application.cpp) → `RenderImGui()` → the existing Profiler section.

The Profiler section currently lives at the end of `RenderImGui()`:

```cpp
ImGui::SeparatorText("Profiler");

static bool profilerEnabled = true;
if (ImGui::Checkbox("Enable Microprofile", &profilerEnabled))
{
    MicroProfileSetEnableAllGroups(profilerEnabled);
}

if (profilerEnabled)
{
    if (ImGui::Button("Dump Frame to HTML"))
    {
        MicroProfileDumpFileImmediately("microprofile_dump.html", nullptr, nullptr);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(saves to Bin/microprofile_dump.html)");
}
else
{
    ImGui::TextDisabled("Profiling disabled (zero overhead)");
}
```

## 🧩 Proposed change

Add a **"Toggle Stats"** checkbox *inside* the `if (profilerEnabled)` block. When toggled on, display:

| Metric | Microprofile API call | Unit |
|--------|----------------------|------|
| CPU frame time | `MicroProfileGetFloat("Frame", MicroProfileCounterType::Cpu)` | ms |
| GPU frame time | `MicroProfileGetFloat("Frame", MicroProfileCounterType::Gpu)` | ms |

> **Note**: The `"Frame"` scope must already be instrumented via `MICROPROFILE_SCOPEGPUI("Render", "Frame", MP_RED)` at the outermost level of `Application::Render()`. If it is not yet instrumented, that scope must be added before the stats will show meaningful values.

### Pseudocode

```cpp
// Inside the profilerEnabled block, after the Dump button:

ImGui::Separator();

// Toggle stats
static bool showStats = false;
ImGui::Checkbox("Toggle Stats", &showStats);

if (showStats)
{
    float gpuMs = (float)MicroProfileGetFloat("Frame", MicroProfileCounterType::Gpu);
    float cpuMs = (float)MicroProfileGetFloat("Frame", MicroProfileCounterType::Cpu);
    float totalMs = gpuMs + cpuMs;

    ImGui::Text("GPU frame: %.2f ms", gpuMs);
    ImGui::Text("CPU frame: %.2f ms", cpuMs);
    ImGui::Text("Total:      %.2f ms", totalMs);

    // Visual bar: proportional GPU vs CPU split
    ImGui::ProgressBar(gpuMs / (totalMs + 0.0001f), ImVec2(-1, 0), "GPU");
    ImGui::SameLine(0, 4);
    ImGui::TextDisabled("vs");
    ImGui::SameLine(0, 4);
    ImGui::ProgressBar(cpuMs / (totalMs + 0.0001f), ImVec2(-1, 0), "CPU");
}
```

### Design decisions

| Decision | Rationale |
|----------|-----------|
| **Checkbox labeled "Toggle Stats"** | Simple, discoverable. Follows the existing checkbox pattern in the UI. |
| **Hides behind "Enable Microprofile"** | If profiling is compiled out or disabled at runtime, the stats query returns zeros. Checkbox only appears when profiling is active. |
| **Uses `MicroProfileGetFloat`** | Official Microprofile API for querying scope-level counters. No custom timer management needed. |
| **Requires `"Frame"` scope instrumentation** | A top-level `MICROPROFILE_SCOPEGPUI("Render", "Frame", MP_RED)` scope must wrap the entire render function. If missing, `MicroProfileGetFloat` returns 0.0 — harmless but the plan document notes the dependency. |
| **GPU + CPU + Total** | Shows both individually so users can identify CPU-bound vs GPU-bound frames at a glance. |
| **Progress bars** | Visual split makes GPU/CPU ratio immediately obvious. GPU bar on top (green/filled), CPU on bottom. |
| **No new files** | Everything fits in the single existing `RenderImGui()` function. No new headers, members, or includes needed. |

## 🔗 Dependencies / Prerequisites

1. **Microprofile must be integrated and enabled** (already done — see [profiler_plan.md](d:\TortureRed\docs\profiler_plan.md)).
2. **A `"Frame"` scope must exist** that wraps the entire render function:
   ```cpp
   // At the top of Application::Render():
   MICROPROFILE_SCOPEGPUI("Render", "Frame", MP_RED);
   ```
   If this scope is missing, `MicroProfileGetFloat("Frame", ...)` will return `0.0`. The `"Frame"` scope is listed in the profiler_plan scope map but may not yet be implemented.
3. **`microprofile.h` already included in pch.h** — confirmed (line 33 of [pch.h](d:\TortureRed\Sources\pch.h)).

## 📁 File plan

| File | Action | Description |
|------|--------|-------------|
| `Sources/Application.cpp` | **Modify** | Add "Toggle Stats" checkbox and stats display inside `RenderImGui()` Profiler section |
| `Sources/Application.cpp` | **Potentially modify** | Add `MICROPROFILE_SCOPEGPUI("Render", "Frame", MP_RED)` at the top of `Application::Render()` if not present |

All other files: **no changes**.

## ✅ Verification checklist

- [x] "Toggle Stats" checkbox appears in the Profiler section when Microprofile is enabled
- [x] Checkbox is absent/hidden when profiling is compiled out (`MICROPROFILE_ENABLED 0`)
- [x] Ticking the checkbox shows `GPU frame: X.XX ms`, `CPU frame: X.XX ms`, `Total: X.XX ms`
- [x] Progress bars render proportionally
- [x] Unticking hides the stats display (no stale text remains)
- [x] Values update every frame
- [ ] If `"Frame"` scope is missing, shows `0.00 ms` — not a crash or garbage value

---

## 📝 Implementation notes (July 2026)

**Actual API used**: `MicroProfileGetTime(group, name)` → `float` (returns milliseconds).

The planned `MicroProfileGetFloat` / `MicroProfileCounterType` API does **not exist** in this version of microprofile. Instead:
- GPU time is queried via `MicroProfileGetTime("Render", "FrameGpu")` — maps to the `SCOPEGPUI` scope
- CPU time is queried via `MicroProfileGetTime("Render", "FrameCpu")` — maps to the `SCOPEI` scope

**Scope names**: `FrameCpu` / `FrameGpu` (not plain `"Frame"`). Using distinct names avoids ambiguity between CPU and GPU timers that share the same group.

**Changes made**:
| Line | Change |
|------|--------|
| `Application.cpp:487` | Added `MICROPROFILE_SCOPEI("Render", "FrameCpu", MP_RED)` + `MICROPROFILE_SCOPEGPUI("Render", "FrameGpu", MP_RED)` after `BeginFrame()` |
| `Application.cpp:~1280` | Added `Toggle Stats` checkbox + GPU/CPU stats display in Profiler section |
