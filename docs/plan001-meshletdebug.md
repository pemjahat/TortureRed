# TortureRed — Meshlet Debug View Implementation Plan

_Plan for adding debug visualization overlays for visibility-buffer meshlet rendering — July 2026_

> **Parent document**: [plan000-meshlet.md](plan000-meshlet.md) — Phase 1 core meshlet pipeline.  
> This document covers **Phase 2** extracted from plan000 to keep each plan focused and manageable.

---

## 🎯 Phase 2 Goals

| Goal | Detail |
|---|---|
| **Visibility buffer output** | Meshlet rasterization writes packed `R32_UINT` (candidateIndex + primitiveID) alongside or instead of direct color |
| **Barycentric reconstruction** | Compute pixel barycentrics from screen UV + triangle clip-space positions (no HW `SV_Barycentrics` needed) |
| **Debug overlay modes** | Instance color, Meshlet color, Primitive (triangle) color |
| **Wireframe overlay** | Edge detection via barycentric derivatives on all debug modes |
| **CPU-side integration** | Console variable, PSO management, render pass dispatch |

**References**: `d:\D3D12_Research\Resources\Shaders\VisibilityDebugView.hlsl`, `VisibilityBuffer.hlsli`, `Random.hlsli`, `Common.hlsli` (Wireframe)

---

## 📋 Phase 2 Steps

### Step 1 — Add Visibility Buffer Output to Rasterization

**File**: `Sources/Shaders/MeshletRasterize.hlsl`

Currently `MeshletRasterize.hlsl` writes directly to the color buffer (Forward shading). Phase 2 adds a **second render target** (`R32_UINT`) to the PSO that stores the visibility token. This does NOT replace the color output — it adds a parallel visibility buffer that the debug view can read.

**Changes**:
1. Add `VisibilityBuffer.hlsli` include with `PackVisBuffer` / `UnpackVisBuffer` helpers
2. Add a second `SV_Target` to the PS output struct: `uint visBufferPixel : SV_TARGET1`
3. In PSMain, write `PackVisBuffer(candidateIndex, primitiveID)` to this target
4. Pass `candidateIndex` through from VS to PS as a flat `uint` attribute

```hlsl
// PS output struct — add visibility buffer target
struct PSOutput {
    float4 color        : SV_TARGET0;
    uint   visBuffer    : SV_TARGET1;
};

// In PSMain:
output.visBuffer = PackVisBuffer(input.candidateIndex, input.primitiveID);
```

**CandidateIndex / PrimitiveID** — in TortureRed's VS+PS pipeline:
- `SV_InstanceID` = index into `VisibleMeshlets[]` → this IS the candidateIndex
- `SV_PrimitiveID` = triangle index within the draw call (since we draw meshlet triangles via `DrawInstanced`)
- These must be forwarded from VS to PS as `nointerpolation` attributes

**PSO change**: Add a second RTV format `DXGI_FORMAT_R32_UINT` to the meshlet raster PSO.

---

### Step 2 — Create Visibility Buffer HLSL Helpers

**File**: New `Sources/Shaders/VisibilityBuffer.hlsli`

Port the core visibility buffer utilities from D3D12_Research:

```hlsl
#ifndef VISIBILITY_BUFFER_HLSLI
#define VISIBILITY_BUFFER_HLSLI

#include "MeshletCommon.hlsli"

// Pack: candidateIndex is 1-based (0 = invalid/sky)
uint PackVisBuffer(uint candidateIndex, uint primitiveID)
{
    return primitiveID | ((candidateIndex + 1) << 7);
}

// Unpack: returns false if pixel is sky/background
bool UnpackVisBuffer(uint data, out uint candidateIndex, out uint primitiveID)
{
    primitiveID    = data & 0x7F;          // bits [0..6]  → triangle index (max 124)
    candidateIndex = data >> 7;
    candidateIndex -= 1;                   // undo 1-based offset
    return candidateIndex != 0xFFFFFFFF;   // 0 means invalid
}
```

**Barycentric Reconstruction** — `GetVertexAttributes(float2 screenUV, ...)`:

The debug view needs per-pixel vertex attributes (world position, UV, normal, barycentrics) to:
- Draw wireframe edges (via `Wireframe(barycentrics)`)
- Colorize by instance/meshlet/primitive

Since TortureRed uses traditional VS+PS (no Mesh Shader `SV_Barycentrics`), barycentrics are computed **analytically** from the screen-space UV and the triangle's three clip-space vertex positions. This is the same technique D3D12_Research uses in `VisibilityBuffer.hlsli::GetVertexAttributes()`.

```hlsl
struct VisBufferVertexAttribute
{
    float3 Barycentrics;    // For wireframe edge detection
    // ... other interpolated attributes as needed ...
};

VisBufferVertexAttribute GetVertexAttributes(
    float2 screenUV,
    InstanceData instance,
    uint meshletIndex,
    uint primitiveID)
{
    // 1. Load MeshData, Meshlet, Triangle
    // 2. Load 3 vertex indices from meshlet indirection table
    // 3. Load 3 vertex positions, transform to clip space
    // 4. Compute barycentrics from screenUV and the 3 clip-space positions
    //    (solving 2D barycentric coordinates in screen space)
    // 5. Return VisBufferVertexAttribute
}
```

The full barycentric math is in `d:\D3D12_Research\Resources\Shaders\VisibilityBuffer.hlsli` lines 68–111 — it uses the screen-space triangle edge functions and Cramer's rule.

---

### Step 3 — Add Helper Shader Libraries

Two new HLSL include files ported from D3D12_Research:

#### 3a. `Sources/Shaders/Random.hlsli`

Hash-based random number generation for deterministic per-ID coloring:

| Function | Purpose |
|---|---|
| `uint SeedThread(uint seed)` | Wang hash → initial RNG state |
| `uint XORShift(inout uint state)` | Xorshift32 iteration |
| `float Random01(inout uint state)` | Uniform float [0,1) |
| `float3 RandomColor(inout uint state)` | Random RGB from single seed |

Source: `d:\D3D12_Research\Resources\Shaders\Random.hlsli` (112 lines)

#### 3b. Wireframe Function (in `Common.hlsl` or new include)

```hlsl
float Wireframe(float3 barycentrics, float thickness = 0.2f, float smoothing = 1.0f)
{
    float3 deltas = fwidth(barycentrics);
    float3 bary = smoothstep(deltas * thickness, deltas * (thickness + smoothing), barycentrics);
    float minBary = min(bary.x, min(bary.y, bary.z));
    return minBary;
}
```

Source: `d:\D3D12_Research\Resources\Shaders\Common.hlsli` lines ~350

---

### Step 4 — Create VisibilityDebugView.hlsl Compute Shader

**File**: New `Sources/Shaders/VisibilityDebugView.hlsl`

A full-screen compute shader (`[numthreads(8,8,1)]`) that reads the visibility buffer and renders debug overlays. This is a direct port of `d:\D3D12_Research\Resources\Shaders\VisibilityDebugView.hlsl`.

**Debug Modes**:

| Mode | Constant | Visualization | Seed |
|---|---|---|---|
| 0 | Off | Normal rendering (pass-through) | — |
| 1 | InstanceID | Random color per **instance** | `candidate.InstanceID` |
| 2 | MeshletID | Random color per **meshlet** | `candidate.MeshletIndex` |
| 3 | PrimitiveID | Random color per **triangle** | `primitiveID` |

**Wireframe overlay**: All modes 1–3 multiply the debug color by `saturate(Wireframe(barycentrics) + 0.8)`, darkening edges.

**Pass params** (CPU→GPU constant buffer):

```hlsl
struct PassParams
{
    uint Mode;
    Texture2D<uint>            VisibilityTexture;
    StructuredBuffer<MeshletCandidate> MeshletCandidates;
    RWTexture2D<float4>        Output;
};
```

---

### Step 5 — CPU-Side Renderer Integration

**File**: `Sources/Renderer.cpp`, `Sources/Renderer.h`

#### 5a. Console Variable

```cpp
// In Renderer.cpp, near other console variables:
ConsoleVariable g_MeshletDebugMode("r.Meshlet.DebugMode", 0);
// 0 = Off, 1 = Instance, 2 = Meshlet, 3 = Primitive
```

#### 5b. New Renderer Members

```cpp
// In Renderer.h:
Ref<PipelineState> m_pMeshletDebugViewPSO;     // VisibilityDebugView.hlsl
```

#### 5c. Resource Creation (`CreateMeshletResources` or `Initialize`)

```cpp
// Create debug view PSO
m_pMeshletDebugViewPSO = m_pDevice->CreateComputePipeline(
    m_CommonRS,
    "VisibilityDebugView.hlsl",
    "DebugRenderCS");
```

#### 5d. Frame Loop Integration

In the render loop, after meshlet rasterization:

```cpp
// ... meshlet rasterization ...

// After rasterization, run debug view if enabled:
if (g_MeshletDebugMode > 0)
{
    context.SetComputeRootSignature(m_CommonRS);
    context.SetPipelineState(m_pMeshletDebugViewPSO);

    PassParams params;
    params.Mode              = g_MeshletDebugMode;
    params.VisibilityTexture = m_VisibilityBuffer->GetSRV();
    params.MeshletCandidates = m_VisibleMeshlets->GetSRV();
    params.Output            = m_ColorTarget->GetUAV();
    context.BindRootSRV(BindingSlot::PerInstance, params);

    BindViewUniforms(context);
    context.Dispatch(threadGroupsX, threadGroupsY, 1);
}
```

#### 5e. ImGui Toggle

```cpp
if (ImGui::CollapsingHeader("Meshlet"))
{
    static constexpr const char* pDebugNames[] = {
        "Off", "InstanceID", "MeshletID", "PrimitiveID"
    };
    ImGui::Combo("Debug View", &g_MeshletDebugMode, pDebugNames, ARRAYSIZE(pDebugNames));
}
```

---

### Step 6 — Conditional Visibility Buffer Output

**Design decision**: The visibility buffer (`R32_UINT` render target) is only needed when `g_MeshletDebugMode > 0`. Two approaches:

| Approach | Pros | Cons |
|---|---|---|
| **Always-on** visibility buffer | Simpler code, no PSO switching | Always paying memory + bandwidth for second RTV |
| **Toggleable** via PSO permutation | Cheaper when debug is off | More PSOs, more complexity |

**Recommendation**: Start with **always-on** for simplicity. The `R32_UINT` render target at 1080p is ~8 MB — negligible. Bandwidth cost of writing a single `uint` per pixel is ~4 bytes/pixel, which is small compared to the existing color writes. This can be revisited for optimization later.

---

## 📦 New Files Summary

| File | Purpose | Source Reference |
|---|---|---|
| `Sources/Shaders/VisibilityBuffer.hlsli` | Pack/Unpack helpers + barycentric reconstruction | `d:\D3D12_Research\Resources\Shaders\VisibilityBuffer.hlsli` |
| `Sources/Shaders/VisibilityDebugView.hlsl` | Full-screen compute shader: 3 debug overlay modes | `d:\D3D12_Research\Resources\Shaders\VisibilityDebugView.hlsl` |
| `Sources/Shaders/Random.hlsli` | Hash-based GPU RNG, RandomColor | `d:\D3D12_Research\Resources\Shaders\Random.hlsli` |

## 🔧 Modified Files Summary

| File | Change |
|---|---|
| `Sources/Shaders/MeshletRasterize.hlsl` | Add `SV_TARGET1` (R32_UINT vis buffer), forward candidateIndex/primitiveID |
| `Sources/Shaders/Common.hlsl` (or `MeshletCommon.hlsli`) | Add `Wireframe()` function |
| `Sources/Renderer.h` | Add `m_pMeshletDebugViewPSO` member |
| `Sources/Renderer.cpp` | Add console variable, PSO creation, debug dispatch in frame loop, ImGui toggle |
| `CMakeLists.txt` | Ensure new `.hlsl` / `.hlsli` files are included in shader compilation |

---

## 🔢 Implementation Order

1. **Step 1** — Visibility buffer RTV output from rasterization
2. **Step 2** — `VisibilityBuffer.hlsli`: pack/unpack + barycentrics
3. **Step 3** — Helper libs: `Random.hlsli` + `Wireframe()`
4. **Step 4** — `VisibilityDebugView.hlsl` compute shader
5. **Step 5** — CPU integration: console var + PSO + dispatch
6. **Step 6** — Visibility buffer RTV toggle vs always-on

---

## 🚧 Known Risks & Notes

| Risk | Mitigation |
|---|---|
| **No HW barycentrics** | TortureRed uses VS+PS, not Mesh Shaders. Barycentrics computed analytically from screen UV + clip-space triangle positions — same approach D3D12_Research uses in `GetVertexAttributes()`. |
| **Visibility buffer bandwidth** | Extra `R32_UINT` RTV write per pixel (~4 bytes). At 4K this is ~33 MB per frame. Acceptable for a debug feature; can be made toggleable later. |
| **CandidateIndex mismatch** | In TortureRed's VS+PS, the VS uses `SV_InstanceID` to index `VisibleMeshlets[]`. This is exactly the candidateIndex. Must be forwarded as `nointerpolation uint` to PS. |
| **PrimitiveID counting** | With `DrawInstanced`, `SV_PrimitiveID` counts triangles within the draw call. Each meshlet is a separate draw instance, so `SV_PrimitiveID` resets to 0 per meshlet — which is correct since it represents the triangle index within that meshlet. |
| **Wireframe edge thickness** | `Wireframe()` uses `fwidth()` which is derivative-based. Works in compute shaders via `ddx`/`ddy` coarse equivalents calculated from the 2×2 quad — need to verify on TortureRed's GPU. |