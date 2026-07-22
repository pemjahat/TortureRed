# TortureRed — ReSTIR GI & NRD Pass Integration

_Built from source analysis — June 2026_

---

## Overview

TortureRed's split diffuse/specular ReSTIR GI pipeline optionally routes through an NRD Relax pass for denoising before final composite. When NRD is disabled, a direct resolve pass writes indirect lighting instead. The ReSTIR DI pipeline currently bypasses NRD entirely, outputting raw radiance straight to the lighting pass.

This document describes:

1. The ReSTIR GI pipeline structure with NRD as an optional denoising pass
2. How ReSTIR DI connects to the final lighting composite
3. Whether both pipelines can share the same NRD pass

---

## Key Source Files

| File | Role |
|---|---|
| [`Renderer.cpp`](../Sources/Renderer.cpp) — `NRDDenoise()` | Host-side scheduling: decides NRD vs direct resolve |
| [`NrdPrepareGuides.hlsl`](../Sources/Shaders/NrdPrepareGuides.hlsl) | Pre-pass: extracts motion vectors, normals, depth for NRD |
| [`NrdPackNoise.hlsl`](../Sources/Shaders/NrdPackNoise.hlsl) | Converts Final* signals into NRD RELAX-compatible packed format |
| [`NrdCompositeIndirect.hlsl`](../Sources/Shaders/NrdCompositeIndirect.hlsl) | Unpacks NRD output back into `FinalDiffuseTex` / `FinalSpecularTex` |
| [`RestirGI_ResolveIntermediates.hlsl`](../Sources/Shaders/RestirGI_ResolveIntermediates.hlsl) | Converts GI reservoir buffers → NRD-normalized float4 intermediates |
| [`NrdStoreShadingOutput.hlsl`](../Sources/Shaders/NrdStoreShadingOutput.hlsl) | Bridges DI/GI intermediates → `FinalDiffuseTex` / `FinalSpecularTex` (overwrite + additive blend) |
| [`Lighting.hlsl`](../Sources/Shaders/Lighting.hlsl) | Composites all lighting terms for final frame |

---

## ReSTIR GI Pipeline (Split Diffuse/Specular)

### Passes

| Order | Pass | Shader | Output |
|---|---|---|---|
| 1 | Diffuse Temporal | `RestirGI_Diffuse_Temporal.hlsl` | `DiffuseReservoirBuffer[curr]`, `DiffuseCandidateBuffer` |
| 2 | Specular Temporal | `RestirGI_Specular_Temporal.hlsl` | `SpecularReservoirBuffer[curr]` (reads `DiffuseCandidateBuffer` for rough-surface reuse) |
| 3 | Diffuse Spatial | `RestirGI_Diffuse_Spatial.hlsl` | `DiffuseReservoirIntermediate` |
| 4 | Specular Spatial | `RestirGI_Specular_Spatial.hlsl` | `SpecularReservoirIntermediate` |
| 5 | Resolve Intermediates | `RestirGI_ResolveIntermediates.hlsl` | `GIDiffuseIntermediate` + `GISpecularIntermediate` (NRD-normalized float4) |
| 6 | Store Shading Output | `NrdStoreShadingOutput.hlsl` (SSO Call 2) | `FinalDiffuseTex` + `FinalSpecularTex` (additive blend on DI base) |
| 7a | **NRD Denoise** | NrdPackNoise → NRD RELAX → Composite | `FinalDiffuseTex` / `FinalSpecularTex` (denoised in-place) |
| 7b | **Skip NRD** (fallback) | _(none — SSO output is final)_ | `FinalDiffuseTex` / `FinalSpecularTex` (raw, no denoising) |

### Branching logic (pass 5)

The decision between NRD and direct resolve happens in [`Renderer.cpp`](../Sources/Renderer.cpp) `DispatchRestirGI()`:

- **NRD path active** when: `enableNrdRelax=1`, no reservoir debug mode, no SHaRC debug
- **Direct resolve** otherwise

When the NRD path runs, it consists of four sub-passes:

1. **PrepareGuides** — extracts motion vectors (screen-space reprojection from `viewProjPrevious`), NRD-packed normal+roughness, and view-space Z from the G-Buffer
2. **PackSignals** — reads `DiffuseReservoirIntermediate` and `SpecularReservoirIntermediate`, evaluates each reservoir with its lobe-specific BRDF, normalizes by NRD material factors, and packs radiance + hit-distance into NRD's compressed texture format
3. **NRD Relax** — black-box spatiotemporal denoiser (diffuse and specular channels, configured with max history lengths, blur radii, etc.)
4. **Composite** — unpacks denoised output, restores BRDF material factors, writes to `RasterIndirectLightingTex`

### NRD restart logic

The host tracks `m_NrdWasActiveLastFrame`. If NRD was inactive on the previous frame (e.g., the user toggled indirect GI off then back on), the accumulation mode is set to `RESTART`, clearing NRD's internal history to avoid stale-data assertions.

### Textures involved in the NRD path

| Texture | Format | Produced by | Consumed by |
|---|---|---|---|
| `NrdMotionVectorsTex` | `R16G16_FLOAT` | PrepareGuides | NRD Relax |
| `NrdNormalRoughnessTex` | `R10G10B10A2_UNORM` | PrepareGuides | NRD Relax |
| `NrdViewZTex` | `R32_FLOAT` | PrepareGuides | NRD Relax |
| `NrdNoisyDiffuseTex` | `R16G16B16A16_FLOAT` | PackSignals (from diffuse reservoirs) | NRD Relax |
| `NrdNoisySpecularTex` | `R16G16B16A16_FLOAT` | PackSignals (from specular reservoirs) | NRD Relax |
| `NrdDenoisedDiffuseTex` | `R16G16B16A16_FLOAT` | NRD Relax | Composite |
| `NrdDenoisedSpecularTex` | `R16G16B16A16_FLOAT` | NRD Relax | Composite |
| `NrdValidationTex` | `R16G16B16A16_FLOAT` | NRD Relax (optional) | Composite (bypasses unpack) |
| `RasterIndirectLightingTex` | `R16G16B16A16_FLOAT` | Composite (or SplitResolve) | `Lighting.hlsl` |

### Key normalization contract

The PackSignals pass divides the evaluated BRDF radiance by NRD material factors before packing. The Composite pass reverses this by multiplying the denoised output by the same factors. This ensures NRD operates on a BRDF-independent signal space.

---

## ReSTIR DI Pipeline

### Passes

| Order | Pass | Shader | Output |
|---|---|---|---|
| 1 | Initial Sampling | `RestirDI_InitialSampling.hlsl` | `DIReservoirBuffer[curr]` (4-candidate RIS, no shadow rays) |
| 2 | Temporal Resampling | `RestirDI_Temporal.hlsl` | `DIReservoirBuffer[curr]` (merged with `[prev]`) |
| 3 | Spatial Resampling | `RestirDI_Spatial.hlsl` | `DIReservoirIntermediate` |
| 4 | Shade | `RestirDI_Shade.hlsl` | `DIOutputTex` (1 shadow ray + BSDF, weighted by RIS W) |

### DI output characteristics

`DIOutputTex` is `R16G16B16A16_FLOAT`. RGB stores full BSDF-weighted radiance (diffuse + specular combined, multiplied by the unbiased RIS weight `W`). Alpha stores `W`. No denoising is applied — the output is single-sample noise.

### DI consumption

In [`Lighting.hlsl`](../Sources/Shaders/Lighting.hlsl), `DIOutputTex` is sampled and added directly to the total direct lighting term alongside the main directional light (which has its own ray-traced shadow). The indirect GI term comes from `RasterIndirectLightingTex`.

---

## Structural Differences Between GI and DI Outputs

| Aspect | GI (post-spatial intermediates) | DI (post-shade output) |
|---|---|---|
| **Signal domain** | Hit-point radiance with continuous ray distance | Light radiance with binary shadow visibility |
| **Hit distance** | `firstBounceHitT` — ray length to the indirect bounce | Not tracked — shadow ray is visible/occluded, no continuous distance |
| **BRDF representation** | Evaluated per-lobe in PackSignals, normalized by NRD material factors | Full `(diff+spec) × BSDF` pre-multiplied, no normalization |
| **Sample type** | Scene hit point (`hitPos`, `hitNormal`) | Light index into `LightsBuffer` |
| **Denoising** | Via NRD Relax (temporal + spatial) | None — raw single-frame output |

---

## 🔧 SHaRC Debug Visualization — Fixed with FullScreenDebug Pass

### Status: ✅ FIXED

`m_SharcDebugPSO` is compiled and dispatched. The debug output is displayed via a dedicated `FullScreenDebug.hlsl` full-screen pass — **not** through `Lighting.hlsl`. This avoids NRD material factor distortion and is reusable for any full-screen debug mode.

### Design: Why a Dedicated Full-Screen Debug Pass?

The old approach (overlaying debug colors on `FinalDiffuseTex`/`FinalSpecularTex` and relying on `Lighting.hlsl` to composite them) had two fundamental problems:

1. **NRD material factor distortion**: `Lighting.hlsl` multiplies `FinalDiffuseTex` by `NRD_MaterialFactors().diffuseFactor` and `FinalSpecularTex` by `specularFactor`. Debug colors are not radiance — multiplying them by these physically-based factors produces incorrect, dim, or perceptually skewed visuals.

2. **Wasted GPU work**: `Lighting.hlsl` evaluates BSDFs, traces shadow rays, and computes tonemapping for pixels that only need a debug color passthrough.

**Solution**: [`FullScreenDebug.hlsl`](../Sources/Shaders/FullScreenDebug.hlsl) is a dedicated full-screen triangle pixel shader that **replaces `Lighting.hlsl` entirely** when any debug mode is active. It reads debug source textures and outputs directly to the screen — no BSDF, no shadow rays, no NRD material factors.

This is a **general-purpose** debug pass, not exclusive to SHaRC. It currently handles:

| Debug Mode | Flag | Source |
|---|---|---|
| Reservoir field debug (GI) | `restirReservoirDebugMode != OFF` | `FinalDiffuseTex` + `FinalSpecularTex` |
| SHaRC voxel visualization | `sharcDebug == 1` | `FinalDiffuseTex` (pre-filled by `SHaRC_Debug.hlsl`) |
| SHaRC bounce heatmap | `sharcDebug == 2` | `FinalDiffuseTex` (pre-filled by `SHaRC_Debug.hlsl`) |
| DI reservoir debug | `restirDIDebugMode != OFF` | `FinalDiffuseTex` + `FinalSpecularTex` |

### Root Cause — What Was Broken

| # | Problem | Detail |
|---|---------|--------|
| **1** | **No dispatch** | `m_SharcDebugPSO` was created at line 239 but never dispatched. |
| **2** | **Wrong output target** | `SHaRC_Debug.hlsl` header referenced removed `RasterIndirectLightingTex`. |
| **3** | **NRD factor distortion** | Passing debug colors through `Lighting.hlsl` applied `diffuseFactor` and `specularFactor` multiplicative weights, distorting the visualization. |

### Pipeline Flow — SHaRC Debug Active

```mermaid
flowchart TD
    A["DI: Temporal → Spatial → SplitShade"] --> B["SSO Call 1: DI → FinalDiffuse/FinalSpecular"]

    C["GI: SHaRC Update + Resolve"] --> D{"sharcDebug != 0?"}

    D -- "YES" --> E["SHaRC_Debug.hlsl (compute)<br/>traces rays, queries SHaRC<br/>→ FinalDiffuseTex + FinalSpecularTex"]
    E --> F["UAV barrier → SRV transition"]
    F --> G["early return (skip ReSTIR + NRD)"]

    G --> H["FullScreenDebug.hlsl (pixel shader)<br/>reads FinalDiffuseTex<br/>→ direct output to screen<br/>no NRD factors, no BSDF"]

    D -- "NO" --> I["Normal ReSTIR GI pipeline"]

    style E fill:#c8e6c9,stroke:#2e7d32
    style H fill:#a5d6a7,stroke:#1b5e20
```

### Three Changes Required

| # | File | Change |
|---|------|--------|
| **1** | `Renderer.cpp` → `DispatchRestirGI` | Insert SHaRC debug dispatch block **after** SHaRC Resolve but **before** ReSTIR temporal passes. When `sharcDebug != 0`: bind `FinalDiffuseTex.uavIndex` → `OutputIdx0`, `FinalSpecularTex.uavIndex` → `OutputIdx1`, dispatch debug PSO, UAV-barrier, SRV-transition, **early-return** (skip all ReSTIR + NRD). |
| **2** | `SHaRC_Debug.hlsl` | Write to **both** `OutputIdx0` and `OutputIdx1`. Both receive the same debug color. Update header comment (remove `RasterIndirectLightingTex` reference). |
| **3** | `FullScreenDebug.hlsl` (NEW) + `Application.cpp` | When any debug mode is active, dispatch `FullScreenDebug.hlsl` instead of `Lighting.hlsl`. This reads `FinalDiffuseTex` / `FinalSpecularTex` as SRVs and outputs directly to the render target. |

### SHaRC Debug Dispatch Pseudocode

```cpp
// Place AFTER SHaRC Resolve barrier (~Renderer.cpp line that dispatches SHaRC Resolve),
// BEFORE ReSTIR Diffuse Temporal dispatch.

if (frame.sharcDebug != 0 && m_SharcDebugPSO)
{
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    BindlessIndices indices = {};
    indices.OutputIdx0 = m_FinalDiffuseTex.uavIndex;
    indices.OutputIdx1 = m_FinalSpecularTex.uavIndex;
    m_CommandList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    m_CommandList->SetPipelineState(m_SharcDebugPSO.Get());
    m_CommandList->Dispatch((m_InternalWidth + 7) / 8, (m_InternalHeight + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER debugBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_FinalDiffuseTex.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_FinalSpecularTex.resource.Get()),
    };
    m_CommandList->ResourceBarrier(_countof(debugBarriers), debugBarriers);

    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    m_CurrentReservoirIndex = previousReservoir;
    return;
}
```

### FullScreenDebug.hlsl — Key Design Points

1. **Full-screen triangle**: Same VS as `Lighting.hlsl` — generates a full-screen quad from 3 vertices.
2. **Zero lighting math**: No `EvaluateBSDF`, no shadow rays, no `NRD_MaterialFactors`.
3. **Sky pixel rejection**: Reads depth from G-Buffer, outputs transparent black for sky pixels.
4. **Tonemapping passthrough**: When TAA is active, outputs raw HDR (TAA resolve handles tonemapping). When TAA is off, applies Reinhard tonemapping in-shader.
5. **Extensible**: New debug modes add `else if` branches — no changes needed to `Application.cpp` or `Renderer.cpp`.

### FullScreenDebug Dispatch (Application.cpp)

```cpp
// In Application::Render, Lighting Pass section:

const bool debugActive =
    (frame.sharcDebug != 0) ||
    (frame.restirReservoirDebugMode != RESTIR_RESERVOIR_DEBUG_OFF) ||
    (frame.restirDIDebugMode != RESTIR_DI_DEBUG_OFF);

// TAA path:
cmdList->SetPipelineState(debugActive
    ? m_Renderer.GetFullScreenDebugHdrPSO()
    : m_Renderer.GetLightingHdrPSO());

// Direct-to-backbuffer path:
cmdList->SetPipelineState(
    m_DebugShadowMap ? m_Renderer.GetDebugPSO() :
    debugActive      ? m_Renderer.GetFullScreenDebugPSO() :
                       m_Renderer.GetLightingPSO());
```

### Why the NRD Gate Alone Wasn't Enough

The NRD gate (`&& frame.sharcDebug == 0` in `useNrd`) correctly prevents NRD from running during debug mode. But it doesn't **replace** NRD's output with anything. The SSO Call 2 writes normal GI radiance into `FinalDiffuseTex` / `FinalSpecularTex`, the NRD path returns early, and `Lighting.hlsl` reads those textures. The result is normal (noisy) GI lighting — not the debug visualization. The missing piece was both the dispatch of `m_SharcDebugPSO` **and** the `FullScreenDebug.hlsl` pass to display the result without NRD material factor distortion.

---

## Feasibility: Sharing the NRD Pass

### Core idea

Merge DI's diffuse contribution into the existing NRD diffuse channel before the NRD pass runs. NRD treats the diffuse channel as a single signal — it has no awareness that the radiance originates from two different reservoir pipelines. The merged signal gets denoised in one dispatch, and the composite pass works unchanged.

### Pipeline change (conceptual)

```mermaid
flowchart LR
    subgraph current["Current"]
        GI1[GI intermediates] --> PS1[PackSignals] --> NRD1[NRD] --> CP1[Composite] --> OUT1[RasterIndirectLightingTex<br/>denoised]
        DI1[DI reservoirs] --> SH1[Shade] --> DO1[DIOutputTex<br/>noisy] --> L1[Lighting.hlsl]
    end
    subgraph proposed["Proposed"]
        GI2[GI intermediates] --> CP2[Combined Pack]
        DO2[DIOutputTex] --> CP2
        CP2 --> NRD2[NRD] --> CT2[Composite] --> OUT2[RasterIndirectLightingTex<br/>both denoised]
        DI_SPEC[DI specular] -.-> S2[handled separately]
    end
```

**Current**: GI passes through NRD and reaches `Lighting.hlsl` denoised. DI reaches `Lighting.hlsl` raw — no denoising at all.

**Proposed**: DI's diffuse contribution merges into the same NRD diffuse channel before denoising. One NRD dispatch handles both. DI specular is handled separately (kept raw, or merged into NRD's specular channel as a second phase).

### Why it works

1. **Same domain**: Both pipelines run at internal resolution on the same G-Buffer surfaces — pixel correspondence is exact.
2. **Same normalization**: Both signals can be divided by NRD material factors before NRD, making them directly additive.
3. **NRD agnosticism**: The denoiser only sees a 2D radiance + hit-distance map with motion/normal guides. Signal origin is irrelevant.

### Challenges

| Challenge | Severity | Mitigation |
|---|---|---|
| DI has no hit distance | Medium | Store light distance per pixel (world-space distance for point/spot; sentinel for directional); use luminance-weighted average for the combined hit distance |
| DI combines diffuse + specular | Medium | Use a roughness heuristic to split: blend DI toward the diffuse channel for rough surfaces, toward specular for smooth surfaces |
| Disocclusion for DI | Low | NRD primarily uses motion vectors and depth for disocclusion; hit distance is a secondary signal |
| DI W-weight already baked in | Low | Both use unbiased RIS; DI's `W` is already multiplied into `DIOutputTex.rgb` |

### Implementation outline

1. Add `lightDistance` to `DIRreservoir` — the spatial resampling passes must carry this field through merges.
2. During DI shade, compute and store world-space distance to the selected light. Use a sentinel value for directional lights.
3. Create a combined pack shader that reads both GI reservoir intermediates and `DIOutputTex`, applies the roughness heuristic to split DI into diffuse/specular portions, normalizes both by NRD material factors, computes a luminance-weighted average hit distance, and packs the combined signal.
4. In `Lighting.hlsl`, when NRD is active, skip the raw DI addition since `RasterIndirectLightingTex` now contains both. When NRD is off, add DI normally.

### Feasibility summary

| Factor | Assessment |
|---|---|
| **Technical feasibility** | High — NRD naturally handles combined signals in its diffuse channel |
| **Effort** | ~250 lines across 5 files: `SharedTypes.h`, `RestirDI_Shade.hlsl`, new combined pack shader, `Renderer.cpp`, `Lighting.hlsl` |
| **Quality gain** | DI gains 12-frame temporal accumulation instead of raw noise |
| **Performance** | Net savings — avoids a separate NRD instance; existing pass does double duty |
| **Risk** | DI hit distance is synthetic; disocclusion may be slightly less precise for DI-dominated pixels |