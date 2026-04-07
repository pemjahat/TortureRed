# Requirements Document: TAA + Temporal Super-Resolution

## Introduction

This feature adds two selectable temporal anti-aliasing and upsampling modes to TortureRed (DX12/HLSL). The renderer currently uses compile-time constants `WINDOW_WIDTH = 1280` / `WINDOW_HEIGHT = 720` for both the swap chain and all internal resources.

The goal is to:

- Fix the **output (swap chain) resolution to 1920×1080** by updating `WINDOW_WIDTH` / `WINDOW_HEIGHT` constants.
- Introduce a **dynamic internal render resolution** derived as `floor(1920 / F) × floor(1080 / F)`, where `F` is a runtime **temporal upsampling factor** (default `1.5`, range `[1.0, 4.0]`).
- Allocate all **scene-dependent resources** (GBuffer, path-tracer output, accumulation buffer, NRD textures, ReSTIR/ReSTIR-GI reservoir buffers, motion vector texture) at the **internal resolution**.
- Keep the **swap chain, TAA output, and TAA history textures** at the fixed **1920×1080 output resolution**.
- Reconstruct a full **1920×1080 image** each frame by accumulating jittered sub-pixel samples over time via the TAA resolve pass.
- Provide **two selectable TAA modes** via an ImGui combo box in the "Renderer Debug" panel:
  - **Mode 0 — Naïve TSR** *(default)*: 2-pass pipeline (reproject + TAA resolve). Cheap, good enough for development and preview.
  - **Mode 1 — Kajiya TAA** *(high quality)*: Full 7-pass Kajiya confidence-based pipeline ported from Vulkan/SPIR-V to DX12/HLSL.

Both modes share the same jitter, resolution, reprojection map, and render-loop integration infrastructure. They differ only in the number of compute passes dispatched and the textures required.

---

## Requirements

### Requirement 1 — Output Resolution: Fix to 1920×1080

**User Story:** As a developer, I want the swap chain and final presented image to always be 1920×1080, so that the output resolution is stable regardless of the upsampling factor.

#### Acceptance Criteria

1. WHEN the application initializes THEN the system SHALL update the compile-time constants `WINDOW_WIDTH = 1920` and `WINDOW_HEIGHT = 1080` in `Renderer.h`.
2. WHEN the swap chain is created THEN the system SHALL use `1920 × 1080` as the swap chain dimensions.
3. WHEN the SDL window is created THEN the system SHALL use `1920 × 1080` as the window dimensions.
4. WHEN the camera aspect ratio is computed THEN the system SHALL use `1920.0f / 1080.0f`.
5. WHEN the TAA output texture is created THEN the system SHALL allocate it at `1920 × 1080`.

---

### Requirement 2 — Internal Resolution: Derived from Upsampling Factor

**User Story:** As a developer, I want the internal render resolution to be automatically derived from the upsampling factor and the fixed 1080p output, so that I can trade rendering cost for reconstruction quality at runtime.

#### Acceptance Criteria

1. WHEN the application initializes THEN the system SHALL compute `m_InternalWidth = floor(1920 / F)` and `m_InternalHeight = floor(1080 / F)`, clamped to a minimum of `320×180` and a maximum of `1920×1080`.
2. WHEN the upsampling factor `F = 1.0` THEN the system SHALL set internal resolution equal to `1920×1080` (TAA-only, no resolution reduction).
3. WHEN the upsampling factor `F = 1.5` (default) THEN the system SHALL set internal resolution to `1280×720`.
4. WHEN the upsampling factor changes at runtime THEN the system SHALL recompute `m_InternalWidth` / `m_InternalHeight` and trigger a full recreation of all internal-resolution resources on the next frame.
5. WHEN internal-resolution resources are recreated THEN the system SHALL GPU-idle (`WaitForPreviousFrame`) before releasing and reallocating resources to avoid hazards.

---

### Requirement 3 — Internal-Resolution Resource Allocation

**User Story:** As a developer, I want all scene-rendering resources to be allocated at the internal resolution, so that the GPU renders fewer pixels and the TAA pass upscales to 1080p.

#### Acceptance Criteria

1. WHEN internal-resolution resources are created THEN the system SHALL allocate the following at `m_InternalWidth × m_InternalHeight`:
   - **GBuffer textures**: `m_GBuffer.albedo`, `m_GBuffer.normal`, `m_GBuffer.material`, `m_GBuffer.depth`
   - **Path-tracer output**: `m_PathTracerOutput` (HDR), `m_PathTracerPresentOutput` (LDR), `m_AccumulationBuffer`
   - **NRD textures**: `m_NrdMotionVectorsTex`, `m_NrdNormalRoughnessTex`, `m_NrdViewZTex`, `m_NrdNoisyDiffuseTex`, `m_NrdNoisySpecularTex`, `m_NrdDenoisedDiffuseTex`, `m_NrdValidationTex`
   - **ReSTIR-DI reservoir buffers**: `m_ReservoirBuffer[2]`, `m_ReservoirIntermediate`
   - **ReSTIR-GI (raster indirect) reservoir buffers**: `m_DiffuseReservoirBuffer[2]`, `m_SpecularReservoirBuffer[2]`, `m_DiffuseReservoirIntermediate`, `m_SpecularReservoirIntermediate`, `m_DiffuseCandidateBuffer`
   - **Raster indirect lighting texture**: `m_RasterIndirectLightingTex`
   - **Motion vector texture**: `m_NrdMotionVectorsTex` (already listed above)
2. WHEN `FrameConstants` is updated each frame THEN the system SHALL write `internalWidth` and `internalHeight` so all shaders that use `g_Frame.screenWidth` / `g_Frame.screenHeight` for dispatch bounds continue to work correctly.
3. WHEN the RTXDI `ReSTIRGIStaticParameters` are set THEN the system SHALL pass `m_InternalWidth` / `m_InternalHeight` as `RenderWidth` / `RenderHeight`.
4. WHEN the NRD integration is initialized or re-initialized THEN the system SHALL pass `m_InternalWidth` / `m_InternalHeight` as the render dimensions.
5. WHEN the GBuffer viewport and scissor rect are set for raster passes THEN the system SHALL use `m_InternalWidth × m_InternalHeight`.

---

### Requirement 4 — `CreateInternalResolutionResources()` Refactor

**User Story:** As a developer, I want a single `Renderer::CreateInternalResolutionResources(uint32_t w, uint32_t h)` method that allocates all internal-resolution resources, so that both initialization and runtime resolution changes call the same code path.

#### Acceptance Criteria

1. WHEN `CreateInternalResolutionResources(w, h)` is called THEN the system SHALL release any previously allocated internal-resolution resources before reallocating.
2. WHEN `CreateInternalResolutionResources(w, h)` is called THEN the system SHALL allocate all resources listed in Requirement 3.1.
3. WHEN `Renderer::Initialize()` runs THEN the system SHALL call `CreateInternalResolutionResources(m_InternalWidth, m_InternalHeight)` instead of the current inline allocations.
4. WHEN the upsampling factor changes at runtime THEN `Application` SHALL call `m_Renderer.CreateInternalResolutionResources(newW, newH)` after GPU-idling.
5. WHEN `CreateInternalResolutionResources` is called THEN the system SHALL re-register all new textures/buffers in the bindless SRV/UAV heap and update any cached descriptor indices.

---

### Requirement 5 — Per-Frame Sub-Pixel Jitter (Halton Sequence)

**User Story:** As a developer, I want the camera projection matrix to be offset by a sub-pixel jitter each frame, so that the TAA accumulation can reconstruct sub-pixel detail.

#### Acceptance Criteria

1. WHEN a new frame begins THEN the system SHALL compute a 2D jitter offset using a **Halton(2,3) sequence** indexed by `frameIndex % N`, where `N = max(8, ceil(F²))` and `F` is the upsampling factor.
2. WHEN the jitter offset is computed THEN the system SHALL store it as `float2 taaJitter` (in pixel units of the internal resolution) in `FrameConstants`.
3. WHEN the projection matrix is built THEN the system SHALL apply the jitter by offsetting the projection matrix by `(2 * jitter.x / internalWidth, -2 * jitter.y / internalHeight)` in clip space.
4. WHEN either TAA mode samples the current frame THEN the system SHALL pass `taaJitter` to the shader via `FrameConstants.taaJitter` so the unjitter kernel can remove the sub-pixel offset during reconstruction.
5. WHEN the camera moves THEN the system SHALL NOT reset the Halton sequence index (jitter continues independently of camera motion).

---

### Requirement 6 — Shared TAA Texture Resources (Output Resolution)

**User Story:** As a developer, I want both TAA modes to share a common set of output-resolution textures, so that switching modes does not require a full resource reallocation.

#### Acceptance Criteria

1. WHEN the TAA resources are initialized THEN the system SHALL create the following **output-resolution (1920×1080)** textures shared by both modes:
   - `m_TaaHistoryTex[2]` — `R16G16B16A16_FLOAT` (ping-pong temporal history, stores `rgb + coverage`)
   - `m_TaaReprojectedHistoryTex` — `R16G16B16A16_FLOAT` (reprojected history for current frame)
   - `m_TaaClosestVelocityTex` — `R16G16_FLOAT` (dilated closest velocity)
   - `m_TaaOutputTex` — `R8G8B8A8_UNORM` (final TAA output, presented to swap chain via `CopyResource`)
2. WHEN **Kajiya TAA mode** resources are initialized THEN the system SHALL additionally create the following **output-resolution** textures:
   - `m_TaaVelocityHistoryTex[2]` — `R16G16_FLOAT`
   - `m_TaaSmoothVarHistoryTex[2]` — `R16G16B16A16_FLOAT`
   - `m_TaaFilteredHistoryTex` — `R16G16B16A16_FLOAT`
3. WHEN **Kajiya TAA mode** resources are initialized THEN the system SHALL additionally create the following **internal-resolution** textures (recreated when upsampling factor changes):
   - `m_TaaFilteredInputTex` — `R16G16B16A16_FLOAT`
   - `m_TaaFilteredInputDeviationTex` — `R16G16B16A16_FLOAT`
   - `m_TaaInputProbTex` — `R16_FLOAT`
   - `m_TaaProbFiltered1Tex` — `R16_FLOAT`
   - `m_TaaProbFiltered2Tex` — `R16_FLOAT`
4. WHEN all TAA textures are created THEN the system SHALL register them in the bindless SRV/UAV heap.

---

### Requirement 7 — Mode 0: Naïve TSR Pipeline (2 Passes, Default)

**User Story:** As a developer, I want a cheap 2-pass temporal upsampling pipeline as the default mode, so that the feature is usable immediately without the full Kajiya complexity.

#### Acceptance Criteria

1. WHEN Naïve TSR mode is active THEN the system SHALL execute exactly **2 compute passes** per frame:
   - **Pass 1 — `naive_tsr_reproject`**: Reproject the previous output-resolution history using motion vectors and depth; 3×3 closest-depth velocity dilation; output `m_TaaReprojectedHistoryTex` and `m_TaaClosestVelocityTex` at output resolution.
   - **Pass 2 — `naive_tsr_resolve`**: At output resolution — bilinear sample the internal-resolution current frame with unjitter, 3×3 neighborhood color bounding-box clamp, blend current frame with clamped history (`lerp(history, current, 0.1)` static / higher on disocclusion), coverage accumulation up to 8 samples, write to `m_TaaHistoryTex[current]` and `m_TaaOutputTex`.
2. WHEN the `naive_tsr_resolve` pass runs THEN the system SHALL pass `internalWidth`/`internalHeight` and `outputWidth`/`outputHeight` (1920×1080) via `FrameConstants` so the shader can compute the upsampling ratio.
3. WHEN a disocclusion is detected THEN the shader SHALL increase the blend weight toward the current frame to reduce ghosting.
4. WHEN Naïve TSR mode is active THEN the system SHALL NOT allocate or dispatch any Kajiya-only passes or textures.

---

### Requirement 8 — Mode 1: Kajiya TAA Pipeline (7 Passes, High Quality)

**User Story:** As a developer, I want the full Kajiya TAA pipeline available as a high-quality option, so that I can use it when reconstruction quality matters.

#### Acceptance Criteria

1. WHEN Kajiya TAA mode is active THEN the system SHALL execute **7 compute passes** per frame in order: `filter_input` → `reproject_history` → `filter_history` → `input_prob` → `filter_prob` → `filter_prob2` → `taa`.
2. WHEN the `taa` pass runs THEN the system SHALL pass `internalWidth`/`internalHeight` and `outputWidth`/`outputHeight` via `FrameConstants`.
3. WHEN the `taa` pass runs THEN the system SHALL use `FrameConstants.taaJitter` for the Lanczos unjitter kernel.
4. WHEN the `taa` pass produces output THEN the system SHALL write the final resolved color to `m_TaaOutputTex` and the temporal accumulation state to `m_TaaHistoryTex[current]`.

---

### Requirement 9 — PSO and Root Signature

**User Story:** As a developer, I want each TAA sub-pass to have its own compute PSO, so that the pipeline integrates cleanly with Torture's existing PSO management.

#### Acceptance Criteria

1. WHEN the renderer initializes THEN the system SHALL compile and create compute PSOs for **Naïve TSR**: `m_NaiveTsrReprojectPSO`, `m_NaiveTsrResolvePSO`, and `m_MotionVectorsPSO`.
2. WHEN the renderer initializes THEN the system SHALL compile and create compute PSOs for **Kajiya TAA**: `m_TaaFilterInputPSO`, `m_TaaReprojectHistoryPSO`, `m_TaaFilterHistoryPSO`, `m_TaaInputProbPSO`, `m_TaaFilterProbPSO`, `m_TaaFilterProb2PSO`, `m_TaaPSO`.
3. WHEN any TAA PSO is created THEN the system SHALL use the existing global root signature (bindless SRV heap, `FrameConstants` CBV at `b0`, `BindlessIndices` at `b1`).
4. WHEN shader hot-reload is triggered THEN the system SHALL recompile all TAA PSOs if any TAA HLSL file has changed.

---

### Requirement 10 — Integration into the Render Loop

**User Story:** As a developer, I want the active TAA mode to run after the main rendering pass and before ImGui, so that the final presented image is the TAA-resolved output.

#### Acceptance Criteria

1. WHEN the path-tracer path is active THEN the system SHALL run `GenerateMotionVectors` → active TAA pipeline after `DispatchRays`, using the path-tracer HDR output as the TAA input.
2. WHEN the raster path is active THEN the system SHALL run the active TAA pipeline after the lighting pass, replacing the direct back-buffer blit.
3. WHEN the active TAA pipeline completes THEN the system SHALL copy `m_TaaOutputTex` to the back buffer via `CopyResource`.
4. WHEN the TAA pipeline runs THEN the system SHALL correctly transition all TAA textures between `UAV` and `NON_PIXEL_SHADER_RESOURCE` states as required by each pass.
5. WHEN the frame index is reset (camera moved in path-tracer mode) THEN the system SHALL clear the TAA history coverage to zero.
6. WHEN the active TAA mode is switched at runtime THEN the system SHALL clear the TAA history coverage to zero on the next frame.

---

### Requirement 11 — ImGui Control

**User Story:** As a developer, I want to select the TAA mode and control the upsampling factor from the ImGui "Renderer Debug" panel at runtime, so that I can interactively compare quality and performance.

#### Acceptance Criteria

1. WHEN the "Renderer Debug" ImGui window is rendered THEN the system SHALL display a `Combo` labeled **"TAA Mode"** with options `"Naive TSR (2-pass, default)"` and `"Kajiya TAA (7-pass, high quality)"`, defaulting to Naïve TSR.
2. WHEN the TAA mode combo selection changes THEN the system SHALL switch the active pipeline on the next frame and clear the TAA history.
3. WHEN the "Renderer Debug" ImGui window is rendered THEN the system SHALL display a `SliderFloat` labeled **"TAA Upsampling Factor"** with range `[1.0, 4.0]`, default `1.5`.
4. WHEN the upsampling factor slider changes THEN the system SHALL recompute the internal resolution, GPU-idle, recreate all internal-resolution resources, and reset the TAA history.
5. WHEN the slider is displayed THEN the system SHALL show a tooltip: `"Internal render resolution: WxH (factor F×)"`.
6. WHEN TAA is active THEN the system SHALL display an info line: `"Internal: WxH → Output: 1920x1080 [Mode Name]"`.

---

### Requirement 12 — Reprojection Map (Motion Vectors)

**User Story:** As a developer, I want both TAA modes to use a compatible motion vector texture generated at internal resolution, so that moving objects are correctly handled without ghosting.

#### Acceptance Criteria

1. WHEN the path-tracer path is active THEN the system SHALL generate a screen-space motion vector texture at **internal resolution** from depth + current/previous jittered `viewProj` matrices via the `MotionVectors.hlsl` compute shader.
2. WHEN the raster path is active THEN the system SHALL reuse the existing `m_NrdMotionVectorsTex` (generated by the NRD prepare-guides pass) as the motion vector input.
3. WHEN the motion vector texture is generated THEN the system SHALL encode vectors as `(prevUV - currentUV)` in normalized UV space `[0,1]`.
4. IF no motion vector data is available for a pixel (sky/background, depth = 1.0) THEN the system SHALL output a zero motion vector.
5. WHEN either TAA mode's reproject pass runs THEN the system SHALL read from the same motion vector texture, so no duplication of generation is needed when switching modes.
