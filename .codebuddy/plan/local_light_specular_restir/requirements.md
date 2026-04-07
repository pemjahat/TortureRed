# Requirements Document: Local Light Specular ReSTIR (Kajiya-style)

## Introduction

TortureRed currently evaluates local lights (point/spot) in the **forward `Lighting.hlsl` pixel shader** using a simple RIS pass (`GetLocalLightDirectLightingRIS`) that fires 4 candidate shadow rays per pixel per frame. This produces noisy, non-denoised local light contributions that are composited directly into the final image without any temporal accumulation or spatial reuse.

The goal of this feature is to replace that approach with a **Kajiya-inspired stochastic local light specular pipeline**:

1. **Sample** — one stochastic local light candidate per pixel (PDF-weighted via the existing LUT), fire one shadow ray, store the result as a **local light specular reservoir** in a new ping-pong buffer.
2. **Spatial Reuse** — 8-neighbour spatial reuse pass (compute shader) that borrows local light samples from nearby pixels using the same specular BRDF target PDF already used by the RTR stream.
3. **Merge into RTR Specular** — the resolved local light specular contribution is **additively merged** into the existing `m_NrdNoisySpecularTex` (or the pre-NRD specular reservoir intermediate) so that it is denoised together with indirect specular by NRD RELAX.

This mirrors Kajiya's design where `spatial_reuse_lights.hlsl` writes local light specular additively into the same `rtr_tex` buffer that holds indirect specular, and both are denoised together.

---

## Requirements

### Requirement 1 — Local Light Sample Pass (Compute Shader)

**User Story:** As a rendering engineer, I want a dedicated compute pass that stochastically samples one local light per pixel and fires one shadow ray, so that local light specular contributions are captured with minimal per-frame cost.

#### Acceptance Criteria

1. WHEN the pass is dispatched at full resolution THEN the system SHALL select one local light per pixel using the existing LUT-based PDF sampling (`SampleSingleLight`) with a random sample drawn from the per-pixel RNG.
2. WHEN a local light is selected THEN the system SHALL evaluate the **specular BRDF only** (`EvaluateBSDF` specular component) at the primary surface hit reconstructed from the GBuffer.
3. WHEN the specular BRDF evaluation yields a non-zero contribution THEN the system SHALL fire exactly one shadow ray toward the selected light using `RayQuery` inline ray tracing.
4. WHEN the shadow ray is unoccluded THEN the system SHALL compute the weighted radiance as `specular_brdf * light_radiance * NdotL / selectionPDF` and store it in a new `LocalLightReservoir` buffer (ping-pong, same `Reservoir` struct as existing streams).
5. WHEN the shadow ray is occluded OR the surface is sky (depth == 0) THEN the system SHALL write a zeroed reservoir for that pixel.
6. WHEN `FrameConstants.numLights <= 1` (no local lights) THEN the system SHALL early-exit and write zeroed reservoirs.

---

### Requirement 2 — Local Light Specular Spatial Reuse Pass (Compute Shader)

**User Story:** As a rendering engineer, I want a spatial reuse pass that borrows local light samples from 8 neighbouring pixels, so that the stochastic 1-sample-per-pixel signal is spread across the screen before denoising.

#### Acceptance Criteria

1. WHEN the spatial reuse pass is dispatched THEN the system SHALL sample **8 neighbours** within a configurable screen-space radius (default 20 pixels) using the same jittered offset strategy as `RestirGI_Specular_Spatial.hlsl`.
2. WHEN evaluating a neighbour's reservoir for reuse THEN the system SHALL apply the same **specular-only target PDF** (`GetSpecularTargetPDF`) used in the RTR specular spatial pass.
3. WHEN a neighbour fails the depth, normal, or material similarity tests (same thresholds as `RestirGI_Specular_Spatial.hlsl`) THEN the system SHALL skip that neighbour.
4. WHEN a valid neighbour reservoir is merged THEN the system SHALL apply the **Jacobian correction** (`ComputeJacobian`) and clamp it to `[RESTIR_SPATIAL_MIN_JACOBIAN, RESTIR_SPATIAL_MAX_JACOBIAN]`.
5. WHEN the spatial reuse is complete THEN the system SHALL normalize the reservoir weight `W = w_sum / (M * selectedPDF)` and write the result to a `LocalLightReservoirIntermediate` buffer.
6. WHEN `FrameConstants.numLights <= 1` THEN the system SHALL early-exit and write zeroed reservoirs.

---

### Requirement 3 — Merge Local Light Specular into NRD Specular Input

**User Story:** As a rendering engineer, I want the resolved local light specular contribution to be additively merged into the NRD noisy specular texture, so that it is denoised together with indirect specular by NRD RELAX without requiring a separate denoiser.

#### Acceptance Criteria

1. WHEN the `NrdPackRasterIndirect.hlsl` pass runs THEN the system SHALL read both the existing specular reservoir intermediate (RTR indirect specular) AND the new `LocalLightReservoirIntermediate` buffer.
2. WHEN evaluating the local light reservoir THEN the system SHALL apply the **specular BRDF factor** (`specularBRDF * W * NdotL`) and divide by the NRD `specularFactor` (same normalization as the existing RTR specular path in `NrdPackRasterIndirect.hlsl`).
3. WHEN the local light specular radiance is computed THEN the system SHALL **add** it to the existing `specularRadiance` before packing with `RELAX_FrontEnd_PackRadianceAndHitDist`.
4. WHEN `FrameConstants.enableIndirectSpecular == 0` THEN the system SHALL skip the local light specular contribution (consistent with the existing specular enable flag).
5. WHEN the NRD path is disabled (non-NRD resolve via `RestirGI_Split_Resolve.hlsl`) THEN the system SHALL also additively include the local light specular contribution in the specular output of that pass.

---

### Requirement 4 — New GPU Buffers and PSOs in Renderer

**User Story:** As a rendering engineer, I want the new local light reservoir buffers and compute PSOs to be properly created and managed in `Renderer.cpp`/`Renderer.h`, so that the pipeline integrates cleanly with the existing frame loop.

#### Acceptance Criteria

1. WHEN `CreateRasterIndirectGIResources()` is called THEN the system SHALL allocate two ping-pong `GPUBuffer` objects (`m_LocalLightReservoirBuffer[2]`) and one intermediate `GPUBuffer` (`m_LocalLightReservoirIntermediate`), each sized `sizeof(Reservoir) * WINDOW_WIDTH * WINDOW_HEIGHT`.
2. WHEN `CreateRasterIndirectGIPipelines()` is called THEN the system SHALL compile and store two new `ID3D12PipelineState` objects: `m_LocalLightSamplePSO` and `m_LocalLightSpatialPSO`.
3. WHEN `DispatchRasterIndirectGI()` runs THEN the system SHALL dispatch the local light sample pass and spatial reuse pass **after** the specular temporal/spatial passes and **before** the NRD pack / split resolve passes.
4. WHEN the frame loop swaps reservoir ping-pong indices THEN the system SHALL use the same `m_CurrentReservoirIndex` as the existing diffuse/specular streams.
5. WHEN `FrameConstants.numLights <= 1` THEN the system SHALL skip dispatching both local light passes entirely (CPU-side guard).

---

### Requirement 5 — Lighting.hlsl: Remove Local Light RIS from Forward Pass

**User Story:** As a rendering engineer, I want to remove the existing `GetLocalLightDirectLightingRIS` call from `Lighting.hlsl`, so that local lights are no longer evaluated twice (once in the forward pass and once in the new ReSTIR pipeline).

#### Acceptance Criteria

1. WHEN `Lighting.hlsl` is updated THEN the system SHALL remove the `GetLocalLightDirectLightingRIS` call and the `if (FrameCB.numLights > 1)` block that wraps it.
2. WHEN `Lighting.hlsl` is updated THEN the system SHALL retain the main directional light (index 0) evaluation with ray-traced shadow unchanged.
3. WHEN `FrameConstants.enableRasterIndirectGI == 0` (ReSTIR GI disabled) THEN the system SHALL fall back to the existing RIS path to avoid losing local light contribution entirely (guard with a compile-time or runtime flag).

---

### Requirement 6 — Shader Correctness and PDF Consistency

**User Story:** As a rendering engineer, I want the local light sampling to maintain correct PDF accounting throughout the pipeline, so that the final denoised image is unbiased and free of energy gain/loss artifacts.

#### Acceptance Criteria

1. WHEN a local light is sampled THEN the system SHALL use `selectionPDF` from `LightConstants` (populated by the existing CPU-side LUT builder) as the proposal PDF.
2. WHEN the reservoir weight `W` is computed THEN the system SHALL use the specular-only target PDF (`GetSpecularTargetPDF`) as the denominator, consistent with the RTR specular stream.
3. WHEN the spatial reuse merges a neighbour THEN the system SHALL apply the Jacobian to account for the shift in primary surface position, preventing energy bias at surface boundaries.
4. WHEN `hitT` is stored in the reservoir THEN the system SHALL use the distance from the primary surface to the local light position (not a ray-traced hit distance), so that NRD's hit-distance-based specular blur radius is correctly guided.
