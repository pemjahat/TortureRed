# Requirements Document: ReSTIR Diffuse / Specular Split

## Introduction

TortureRed currently runs a **single unified ReSTIR GI pipeline** (`RestirGI_Raster_Temporal.hlsl`, `RestirGI_Raster_Spatial.hlsl`, `RestirGI_Raster_Resolve.hlsl`) that handles both diffuse and specular indirect lighting in one reservoir stream. The reservoir stores a lobe-type flag (`RESERVOIR_SPECULAR_FLAG`) and applies roughness-scaled parameters at runtime, but both lobes still share the same reservoir buffer, the same temporal/spatial passes, and the same resolve output texture.

This feature splits the pipeline into **two independent ReSTIR streams** following the Kajiya renderer architecture:

- **RTDGI stream** — hemisphere-sampled diffuse indirect GI, long temporal history (M ≤ 16), loose Jacobian tolerance.
- **RTR stream** — VNDF/GGX-sampled specular reflections, short temporal history (M ≤ 3), tight Jacobian tolerance, reflection-direction validity check.

Additionally, the Kajiya-style **rough-surface ray reuse** strategy is implemented: when a surface's roughness exceeds a configurable threshold, the RTR pass skips its own ray trace and reuses the candidate ray already traced by the RTDGI pass for that pixel, saving one ray per rough pixel.

The final resolve pass composites both streams into a single indirect lighting texture, applying the correct BRDF lobe to each reservoir independently.

---

## Requirements

### Requirement 1 — Separate Diffuse Reservoir Buffers

**User Story:** As a rendering engineer, I want the diffuse ReSTIR stream to own its own pair of ping-pong reservoir buffers, so that its temporal history is never contaminated by specular samples.

#### Acceptance Criteria

1. WHEN the renderer initializes THEN the system SHALL allocate two `GPUBuffer` objects (`m_DiffuseReservoirBuffer[2]`) sized `screenWidth × screenHeight × sizeof(Reservoir)`, separate from any specular reservoir buffers.
2. WHEN the renderer resizes THEN the system SHALL reallocate both diffuse reservoir buffers to match the new resolution.
3. IF the diffuse reservoir buffers already exist THEN the system SHALL release them before reallocating.

---

### Requirement 2 — Separate Specular Reservoir Buffers

**User Story:** As a rendering engineer, I want the specular ReSTIR stream to own its own pair of ping-pong reservoir buffers, so that its short-history, view-dependent parameters never affect diffuse accumulation.

#### Acceptance Criteria

1. WHEN the renderer initializes THEN the system SHALL allocate two `GPUBuffer` objects (`m_SpecularReservoirBuffer[2]`) sized `screenWidth × screenHeight × sizeof(Reservoir)`, separate from the diffuse reservoir buffers.
2. WHEN the renderer resizes THEN the system SHALL reallocate both specular reservoir buffers to match the new resolution.
3. IF the specular reservoir buffers already exist THEN the system SHALL release them before reallocating.

---

### Requirement 3 — Dedicated Diffuse Temporal Pass (RTDGI)

**User Story:** As a rendering engineer, I want a dedicated diffuse temporal ReSTIR compute shader that only traces hemisphere-sampled rays and applies diffuse-optimized parameters, so that diffuse GI accumulates with maximum temporal stability.

#### Acceptance Criteria

1. WHEN the diffuse temporal pass executes THEN the system SHALL sample the indirect ray using cosine-weighted hemisphere sampling (`isDiffuse = true` forced, specular sampling disabled).
2. WHEN building the initial reservoir THEN the system SHALL use `RESTIR_TEMPORAL_MAX_HISTORY_LENGTH = 16` and `RESTIR_TEMPORAL_MAX_JACOBIAN = 10.0` (no glossy blending).
3. WHEN performing temporal reuse THEN the system SHALL skip the reflection-direction validity check entirely (diffuse has no dominant reflection lobe).
4. WHEN tagging the reservoir THEN the system SHALL always write `RESERVOIR_SPECULAR_FLAG = 0` (diffuse lobe) into `historyAge`.
5. WHEN the SHaRC cache is valid at the first-bounce hit THEN the system SHALL use cached radiance as continuation; otherwise SHALL fall back to direct lighting at the hit point.
6. WHEN the pass completes THEN the system SHALL write the result into `m_DiffuseReservoirBuffer[current]`.

---

### Requirement 4 — Dedicated Specular Temporal Pass (RTR)

**User Story:** As a rendering engineer, I want a dedicated specular temporal ReSTIR compute shader that uses VNDF/GGX sampling and applies specular-optimized parameters, so that reflections respond quickly to camera movement without ghosting.

#### Acceptance Criteria

1. WHEN the specular temporal pass executes THEN the system SHALL sample the indirect ray using GGX VNDF importance sampling (`isDiffuse = false` forced, diffuse sampling disabled).
2. WHEN building the initial reservoir THEN the system SHALL use `RESTIR_TEMPORAL_GLOSSY_MAX_HISTORY = 3` and `RESTIR_TEMPORAL_GLOSSY_MAX_JACOBIAN = 1.5`.
3. WHEN performing temporal reuse THEN the system SHALL apply the reflection-direction validity check: `dot(currentReflectionDir, prevReflectionDir) > RESTIR_TEMPORAL_REFLECTION_THRESHOLD_MAX (0.995)`.
4. WHEN tagging the reservoir THEN the system SHALL always write `RESERVOIR_SPECULAR_FLAG = 1` (specular lobe) into `historyAge`.
5. WHEN the pass completes THEN the system SHALL write the result into `m_SpecularReservoirBuffer[current]`.

---

### Requirement 5 — Rough-Surface Ray Reuse (Kajiya Strategy)

**User Story:** As a rendering engineer, I want the specular temporal pass to skip its own ray trace and reuse the diffuse candidate ray when surface roughness is high, so that we save one ray per rough pixel without quality loss.

#### Acceptance Criteria

1. WHEN the specular temporal pass begins for a pixel THEN the system SHALL read the diffuse candidate data (hit position, hit normal, continuation radiance) produced by the RTDGI pass for the same pixel from a shared intermediate buffer.
2. WHEN `surface.roughness >= RESTIR_RTR_ROUGH_REUSE_THRESHOLD` (configurable, default `0.6`) THEN the system SHALL skip the specular ray trace and use the diffuse candidate data directly as the specular initial sample.
3. WHEN `surface.roughness < RESTIR_RTR_ROUGH_REUSE_THRESHOLD` THEN the system SHALL trace its own VNDF-sampled specular ray as normal.
4. WHEN reusing the diffuse candidate THEN the system SHALL re-evaluate the target PDF using the specular BRDF (`GetTargetPDF` with specular-only evaluation) against the reused hit position and radiance.
5. WHEN the diffuse candidate data is invalid (no hit) THEN the system SHALL produce an empty initial reservoir for the specular stream regardless of roughness.

---

### Requirement 6 — Diffuse Candidate Intermediate Buffer

**User Story:** As a rendering engineer, I want the RTDGI pass to write its first-bounce candidate data to a shared intermediate buffer, so that the RTR pass can read it for rough-surface reuse without re-tracing.

#### Acceptance Criteria

1. WHEN the renderer initializes THEN the system SHALL allocate a `GPUBuffer` (`m_DiffuseCandidateBuffer`) storing per-pixel `{float3 hitPos, float3 hitNormal, float3 radiance, float hitT}` (48 bytes per pixel).
2. WHEN the RTDGI temporal pass traces a valid first-bounce hit THEN the system SHALL write `{hitPos, hitNormal, continuationRadiance, hitT}` to `m_DiffuseCandidateBuffer[pixelIndex]`.
3. WHEN the RTDGI temporal pass has no first-bounce hit THEN the system SHALL write a sentinel value (e.g., `hitT = -1.0`) to mark the entry as invalid.
4. WHEN the RTR temporal pass reads from `m_DiffuseCandidateBuffer` THEN the system SHALL treat entries with `hitT <= 0` as invalid candidates.

---

### Requirement 7 — Dedicated Diffuse Spatial Pass

**User Story:** As a rendering engineer, I want a dedicated diffuse spatial ReSTIR pass that only reuses from the diffuse reservoir stream, so that specular samples never leak into diffuse accumulation during spatial reuse.

#### Acceptance Criteria

1. WHEN the diffuse spatial pass executes THEN the system SHALL read neighbor reservoirs exclusively from `m_DiffuseReservoirBuffer[current]`.
2. WHEN evaluating neighbor validity THEN the system SHALL use diffuse-optimized thresholds: `MAX_JACOBIAN = 10.0`, `REUSE_WEIGHT_CLAMP = 64.0`, no reflection-direction check.
3. WHEN the pass completes THEN the system SHALL write results to a diffuse intermediate reservoir buffer (`m_DiffuseReservoirIntermediate`).

---

### Requirement 8 — Dedicated Specular Spatial Pass

**User Story:** As a rendering engineer, I want a dedicated specular spatial ReSTIR pass that only reuses from the specular reservoir stream with tight specular constraints, so that stale or misaligned specular samples are rejected.

#### Acceptance Criteria

1. WHEN the specular spatial pass executes THEN the system SHALL read neighbor reservoirs exclusively from `m_SpecularReservoirBuffer[current]`.
2. WHEN evaluating neighbor validity THEN the system SHALL apply the reflection-direction check per neighbor: `dot(centerReflectionDir, candidateDir) > RESTIR_SPATIAL_REFLECTION_THRESHOLD_MAX (0.98)`.
3. WHEN evaluating neighbor validity THEN the system SHALL use specular-optimized thresholds: `MAX_JACOBIAN = 1.25`, `REUSE_WEIGHT_CLAMP = 6.0`.
4. WHEN the pass completes THEN the system SHALL write results to a specular intermediate reservoir buffer (`m_SpecularReservoirIntermediate`).

---

### Requirement 9 — Split Resolve Pass

**User Story:** As a rendering engineer, I want the resolve pass to read both the diffuse and specular reservoir streams independently and composite them into a single indirect lighting texture, so that each lobe is evaluated with its correct BRDF.

#### Acceptance Criteria

1. WHEN the resolve pass executes THEN the system SHALL read from both `m_DiffuseReservoirIntermediate` and `m_SpecularReservoirIntermediate` (or their post-spatial equivalents).
2. WHEN evaluating the diffuse reservoir THEN the system SHALL apply only the diffuse BRDF lobe (`diffBRDF * r_diffuse.radiance * r_diffuse.W * NdotL`).
3. WHEN evaluating the specular reservoir THEN the system SHALL apply only the specular BRDF lobe (`specBRDF * r_specular.radiance * r_specular.W * NdotL`), gated by `FrameCB.enableIndirectSpecular`.
4. WHEN either reservoir has `W <= 0` THEN the system SHALL contribute zero for that lobe.
5. WHEN the pass completes THEN the system SHALL write `diffuse_contribution + specular_contribution` to `m_RasterIndirectLightingTex`.

---

### Requirement 10 — CPU-Side Pipeline Wiring (Renderer.cpp)

**User Story:** As a rendering engineer, I want the CPU-side renderer to create the new PSOs, allocate the new buffers, and dispatch the four passes (RTDGI temporal → RTR temporal → diffuse spatial → specular spatial → split resolve) in the correct order each frame.

#### Acceptance Criteria

1. WHEN `CreateRasterIndirectGIResources()` is called THEN the system SHALL allocate `m_DiffuseReservoirBuffer[2]`, `m_SpecularReservoirBuffer[2]`, `m_DiffuseReservoirIntermediate`, `m_SpecularReservoirIntermediate`, and `m_DiffuseCandidateBuffer`.
2. WHEN `CreateRasterIndirectGIPipelines()` is called THEN the system SHALL compile and store PSOs for: `RestirGI_Diffuse_Temporal`, `RestirGI_Specular_Temporal`, `RestirGI_Diffuse_Spatial`, `RestirGI_Specular_Spatial`, and `RestirGI_Split_Resolve`.
3. WHEN `DispatchRasterIndirectGI()` is called each frame THEN the system SHALL dispatch passes in order: (1) RTDGI Temporal → (2) RTR Temporal → (3) Diffuse Spatial → (4) Specular Spatial → (5) Split Resolve, with UAV barriers between each pass.
4. WHEN binding `BindlessIndices` for the RTR Temporal pass THEN the system SHALL include the `m_DiffuseCandidateBuffer` SRV index so the shader can read diffuse candidates.
5. WHEN the old unified `m_RestirGIRasterTemporalPSO`, `m_RestirGIRasterSpatialPSO`, and `m_RestirGIRasterResolvePSO` are replaced THEN the system SHALL remove their dispatch calls and buffer bindings from `DispatchRasterIndirectGI()`.

---

### Requirement 11 — Configurable Roughness Reuse Threshold

**User Story:** As a rendering engineer, I want the rough-surface reuse threshold to be a runtime-configurable value in `FrameConstants`, so that I can tune the crossover point between specular ray tracing and diffuse ray reuse without recompiling shaders.

#### Acceptance Criteria

1. WHEN `FrameConstants` is defined THEN the system SHALL include a `float rtrRoughReuseThreshold` field (default `0.6`).
2. WHEN the RTR temporal shader evaluates roughness THEN the system SHALL compare `surface.roughness` against `g_Frame.rtrRoughReuseThreshold` rather than a compile-time constant.
3. WHEN the application UI exposes this parameter THEN the system SHALL allow the user to adjust it in the range `[0.3, 1.0]`.

---

### Requirement 12 — Backward Compatibility and Feature Flag

**User Story:** As a rendering engineer, I want the split pipeline to be gated behind the existing `enableRasterIndirectGI` flag, so that disabling raster indirect GI still works correctly and the old path-tracer fallback is unaffected.

#### Acceptance Criteria

1. WHEN `FrameConstants.enableRasterIndirectGI == 0` THEN the system SHALL skip all five new passes and leave `m_RasterIndirectLightingTex` cleared to zero.
2. WHEN the split pipeline is active THEN the system SHALL NOT dispatch the old unified `m_RestirGIRasterTemporalPSO` / `m_RestirGIRasterSpatialPSO` / `m_RestirGIRasterResolvePSO`.
3. WHEN shader hot-reload is triggered THEN the system SHALL recompile all five new PSOs independently.
