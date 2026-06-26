# RTXDI ReSTIR — Diffuse/Specular Split, NRD Feed, and Final Compositing

## 1. Overview: Render Pipeline Order

The high-level frame rendering order in the RTXDI FullSample is:

```
GBuffer → PrepareLights → PresampleLights
    → RenderDirectLighting   (ReSTIR DI)
    → RenderIndirectLighting (BRDF Path Trace / ReSTIR GI)
    → Denoiser (NRD: RELAX or REBLUR)
    → CompositingPass
    → AA / Post-process
```

Key insight: **ReSTIR DI runs first, indirect lighting runs second**, and they both write into the **same** `DiffuseLighting` / `SpecularLighting` textures — with the second pass blending **additively** into the first.

---

## 2. The `SplitBrdf` Structure

Defined in [`ShadingHelpers.hlsli`](d:/RTXDI/Samples/FullSample/Shaders/LightingPasses/ShadingHelpers.hlsli):

```hlsl
struct SplitBrdf
{
    float  demodulatedDiffuse;   // Lambert NdotL (no albedo)
    float3 specular;             // GGX * NdotL * Fresnel
};
```

The BRDF evaluation function used by both ReSTIR DI and GI is:

- **`EvaluateBrdf(surface, samplePosition)`** — computes L from surface-to-sample direction, returns a `SplitBrdf` with the diffuse and specular components separated.

---

## 3. Reservoir Structures

RTXDI uses separate reservoir types for direct and indirect lighting. Both follow the weighted reservoir sampling (WRS) model from the ReSTIR paper: new samples are streamed in, combined from neighbors, and finalized with a normalization step.

### 3.1 `RTXDI_DIReservoir` — Direct Lighting Reservoir

Defined in [`DI/Reservoir.hlsli`](d:/RTXDI/Libraries/Rtxdi/Include/Rtxdi/DI/Reservoir.hlsli):

```hlsl
struct RTXDI_DIReservoir
{
    uint   lightData;          // Light index (bits 0..30) + validity bit (31)
    uint   uvData;             // Sample UV (16.16 fixed point on the light source)
    float  weightSum;          // RIS weight sum (streaming) → inv PDF (after finalize)
    float  targetPdf;          // Target PDF of the selected sample
    float  M;                  // Number of samples considered (can be fractional for pairwise MIS)
    uint   packedVisibility;   // Per-channel visibility packed into 6+6+6 bits
    int2   spatialDistance;    // Screen-distance from visibility origin (minus motion)
    uint   age;                // Frames since visibility was generated
    float  canonicalWeight;    // For pairwise MIS computations
};
```

**Key fields explained:**

| Field | Role |
|-------|------|
| `lightData` | Encodes which light was picked (lower 31 bits) and whether the reservoir is valid (bit 31). `RTXDI_IsValidDIReservoir()` simply checks `lightData != 0`. |
| `uvData` | Where on the light source the sample landed. Recovered as `float2(sampleUV)` via `RTXDI_GetDIReservoirSampleUV()`. |
| `weightSum` | During streaming/resampling: the running RIS weight sum Σ w_i. After `RTXDI_FinalizeResampling()`: the inverse PDF `1/(p̂·M)`, used as `RTXDI_GetDIReservoirInvPdf()` during shading. |
| `targetPdf` | The target function value p̂(x) for the selected sample. |
| `M` | Total number of candidates considered (integer for basic RIS, float for pairwise MIS). |
| `packedVisibility` | Three 6-bit channels encoding RGB visibility (each 0–63 → 0.0–1.0). Stored for reuse across frames to reduce shadow rays. |
| `spatialDistance` | Pixel offset from where visibility was evaluated. Used with `age` and `maxDistance` to decide if visibility reuse is valid. |
| `age` | How many frames old the visibility data is. Together with `maxAge`, controls visibility reuse lifetime. |

**Packed form** (`RTXDI_PackedDIReservoir`, 24 bytes):
```hlsl
struct RTXDI_PackedDIReservoir
{
    uint32_t lightData;
    uint32_t uvData;
    uint32_t mVisibility;     // Packed: M (14 bits) + 3×6-bit visibility
    uint32_t distanceAge;     // Packed: spatialDistance + age
    float    targetPdf;
    float    weight;
};
```

**Core operations:**
- `RTXDI_StreamSample()` — Algorithm 3: stream a raw light sample into the reservoir via WRS
- `RTXDI_CombineDIReservoirs()` — Algorithm 4: merge another reservoir's stream (used for spatial/temporal reuse after applying Jacobian)
- `RTXDI_FinalizeResampling()` — Equation 6: normalize weight sum to produce the final inv PDF
- `RTXDI_GetDIReservoirVisibility()` — Try to reuse cached visibility from a prior frame

### 3.2 `RTXDI_GIReservoir` — Indirect Lighting Reservoir

Defined in [`GI/Reservoir.hlsli`](d:/RTXDI/Libraries/Rtxdi/Include/Rtxdi/GI/Reservoir.hlsli):

```hlsl
struct RTXDI_GIReservoir
{
    float3 position;    // World-space position of the secondary (2nd bounce) surface
    float3 normal;      // Shading normal at the secondary surface
    float3 radiance;    // Incoming radiance from the secondary surface
    float  weightSum;   // RIS weight sum (streaming) → inv PDF (after finalize)
    uint   M;           // Number of samples considered
    uint   age;         // Number of frames the sample has survived
};
```

**Key differences from DI reservoir:**

| Aspect | `RTXDI_DIReservoir` | `RTXDI_GIReservoir` |
|--------|---------------------|----------------------|
| **What is sampled** | Light source identity (index + UV) | Secondary surface (world position + normal) |
| **Sample payload** | `lightData` + `uvData` | `position` + `normal` + `radiance` |
| **Target function** | p̂(light) → `targetPdf` | Not stored separately; derived from target PDF of the secondary surface |
| **Visibility reuse** | Yes — `packedVisibility` + `spatialDistance` + `age` | No — GI does not cache visibility |
| **Pairwise MIS** | Yes — `canonicalWeight` | No |
| **Creation** | `RTXDI_StreamSample()` | `RTXDI_MakeGIReservoir(pos, normal, radiance, pdf)` |
| **Combining** | `RTXDI_CombineDIReservoirs()` | `RTXDI_CombineGIReservoirs()` |
| **Normalization** | `RTXDI_FinalizeResampling()` | `RTXDI_FinalizeGIResampling()` |
| **Validation** | `lightData != 0` | `M != 0` |

**Packed form** (`RTXDI_PackedGIReservoir`, 32 bytes):
```hlsl
struct RTXDI_PackedGIReservoir
{
    float3   position;
    uint32_t packed_miscData_age_M;  // misc flags (16b) + age (8b) + M (8b)
    uint32_t packed_radiance;        // LogLUV encoded
    float    weight;
    uint32_t packed_normal;          // Octahedral snorm2x16
    float    unused;
};
```

**Core operations:**
- `RTXDI_MakeGIReservoir(pos, normal, radiance, pdf)` — Create from a raw secondary-surface hit
- `RTXDI_CombineGIReservoirs()` — Merge another GI reservoir's stream (spatial/temporal reuse)
- `RTXDI_FinalizeGIResampling()` — Normalize the weight sum
- `RTXDI_IsValidGIReservoir()` — Check `M != 0`

### 3.3 Reservoir Lifecycle

Both reservoir types follow the same RIS lifecycle:

```
 1. INITIAL SAMPLE     ──►  MakeReservoir / StreamSample
                             weightSum = 1/pdf, M = 1

 2. STREAMING / MERGE   ──►  CombineReservoirs (temporal, spatial)
                             M += neighbor.M
                             weightSum accumulates risWeights

 3. FINALIZE            ──►  FinalizeResampling
                             weightSum becomes invPdf = (Σw · num) / (p̂ · denom)

 4. SHADE               ──►  Read invPdf from weightSum
                             Compute radiance = reservoir.radiance * weightSum (GI)
                             or scale by invPdf / solidAnglePdf (DI)
```

The `weightSum` field is **overloaded**: during steps 1–2 it stores the running RIS weight sum, and after step 3 it becomes the inverse PDF used for shading.

---

## 4. ReSTIR DI — Direct Lighting

### 4.1 Shading Path

For each pixel, ReSTIR DI:

1. Load a `RTXDI_DIReservoir` (result of initial sampling + temporal + spatial resampling)
2. Call `ShadeSurfaceWithLightSample()` → which internally calls `EvaluateBrdf()`:

```hlsl
SplitBrdf brdf = EvaluateBrdf(surface, lightSample.position);
diffuse  = brdf.demodulatedDiffuse * lightSample.radiance;   // demodulated diffuse radiance
specular = brdf.specular * lightSample.radiance;             // full specular radiance
```

3. **Demodulate specular** — divide by `specularF0`:

```hlsl
specular = DemodulateSpecular(surface.material.specularF0, specular);
// DemodulateSpecular(x, y) = y / max(0.01, x)
```

This is the "material demodulation" required by NRD: specular is stored as radiance/`specularF0` so NRD works on pure radiance.

4. Call `StoreShadingOutput()` with `isFirstPass = true`.

### 4.2 Key Files

| Shader | Purpose |
|--------|---------|
| `DI/GenerateInitialSamples.hlsl` | Initial light sampling |
| `DI/TemporalResampling.hlsl` | Temporal reuse |
| `DI/SpatialResampling.hlsl` | Spatial reuse |
| `DI/FusedResampling.hlsl` | Fused spatiotemporal + shading |
| `DI/ShadeSamples.hlsl` | Separate final shading pass |
| [`ShadingHelpers.hlsli`](d:/RTXDI/Samples/FullSample/Shaders/LightingPasses/ShadingHelpers.hlsli) | `ShadeSurfaceWithLightSample()`, `EvaluateBrdf()`, `StoreShadingOutput()` |

---

## 5. ReSTIR GI — Indirect Lighting (Global Illumination)

### 5.1 Pipeline

1. **BRDF Path Tracing** (`BrdfRayTracing.hlsl`) — trace one BRDF ray per pixel from G-buffer surfaces
2. **Shade Secondary Surfaces** (`ShadeSecondarySurfaces.hlsl`) — shade the hit points, optionally apply ReSTIR DI to them, and build initial GI reservoirs
3. **Temporal Resampling** (`GI/TemporalResampling.hlsl`) — reuse GI reservoirs across frames
4. **Spatial Resampling** (`GI/SpatialResampling.hlsl`) — reuse GI reservoirs spatially
5. **Final Shading** (`GI/FinalShading.hlsl`) — evaluate the final BRDF and output

### 5.2 Final Shading — Diffuse / Specular Split

In [`GI/FinalShading.hlsl`](d:/RTXDI/Samples/FullSample/Shaders/LightingPasses/GI/FinalShading.hlsl):

```hlsl
const RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(...);

float3 diffuse  = 0;
float3 specular = 0;

if (RTXDI_IsValidGIReservoir(reservoir))
{
    float3 radiance = reservoir.radiance * reservoir.weightSum;  // weighted radiance

    // Optional final visibility trace
    radiance *= visibility;

    // Evaluate BRDF toward the GI sample position
    const SplitBrdf brdf = EvaluateBrdf(primarySurface, reservoir.position);

    if (enableFinalMIS)
    {
        // MIS blend between ReSTIR result and initial sample
        // for rough surfaces, prefer ReSTIR; for shiny surfaces, prefer initial sample
        diffuse  = brdf.demodulatedDiffuse * radiance * finalWeight
                 + brdf0.demodulatedDiffuse * initialRadiance * initialWeight;
        specular = brdf.specular * radiance * finalWeight
                 + brdf0.specular * initialRadiance * initialWeight;
    }
    else
    {
        diffuse  = brdf.demodulatedDiffuse * radiance;
        specular = brdf.specular * radiance;
    }

    // Demodulate specular
    specular = DemodulateSpecular(primarySurface.material.specularF0, specular);
}

// isFirstPass=false → additive blend with prior data (ReSTIR DI)
StoreShadingOutput(GlobalIndex, pixelPosition,
    viewDepth, roughness, diffuse, specular, lightDistance=0, isFirstPass=false, isLastPass=true);
```

**Key point**: `isFirstPass=false` means GI output is **added** to whatever ReSTIR DI already wrote. This is how direct + indirect lighting are combined **before** denoising.

### 5.3 MIS (Multiple Importance Sampling) Detail

ReSTIR GI uses a clever MIS scheme:
- `kMISRoughness = 0.3` — threshold roughness
- A "roughened" surface version forces roughness ≥ 0.3
- `GetMISWeight()` compares rough BRDF vs true BRDF → prefers initial sample on shiny surfaces, ReSTIR result on rough surfaces
- This avoids ReSTIR's poor specular performance on low-roughness surfaces

---

## 6. `StoreShadingOutput` — Multi-Pass Blend & NRD Packing

This is the **central function** that handles both multi-pass additive blending and NRD front-end packing. Defined in [`ShadingHelpers.hlsli`](d:/RTXDI/Samples/FullSample/Shaders/LightingPasses/ShadingHelpers.hlsli).

```hlsl
void StoreShadingOutput(..., float lightDistance, bool isFirstPass, bool isLastPass)
{
    // If not first pass, additively blend with prior data
    if (!isFirstPass) {
        float4 priorDiffuse = u_DiffuseLighting[pos];
        float4 priorSpecular = u_SpecularLighting[pos];

        // hitT selection: use prior hitT if this lobe is less luminous
        if (luminance(diffuse)  > luminance(priorDiffuse.rgb)  || lightDistance==0)  diffuseHitT  = priorDiffuse.w;
        if (luminance(specular) > luminance(priorSpecular.rgb) || lightDistance==0)  specularHitT = priorSpecular.w;

        diffuse  += priorDiffuse.rgb;   // additive blend
        specular += priorSpecular.rgb;
    }

    // Checkerboard handling when denoiser is off ...

    // NRD front-end packing (only on last pass)
    if (denoiserMode == RELAX) {
        u_DiffuseLighting[pos]  = RELAX_FrontEnd_PackRadianceAndHitDist(diffuse,  diffuseHitT,  sanitize);
        u_SpecularLighting[pos] = RELAX_FrontEnd_PackRadianceAndHitDist(specular, specularHitT, sanitize);
    } else { // REBLUR
        u_DiffuseLighting[pos]  = REBLUR_FrontEnd_PackRadianceAndNormHitDist(diffuse,  diffNormDist,  sanitize);
        u_SpecularLighting[pos] = REBLUR_FrontEnd_PackRadianceAndNormHitDist(specular, specNormDist,  sanitize);
    }
}
```

### hitT Selection Strategy

The hitT selection is crucial for NRD:
- Uses the hit distance from the **more luminous** lobe
- When GI follows DI (isFirstPass=false), GI's lightDistance=0, so DI's hitT is preserved
- This means NRD gets hit distances from direct lighting for both lobes, which is generally better quality



---

## 7. NRD (RELAX / REBLUR) Integration

### 7.1 Texture Mapping

The NrdIntegration maps RTXDI textures to NRD resources ([`NrdIntegration.cpp`](d:/RTXDI/Samples/FullSample/Source/RenderPasses/DenoisingPasses/NrdIntegration.cpp)):

| NRD Resource | RTXDI Texture | Contents |
|-------------|---------------|----------|
| `IN_DIFF_RADIANCE_HITDIST` | `DiffuseLighting` | Packed diffuse radiance + hitT |
| `IN_SPEC_RADIANCE_HITDIST` | `SpecularLighting` | Packed specular radiance + hitT |
| `IN_NORMAL_ROUGHNESS` | `NormalRoughness` | World normal + roughness |
| `IN_VIEWZ` | `Depth` | Linear view-space depth |
| `IN_MV` | `MotionVectors` | Motion vectors |
| `IN_DIFF_CONFIDENCE` | `DiffuseConfidence` | Gradient-based confidence |
| `IN_SPEC_CONFIDENCE` | `SpecularConfidence` | Gradient-based confidence |
| `OUT_DIFF_RADIANCE_HITDIST` | `DenoisedDiffuseLighting` | Denoised output |
| `OUT_SPEC_RADIANCE_HITDIST` | `DenoisedSpecularLighting` | Denoised output |

### 7.2 RELAX Pipeline (GPU-side)

From [`Relax.cpp`](d:/RTXDI/External/NRD/Source/Relax.cpp):
```
CLASSIFY_TILES → HITDIST_RECONSTRUCTION → PREPASS → TEMPORAL_ACCUMULATION
    → HISTORY_FIX → HISTORY_CLAMPING → COPY → ANTI_FIREFLY → ATROUS → SPLIT_SCREEN
```

### 7.3 Checkerboard Mode

When checkerboard rendering is active:
- Noisy inputs (`DiffuseLighting`/`SpecularLighting`) are packed at half horizontal resolution
- NRD's checkerboard mode clips the viewport to the active field
- The compositing pass expands back: `illuminationPos.x /= 2`

---

## 8. CompositingPass — Final Merge

In [`CompositingPass.hlsl`](d:/RTXDI/Samples/FullSample/Shaders/CompositingPass.hlsl):

```hlsl
// 1. Read G-buffer materials
float3 diffuseAlbedo = ...;
float3 specularF0    = ...;

// 2. Read noisy (pre-denoise) illumination
float4 diffuse_illum  = t_Diffuse[pos];   // demodulated diffuse radiance
float4 specular_illum = t_Specular[pos];  // demodulated specular radiance

// 3. If denoising: read denoised output and replace noisy radiance
if (denoiserMode != OFF) {
    float4 denoised_diffuse  = t_DenoisedDiffuse[pos];
    float4 denoised_specular = t_DenoisedSpecular[pos];

    // REBLUR unpacking ...
    diffuse_illum.rgb  = denoised_diffuse.rgb;
    specular_illum.rgb = denoised_specular.rgb;
}

// 4. Re-modulate (apply material)
diffuse_illum.rgb  *= diffuseAlbedo;
specular_illum.rgb *= max(0.01, specularF0);

// 5. Final composite
compositedColor = diffuse_illum.rgb + specular_illum.rgb + emissive.rgb;
```

This is the **material remodulation** step — the reverse of the demodulation done before NRD.

---

## 9. Complete Data Flow Diagram

```mermaid
flowchart TD
    subgraph DI["ReSTIR DI (Direct Lighting)"]
        DI_Sample["Initial Sample Lights"] --> DI_Resample["Temporal + Spatial Resampling"]
        DI_Resample --> DI_Shade["ShadeSurfaceWithLightSample"]
        DI_Shade --> DI_Split["SplitBrdf: demodDiffuse + specular"]
        DI_Split --> DI_Demod["DemodulateSpecular(specular/F0)"]
        DI_Demod --> DI_Store["StoreShadingOutput<br/>isFirstPass=true"]
    end

    subgraph GI["ReSTIR GI (Indirect Lighting)"]
        GI_Trace["BRDF Path Trace"] --> GI_Shade2nd["Shade Secondary Surfaces"]
        GI_Shade2nd --> GI_Reservoir["Build GI Reservoirs"]
        GI_Reservoir --> GI_Resample["Temporal + Spatial Resampling"]
        GI_Resample --> GI_FinalShade["Final Shading: EvaluateBrdf"]
        GI_FinalShade --> GI_Split["SplitBrdf: demodDiffuse + specular"]
        GI_Split --> GI_Demod["DemodulateSpecular(specular/F0)"]
        GI_Demod --> GI_Store["StoreShadingOutput<br/>isFirstPass=false<br/>ADDITIVE to DI data"]
    end

    DI_Store --> DiffSpec["u_DiffuseLighting<br/>u_SpecularLighting"]
    GI_Store --> DiffSpec

    DiffSpec --> NRD_Pack["NRD Front-End Pack<br/>RELAX/REBLUR format"]
    NRD_Pack --> NRD["NRD Denoiser<br/>(RELAX/REBLUR)"]
    NRD --> NRD_Out["DenoisedDiffuseLighting<br/>DenoisedSpecularLighting"]

    NRD_Out --> Comp["CompositingPass"]
    DiffSpec --> Comp

    subgraph Comp_Detail["CompositingPass"]
        Comp_Read["Read noisy + denoised"] --> Comp_Replace["Replace noisy radiance<br/>with denoised"]
        Comp_Replace --> Comp_Remod["Re-modulate:<br/>diffuse *= albedo<br/>specular *= F0"]
        Comp_Remod --> Comp_Final["compositedColor =<br/>diffuse + specular + emissive"]
    end
```

---

## 10. Summary: Key Design Decisions

| Aspect | Design |
|--------|--------|
| **Diffuse/Specular split** | **Always split** at the primary hit. `SplitBrdf` with `demodulatedDiffuse` (Lambert) and `specular` (GGX×Fresnel×NdotL). |
| **Material demodulation** | Diffuse is stored as Lambert radiance (without albedo). Specular is stored as `radiance / specularF0`. Materials are re-applied in compositing. |
| **Multi-pass blending** | DI runs first (isFirstPass=true). GI runs second (isFirstPass=false), **additively** blending into the same textures. |
| **hitT selection** | The more luminous lobe's hitT is preserved. GI contributes zero hitT so DI's hitT is used. |
| **NRD feed** | Two separate NRD input channels: `IN_DIFF_RADIANCE_HITDIST` and `IN_SPEC_RADIANCE_HITDIST`. Packed via `RELAX_FrontEnd_PackRadianceAndHitDist` or `REBLUR_FrontEnd_PackRadianceAndNormHitDist`. |
| **NRD output** | Two separate NRD output channels. The compositing pass replaces noisy radiance with denoised radiance, then re-modulates. |
| **Confidence** | Gradient-based confidence from ReSTIR DI luminance gradients, fed as `IN_DIFF_CONFIDENCE` / `IN_SPEC_CONFIDENCE`. |
| **Checkerboard** | Input textures packed at half-width; NRD handles checkerboard internally; compositing expands back to full width. |

---

## 11. TortureRed Implementation — ReSTIR DI/GI Diffuse-Specular Pipeline

This section documents the **actual implemented** ReSTIR DI + GI diffuse/specular split in TortureRed, including the StoreShadingOutput two-call merge pattern and the unified `FinalDiffuse`/`FinalSpecular` interchange pair.

### 11.1 Render Pipeline Order

From [`Application::Render()`](d:/TortureRed/Sources/Application.cpp):

```
1. Depth Pre-Pass
2. G-Buffer Pass (albedo, normal, material, depth)
3. DispatchRestirDI           ← ReSTIR DI (local lights) + StoreShadingOutput Call 1 → FinalDiffuse/FinalSpecular + NRD trigger (if GI disabled)
4. DispatchRestirGI           ← ReSTIR GI + SHaRC + GI Resolve Intermediates + StoreShadingOutput Call 2 → FinalDiffuse/FinalSpecular + NRD trigger
5. Lighting Pass              ← fullscreen triangle: analytic dir light + FinalDiffuse×diffFactor + FinalSpecular×specFactor (zero branching)
6. TAA (if enabled)
7. Transparency Pass
```

**DI runs first, GI runs second.** This allows the two-call StoreShadingOutput merge: DI writes the base, GI additively blends on top.

**NRD denoising trigger logic:**

| Condition | Who triggers NRD | NRD reads from |
|-----------|-----------------|----------------|
| Both DI + GI enabled | `DispatchRestirGI` calls `NRDDenoise()` after GI StoreShadingOutput Call 2 | `FinalDiffuse`/`FinalSpecular` (DI+GI merged) |
| GI only | `DispatchRestirGI` calls `NRDDenoise()`, GI overwrites Final (isFirstPass=true) | `FinalDiffuse`/`FinalSpecular` (GI only) |
| DI only | `DispatchRestirDI` calls `NRDDenoise()` after DI StoreShadingOutput Call 1 | `FinalDiffuse`/`FinalSpecular` (DI only) |
| Neither | NRD not run | `FinalDiffuse`/`FinalSpecular` may contain stale data — Lighting handles this |

**SSO always runs** when its source pass is active (regardless of NRD state). This eliminates all legacy fallback paths (`DIOutputTex`, `RasterIndirectLightingTex`).

### 11.2 Light Source Breakdown

| Source | Handled by | Goes through |
|--------|-----------|-------------|
| **Main directional light** (index 0) | `Lighting.hlsl` — analytic `EvaluateBSDF(N, V, L)` with ray-traced shadow ray | Nothing — stays in lighting pass **only** |
| **Local lights** (index 1+) | `RestirDI_SplitShade.hlsl` — ReSTIR DI → diffuse/specular intermediates → StoreShadingOutput | Full ReSTIR + NRD pipeline |
| **Indirect (all lights)** | `RestirGI_Diffuse_*` / `RestirGI_Specular_*` — already split diffuse/specular → GI Resolve Intermediates → StoreShadingOutput | Full ReSTIR + NRD pipeline |

### 11.3 Implemented Shaders and Their Roles

#### DI Pipeline

| Shader | File | Role |
|--------|------|------|
| **DI Temporal** | [`RestirDI_Temporal.hlsl`](d:/TortureRed/Sources/Shaders/RestirDI_Temporal.hlsl) | Combined initial sample + temporal resampling → `DIReservoirBuffer[curr]` |
| **DI Spatial** | [`RestirDI_Spatial.hlsl`](d:/TortureRed/Sources/Shaders/RestirDI_Spatial.hlsl) | Spatial resampling → `DIReservoirIntermediate` |
| **DI Split Shade** | [`RestirDI_SplitShade.hlsl`](d:/TortureRed/Sources/Shaders/RestirDI_SplitShade.hlsl) | Per-lobe BSDF × radiance ÷ `NRD_MaterialFactors`, writes raw `float4(radiance, lightDist)` to `DIDiffuseIntermediate` + `DISpecularIntermediate` |

#### GI Pipeline

| Shader | File | Role |
|--------|------|------|
| **SHaRC Update** | [`SHaRC_Update.hlsl`](d:/TortureRed/Sources/Shaders/SHaRC_Update.hlsl) | 5×5 downsampled secondary ray trace → hash table accumulation |
| **SHaRC Resolve** | [`SHaRC_Resolve.hlsl`](d:/TortureRed/Sources/Shaders/SHaRC_Resolve.hlsl) | EMA blend accumulation → resolved radiance cache |
| **Diffuse Temporal** | [`RestirGI_Diffuse_Temporal.hlsl`](d:/TortureRed/Sources/Shaders/RestirGI_Diffuse_Temporal.hlsl) | Temporal ReSTIR + SHaRC query + BSDF |
| **Diffuse Spatial** | [`RestirGI_Diffuse_Spatial.hlsl`](d:/TortureRed/Sources/Shaders/RestirGI_Diffuse_Spatial.hlsl) | 3×3 spatial reuse → `DiffuseReservoirIntermediate` |
| **Specular Temporal** | [`RestirGI_Specular_Temporal.hlsl`](d:/TortureRed/Sources/Shaders/RestirGI_Specular_Temporal.hlsl) | Temporal ReSTIR for specular lobe |
| **Specular Spatial** | [`RestirGI_Specular_Spatial.hlsl`](d:/TortureRed/Sources/Shaders/RestirGI_Specular_Spatial.hlsl) | 3×3 spatial reuse → `SpecularReservoirIntermediate` |
| **GI Resolve Intermediates** | [`RestirGI_ResolveIntermediates.hlsl`](d:/TortureRed/Sources/Shaders/RestirGI_ResolveIntermediates.hlsl) | Resolve `StructuredBuffer<Reservoir>` → raw `float4(radiance, firstBounceHitT)` formatting `GIDiffuseIntermediate` + `GISpecularIntermediate`. Always dispatched when GI active. |

#### NRD Pipeline

| Shader | File | Role |
|--------|------|------|
| **Prepare Guides** | [`NrdPrepareGuides.hlsl`](d:/TortureRed/Sources/Shaders/NrdPrepareGuides.hlsl) | Motion vectors, normal+roughness, viewZ |
| **StoreShadingOutput** | [`NrdStoreShadingOutput.hlsl`](d:/TortureRed/Sources/Shaders/NrdStoreShadingOutput.hlsl) | Generic 2-input/2-output bridge shader. Called twice: after DI (isFirstPass=true, overwrite) and after GI (isFirstPass=(DI was NOT enabled), load+add). Always dispatched when source is active (independent of NRD). Bridges per-source intermediates → `FinalDiffuseTex` / `FinalSpecularTex`. |
| **Pack Noise** | [`NrdPackNoise.hlsl`](d:/TortureRed/Sources/Shaders/NrdPackNoise.hlsl) | Reads raw `float4(radiance, hitT)` from `FinalDiffuse`/`FinalSpecular` → RELAX front-end format `NrdRelaxDiffuseTex` / `NrdRelaxSpecularTex` |
| **NRD Relax** | NRD SDK | Denoise `IN_DIFF_RADIANCE_HITDIST` + `IN_SPEC_RADIANCE_HITDIST` (reads `NrdRelax*` RELAX-packed textures) |
| **Composite Indirect** | [`NrdCompositeIndirect.hlsl`](d:/TortureRed/Sources/Shaders/NrdCompositeIndirect.hlsl) | Unpack NRD denoised output → write back to `FinalDiffuseTex` + `FinalSpecularTex` (circular write-back) |

#### Removed (Legacy)

| Shader | Reason |
|--------|--------|
| `RestirDI_Shade.hlsl` | Legacy combined DI shade → `DIOutputTex`. Replaced by SplitShade + SSO → Final. |
| `RestirGI_Split_Resolve.hlsl` | Legacy combined GI resolve → `RasterIndirectLightingTex`. Replaced by ResolveIntermediates + SSO → Final. |
| `NrdMergeSignals.hlsl` | Legacy cross-cutting DI+GI merge. Replaced by SSO bridge. |
| `NrdPackRasterIndirect.hlsl` | Legacy GI-only NRD pack. Replaced by ResolveIntermediates + SSO + NrdPackNoise. |

### 11.4 Key Design: Two-Call StoreShadingOutput Bridge

[`NrdStoreShadingOutput.hlsl`](d:/TortureRed/Sources/Shaders/NrdStoreShadingOutput.hlsl) is a **source-agnostic** 2-input/2-output shader. Each caller (DI, GI) provides its own Diffuse+Specular intermediate pair. The shader has zero knowledge of which system is calling it.

```hlsl
// NrdStoreShadingOutput.hlsl — Generic bridge shader
//
// Per-call inputs  (Texture2D<float4>, raw radiance + hitT):
//   InputIdx0 = SourceDiffuseIntermediate
//   InputIdx1 = SourceSpecularIntermediate
//
// Per-call outputs (RWTexture2D<float4>):
//   OutputIdx0 = FinalDiffuseTex          ← universal interchange
//   OutputIdx1 = FinalSpecularTex         ← universal interchange
//
// cbuffer StoreShadingOutputCB (b2):
//   isFirstPass — 1 = overwrite, 0 = load + additive blend

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // ... sky pixel early-out ...

    float4 srcDiffuse  = SourceDiffuseIntermediate.Load(screenPos);
    float4 srcSpecular = SourceSpecularIntermediate.Load(screenPos);

    if (isFirstPass)
    {
        // Overwrite: this source (DI or GI-only) sets the base
        FinalDiffuseTex[screenPos]  = srcDiffuse;
        FinalSpecularTex[screenPos] = srcSpecular;
    }
    else
    {
        // Additive blend: load prior (DI contribution in Final) + add this source (GI)
        float4 priorDiffuse  = FinalDiffuseTex[screenPos];   // UAV Load — DI base from Call 1
        float4 priorSpecular = FinalSpecularTex[screenPos];

        float3 outDiffuse  = priorDiffuse.rgb  + srcDiffuse.rgb;
        float3 outSpecular = priorSpecular.rgb + srcSpecular.rgb;

        // hitT: use the brighter contributor's hitT
        float dHitT = priorDiffuse.a, sHitT = priorSpecular.a;
        if (Luminance(srcDiffuse.rgb)  > Luminance(priorDiffuse.rgb))  dHitT = srcDiffuse.a;
        if (Luminance(srcSpecular.rgb) > Luminance(priorSpecular.rgb)) sHitT = srcSpecular.a;

        FinalDiffuseTex[screenPos]  = float4(outDiffuse,  dHitT);
        FinalSpecularTex[screenPos] = float4(outSpecular, sHitT);
    }
}
```

**Dispatch flow from [`Renderer.cpp`](d:/TortureRed/Sources/Renderer.cpp):**

| Call | Dispatched from | Gate | `isFirstPass` | Input pair | Output → |
|------|----------------|------|:---:|---|---|
| **Call 1** | `DispatchRestirDI` | `enableRestirDI != 0` (always when DI active) | `1` | `DIDiffuseIntermediate` + `DISpecularIntermediate` | `FinalDiffuseTex` + `FinalSpecularTex` (overwrite) |
| **Call 2** | `DispatchRestirGI` | `enableRasterIndirectGI != 0` (always when GI active) | `0` if DI ran, `1` if DI was off | `GIDiffuseIntermediate` + `GISpecularIntermediate` | `FinalDiffuseTex` + `FinalSpecularTex` (load+add or overwrite) |

**UAV barrier** between Call 1 and Call 2 on `FinalDiffuseTex` / `FinalSpecularTex` is required (DI writes, GI reads+modifies).

**`isFirstPass` for Call 2 is set at dispatch time** by the C++ side: `const UINT isFirstPass = (frame.enableRestirDI != 0u) ? 0u : 1u;`. When DI is disabled, Final may contain stale previous-frame data, so GI must overwrite (isFirstPass=1).

### 11.5 GI Intermediate Resolve

[`RestirGI_ResolveIntermediates.hlsl`](d:/TortureRed/Sources/Shaders/RestirGI_ResolveIntermediates.hlsl) converts GI reservoir `StructuredBuffer`s into the same uniform `float4(radiance, hitT)` format as DI intermediates. This is what makes the generic StoreShadingOutput shader possible.

It extracts the G-buffer reconstruction + `EvaluateBSDF()` + `NRD_MaterialFactors()` normalization:

```hlsl
// Inputs:  DiffuseReservoirIntermediate   (StructuredBuffer<Reservoir>)
//          SpecularReservoirIntermediate  (StructuredBuffer<Reservoir>)
// Outputs: GIDiffuseIntermediate   (RWTexture2D<float4>: radiance, firstBounceHitT)
//          GISpecularIntermediate  (RWTexture2D<float4>: radiance, firstBounceHitT)

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // 1. Reconstruct surface from G-buffer (albedo, normal, material, depth)
    // 2. Compute NRD_MaterialFactors(diffuseFactor, specularFactor)
    // 3. For diffuse reservoir:
    //      if (rDiffuse.W > 0 && rDiffuse.firstBounceHitT > 0)
    //        EvaluateBSDF(..., diffuseBRDF, specularBRDF)
    //        giDiffuse = diffuseBRDF * rDiffuse.radiance * (rDiffuse.W * NdotL) / diffuseFactor
    //        giDiffuseHitT = rDiffuse.firstBounceHitT
    // 4. Same for specular reservoir
    // 5. Write GIDiffuseIntermediate = float4(giDiffuse, giDiffuseHitT)
    //         GISpecularIntermediate = float4(giSpecular, giSpecularHitT)
}
```

**Always dispatched when GI is active** — regardless of NRD state. SSO always needs the intermediates to bridge to `Final*`. The legacy `SplitResolve` path has been removed.

### 11.6 NRD Pack — Final → RELAX Format

[`NrdPackNoise.hlsl`](d:/TortureRed/Sources/Shaders/NrdPackNoise.hlsl) is the only shader that knows about the NRD RELAX packing format. It reads the merged raw `float4(radiance, hitT)` from `FinalDiffuse`/`FinalSpecular` and packs into `RELAX_FrontEnd_PackRadianceAndHitDist`:

```hlsl
// Reads:  FinalDiffuseTex   (raw float4: radiance, hitT)
//         FinalSpecularTex  (raw float4: radiance, hitT)
// Writes: NrdRelaxDiffuseTex   (RELAX_FrontEnd_PackRadianceAndHitDist)
//         NrdRelaxSpecularTex  (RELAX_FrontEnd_PackRadianceAndHitDist)

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    float4 nDiff = FinalDiffuseTex.Load(screenPos);
    float4 nSpec = FinalSpecularTex.Load(screenPos);
    bool hasHit = (nDiff.a > 0.0f);
    NrdRelaxDiffuseTex[screenPos]  = RELAX_FrontEnd_PackRadianceAndHitDist(nDiff.rgb, nDiff.a, hasHit);
    NrdRelaxSpecularTex[screenPos] = RELAX_FrontEnd_PackRadianceAndHitDist(nSpec.rgb, nSpec.a, hasHit);
}
```

**No `NrdNoise*` intermediary.** `Final*` serves as both the merge target (written by SSO) and the NRD source (read by NrdPackNoise). This cleanly separates concerns: StoreShadingOutput is NRD-format-agnostic.

### 11.7 NRD Pipeline Dispatch Flow

From [`Renderer::NRDDenoise()`](d:/TortureRed/Sources/Renderer.cpp):

```
NRDDenoise():
  1. NrdPrepareGuides → NrdMotionVectorsTex, NrdNormalRoughnessTex, NrdViewZTex
  2. NrdPackNoise      → NrdRelaxDiffuseTex, NrdRelaxSpecularTex       (read Final* → RELAX)
  3. NRD Relax         → NrdDenoisedDiffuseTex, NrdDenoisedSpecularTex   (RELAX denoising)
  4. NrdComposite      → FinalDiffuseTex, FinalSpecularTex              (circular write-back)
```

**Circular write-back**: `Final*` is consumed by NrdPackNoise in step 2, then overwritten with denoised radiance by NrdComposite in step 4. By the time Lighting reads `Final*`, it always contains the correct signal — denoised if NRD ran, raw if it didn't.

### 11.8 Full Pipeline Dispatch Flow (Annotated)

```
Application::Render():
  1. Depth Pre-Pass
  2. G-Buffer Pass

  3. DispatchRestirDI:
     a. DI Temporal → Spatial → DIReservoirIntermediate
     b. DI SplitShade → DIDiffuseIntermediate + DISpecularIntermediate
     c. [always, if enableRestirDI] StoreShadingOutput Call 1 (isFirstPass=1)
          read:  DIDiffuseIntermediate, DISpecularIntermediate
          write: FinalDiffuseTex, FinalSpecularTex   ← overwrite DI base

  ── UAV barrier: FinalDiffuseTex, FinalSpecularTex ──

  4. DispatchRestirGI (if enabled):
     a. SHaRC update + resolve
     b. Diffuse temporal → spatial → DiffuseReservoirIntermediate
     c. Specular temporal → spatial → SpecularReservoirIntermediate
     d. [always] GI Resolve Intermediates → GIDiffuseIntermediate + GISpecularIntermediate
     e. [always] StoreShadingOutput Call 2 (isFirstPass=0 if DI ran)
          read:  GIDiffuseIntermediate, GISpecularIntermediate
          read:  FinalDiffuseTex, FinalSpecularTex  (UAV Load — DI contribution)
          write: FinalDiffuseTex, FinalSpecularTex  ← additive blend or overwrite
     f. [if NRD enabled] NRDDenoise():
        i.   NrdPrepareGuides
        ii.  NrdPackNoise: read Final → write NrdRelaxDiffuseTex + NrdRelaxSpecularTex
        iii. NRD Relax → NrdDenoisedDiffuseTex + NrdDenoisedSpecularTex
        iv.  NrdCompositeIndirect → FinalDiffuseTex + FinalSpecularTex  (denoised, write #3)

  ── UAV barrier (or transition to SRV): FinalDiffuseTex, FinalSpecularTex ──

  5. Lighting Pass:
     MainDir + FinalDiffuse × diffuseFactor + FinalSpecular × specularFactor

  6. TAA
  7. Transparency
```

### 11.9 When DI/GI/Both/Neither Are Active

| Condition | Call 1 (from DI) | Call 2 (from GI) | `FinalDiffuse`/`FinalSpecular` writes |
|-----------|:---:|:---:|:---|
| Both DI + GI | `isFirstPass=1`, overwrites DI base | `isFirstPass=0`, loads DI + adds GI | 1(DI)→2(merged) → optionally 3(denoised by NRD) |
| DI only | `isFirstPass=1`, overwrites | Skipped (GI disabled) | 1(DI) → optionally 3(denoised by NRD) |
| GI only | Skipped (DI disabled) | `isFirstPass=1`, overwrites GI | 2(GI) → optionally 3(denoised by NRD) |
| Neither | Skipped | Skipped | Stale — Lighting only uses ambient |

### 11.10 Lighting Pass — Three-Term Final Composite

[`Lighting.hlsl`](d:/TortureRed/Sources/Shaders/Lighting.hlsl) composes the final lit color from three independent additive terms. **Lighting always reads `FinalDiffuseTex` / `FinalSpecularTex` — zero branching on NRD or DI/GI state.**

| Term | Source | Computation |
|------|--------|-------------|
| **mainDirTerm** | Sun/directional (index 0) | `EvaluateBSDF(N, V, L_main) * lightColor * intensity * NdotL * shadowFactor` — ray-traced shadow ray, no ReSTIR, no NRD |
| **diffuseTerm** | DI local lights + GI indirect, unified through `FinalDiffuseTex` | `FinalDiffuseTex.rgb * diffuseFactor` from `NRD_MaterialFactors()` |
| **specularTerm** | DI local lights + GI indirect, unified through `FinalSpecularTex` | `FinalSpecularTex.rgb * specularFactor` from `NRD_MaterialFactors()` |

`FinalDiffuse`/`FinalSpecular` contain **NRD-normalized radiance** — the same per-source passes (DI SplitShade, GI ResolveIntermediates) have already divided by `diffuseFactor`/`specularFactor`. Lighting re-modulates to recover the final lit color. This works identically whether the content is raw or denoised.

```hlsl
// Term 1: Main directional (analytic, ray-traced shadows)
EvaluateBSDF(N, V, L_main, albedo, metallic, roughness, diff_main, spec_main);
finalColor += (diff_main + spec_main) * mainLight.color * mainLight.intensity * NdotL * shadowFactor;

// Term 2+3: Unified DI+GI (raw or denoised, via FinalDiffuse/FinalSpecular)
if (FrameCB.enableRestirDI || FrameCB.enableRasterIndirectGI)
{
    Texture2D<float4> finalDiffuseTex  = ResourceDescriptorHeap[g_Indices.InputIdx0];
    Texture2D<float4> finalSpecularTex = ResourceDescriptorHeap[g_Indices.InputIdx1];

    float3 indirectDiffuse  = finalDiffuseTex.SampleLevel(g_LinearSampler, input.texCoord, 0).rgb;
    float3 indirectSpecular = finalSpecularTex.SampleLevel(g_LinearSampler, input.texCoord, 0).rgb;

    NRD_MaterialFactors(N, V, albedo, F0, roughness, diffuseFactor, specularFactor);
    finalColor += indirectDiffuse  * diffuseFactor;
    finalColor += indirectSpecular * specularFactor;
}
```

**No more branching** on `nrdActive`, `diMergedIntoNrd`, `DIOutputTex`, or `RasterIndirectLightingTex`. The pipeline ensures `Final*` always contains the correct signal regardless of configuration.

### 11.11 Complete TortureRed Data Flow Diagram

```mermaid
flowchart TD
    subgraph GBuffer["G-Buffer"]
        G_Albedo["Albedo"]; G_Normal["Normal"]
        G_Material["Material (roughness, metallic)"]; G_Depth["Depth"]
    end

    subgraph DI["DispatchRestirDI (Local Lights Only)"]
        DI_Temp["Temporal Resampling"]
        DI_Spatial["Spatial Resampling"]
        DI_SplitShade["Split Shade<br/>BRDF + NRD normalize"]
        DI_Temp --> DI_Spatial --> DI_SplitShade
        DI_SplitShade --> DIDiff["DIDiffuseIntermediate<br/>(raw float4: radiance, lightDist)"]
        DI_SplitShade --> DISpec["DISpecularIntermediate<br/>(raw float4: radiance, lightDist)"]
    end

    subgraph SSO1["StoreShadingOutput Call 1<br/>(always when DI active)"]
        SSO1_Op["isFirstPass=1 → overwrite"]
        DIDiff --> SSO1_Op
        DISpec --> SSO1_Op
    end

    subgraph GI["DispatchRestirGI (Indirect, All Lights)"]
        GI_SHARC["SHaRC Update + Resolve"]
        GI_DiffT["Diffuse Temporal + Spatial"]
        GI_SpecT["Specular Temporal + Spatial"]
        GI_SHARC --> GI_DiffT
        GI_SHARC --> GI_SpecT
        GI_DiffT --> GIDiffRes["DiffuseReservoirIntermediate<br/>(StructuredBuffer)"]
        GI_SpecT --> GISpecRes["SpecularReservoirIntermediate<br/>(StructuredBuffer)"]
        GI_Resolve["GI Resolve Intermediates<br/>BRDF eval + NRD normalize"]
        GIDiffRes --> GI_Resolve
        GISpecRes --> GI_Resolve
        GI_Resolve --> GIDiff["GIDiffuseIntermediate<br/>(raw float4: radiance, hitT)"]
        GI_Resolve --> GISpec["GISpecularIntermediate<br/>(raw float4: radiance, hitT)"]
    end

    subgraph SSO2["StoreShadingOutput Call 2<br/>(always when GI active)"]
        SSO2_Op["isFirstPass=0 → load Final + add GI<br/>(or =1 if DI disabled)"]
        GIDiff --> SSO2_Op
        GISpec --> SSO2_Op
    end

    subgraph FinalPair["FinalDiffuse / FinalSpecular<br/>(universal interchange, 3 writes per frame)"]
        FD["FinalDiffuseTex<br/>R16G16B16A16_FLOAT"]
        FS["FinalSpecularTex<br/>R16G16B16A16_FLOAT"]
    end

    subgraph NRD["NRD Pipeline (if enableNrdRelax)"]
        Pack["NrdPackNoise<br/>read Final → RELAX pack"]
        Pack --> RDF["NrdRelaxDiffuseTex<br/>(RELAX-packed)"]
        Pack --> RSP["NrdRelaxSpecularTex<br/>(RELAX-packed)"]
        Relax["NRD Relax"]
        RDF --> Relax
        RSP --> Relax
        Relax --> DenDiff["NrdDenoisedDiffuseTex"]
        Relax --> DenSpec["NrdDenoisedSpecularTex"]
        Comp["NrdCompositeIndirect<br/>unpack → write back to Final<br/>(circular write-back)"]
        DenDiff --> Comp
        DenSpec --> Comp
    end

    subgraph Lighting["Lighting.hlsl<br/>(zero branching)"]
        L_Dir["Term 1: Analytic Directional<br/>EvaluateBSDF + shadow ray"]
        L_Diff["Term 2: FinalDiffuse × diffuseFactor"]
        L_Spec["Term 3: FinalSpecular × specularFactor"]
        L_Sum["finalColor = T1 + T2 + T3"]
        L_Dir --> L_Sum
        L_Diff --> L_Sum
        L_Spec --> L_Sum
    end

    SSO1_Op --> FD
    SSO1_Op --> FS
    SSO2_Op -- "UAV Load prior + write" --> FD
    SSO2_Op -- "UAV Load prior + write" --> FS
    FD --> Pack
    FS --> Pack
    Comp -- "denoised" --> FD
    Comp -- "denoised" --> FS
    FD --> L_Diff
    FS --> L_Spec

    GBuffer --> DI_Temp
    GBuffer --> DI_Spatial
    GBuffer --> DI_SplitShade
    GBuffer --> GI_SHARC
    GBuffer --> GI_DiffT
    GBuffer --> GI_SpecT
    GBuffer --> GI_Resolve
    GBuffer --> L_Dir
    GBuffer --> L_Diff
    GBuffer --> L_Spec

    style SSO1 fill:#e1f5fe,stroke:#0288d1
    style SSO2 fill:#e1f5fe,stroke:#0288d1
    style FinalPair fill:#a5d6a7,stroke:#2e7d32
    style NRD fill:#f3e5f5,stroke:#7b1fa2
```

### 11.12 Textures Summary

#### Active Textures

| Texture | Format | Written by | Read by |
|---------|--------|-----------|---------|
| `DIReservoirBuffer[2]` | `StructuredBuffer<DIRreservoir>` | DI Temporal (swap) | DI Spatial |
| `DIReservoirIntermediate` | `StructuredBuffer<DIRreservoir>` | DI Spatial | DI Split Shade |
| `DIDiffuseIntermediate` | `R16G16B16A16_FLOAT` raw | DI Split Shade | StoreShadingOutput Call 1 |
| `DISpecularIntermediate` | `R16G16B16A16_FLOAT` raw | DI Split Shade | StoreShadingOutput Call 1 |
| `DiffuseReservoirBuffer[2]` | `StructuredBuffer<Reservoir>` | Diffuse Temporal (swap) | Diffuse Spatial |
| `SpecularReservoirBuffer[2]` | `StructuredBuffer<Reservoir>` | Specular Temporal/Resolve | Specular Spatial |
| `DiffuseReservoirIntermediate` | `StructuredBuffer<Reservoir>` | Diffuse Spatial | GI Resolve Intermediates |
| `SpecularReservoirIntermediate` | `StructuredBuffer<Reservoir>` | Specular Spatial | GI Resolve Intermediates |
| `GIDiffuseIntermediate` | `R16G16B16A16_FLOAT` raw | GI Resolve Intermediates | StoreShadingOutput Call 2 |
| `GISpecularIntermediate` | `R16G16B16A16_FLOAT` raw | GI Resolve Intermediates | StoreShadingOutput Call 2 |
| **`FinalDiffuseTex`** | `R16G16B16A16_FLOAT` raw | SSO Call 1+2, NrdComposite | NrdPackNoise, Lighting.hlsl |
| **`FinalSpecularTex`** | `R16G16B16A16_FLOAT` raw | SSO Call 1+2, NrdComposite | NrdPackNoise, Lighting.hlsl |
| `NrdRelaxDiffuseTex` | `R16G16B16A16_FLOAT` RELAX | NrdPackNoise | NRD Relax |
| `NrdRelaxSpecularTex` | `R16G16B16A16_FLOAT` RELAX | NrdPackNoise | NRD Relax |
| `NrdMotionVectorsTex` | `R16G16_FLOAT` | NrdPrepareGuides | NRD Relax |
| `NrdNormalRoughnessTex` | `R10G10B10A2_UNORM` | NrdPrepareGuides | NRD Relax |
| `NrdViewZTex` | `R32_FLOAT` | NrdPrepareGuides | NRD Relax |
| `NrdDenoisedDiffuseTex` | `R16G16B16A16_FLOAT` | NRD Relax | NrdCompositeIndirect |
| `NrdDenoisedSpecularTex` | `R16G16B16A16_FLOAT` | NRD Relax | NrdCompositeIndirect |

**`FinalDiffuseTex` / `FinalSpecularTex`** are the universal interchange pair — written by three passes (SSO×2 + NrdComposite) and read by two (NrdPackNoise + Lighting). No separate `NrdNoise*`, `NrdUnpacked*`, `DIOutputTex`, or `RasterIndirectLightingTex` textures.

#### Removed Textures

| Texture | Reason |
|---------|--------|
| `NrdNoiseDiffuseTex` | Replaced by NrdPackNoise reading `FinalDiffuseTex` directly |
| `NrdNoiseSpecularTex` | Replaced by NrdPackNoise reading `FinalSpecularTex` directly |
| `NrdUnpackedDiffuseTex` | Replaced by NrdComposite writing back to `FinalDiffuseTex` |
| `NrdUnpackedSpecularTex` | Replaced by NrdComposite writing back to `FinalSpecularTex` |
| `DIOutputTex` | Legacy DI shade path — replaced by SplitShade + SSO → Final |
| `RasterIndirectLightingTex` | Legacy GI resolve path — replaced by ResolveIntermediates + SSO → Final |
| `NrdNoisyDiffuseTex` | Legacy RELAX-packed merge output |
| `NrdNoisySpecularTex` | Legacy RELAX-packed merge output |

### 11.13 Key Differences from RTXDI

| Aspect | RTXDI FullSample | TortureRed |
|--------|-----------------|-----------|
| **StoreShadingOutput** | Single in-shader function, called per pass (DI then GI) | **Dedicated compute shader** dispatched twice, source-agnostic 2-input/2-output bridge to `FinalDiffuse`/`FinalSpecular` |
| **Noise output format** | RELAX/REBLUR-packed directly in StoreShadingOutput | **Raw float4** → `FinalDiffuse`/`FinalSpecular` → separate `NrdPackNoise` step converts to RELAX |
| **NRD output** | Uses separate `NrdUnpacked*` textures for denoised result | **Circular write-back** to `Final*` (NrdComposite overwrites Final with denoised radiance) |
| **DI sample payload** | Light index + UV (`RTXDI_DIReservoir`) | Light index + hit position (custom `DIRreservoir`) |
| **DI sampling** | `RTXDI_StreamSample` / `RTXDI_CombineDIReservoirs` (RIS) | Custom RIS implementation |
| **DI visibility** | Reused across frames (`packedVisibility`, `age`) | Fresh shadow ray per frame |
| **GI radiance source** | Second-bounce surface (path traced) | SHaRC radiance cache (probe-based interpolation) |
| **GI reservoir type** | `RTXDI_GIReservoir` (position + normal + radiance) | Custom `Reservoir` struct (hitPos + radiance + W + firstBounceHitT) |
| **GI → intermediate resolve** | BRDF eval in StoreShadingOutput function | **Separate `RestirGI_ResolveIntermediates` pass** before StoreShadingOutput |
| **NRD format** | RELAX or REBLUR (selectable) | RELAX only |
| **Confidence inputs** | Gradient-based `DiffuseConfidence` / `SpecularConfidence` | Not used |
| **Checkerboard** | Half-width interleaved rendering | Not implemented |
| **PSR** | Primary Surface Replacement for mirrors | Not implemented |
| **MIS (GI)** | Roughened BRDF MIS for low-roughness surfaces | Not implemented |
| **Main directional** | Sampled via ReSTIR DI with all other lights | Excluded from ReSTIR, shaded analytically |
| **Lighting branching** | `nrdActive`, `diMergedIntoNrd` checks | **Zero branching** — always reads `FinalDiffuse`/`FinalSpecular` |

---

## 12. 🔧 TODO: Restore SHaRC Debug Visualization

> **Status**: `m_SharcDebugPSO` is compiled (line 239) but never dispatched. The debug shader targets the now-removed `RasterIndirectLightingTex`. Three fixes needed.

### 12.1 Root Cause

| Problem | Detail |
|---------|--------|
| **No dispatch** | `m_SharcDebugPSO` created at [Renderer.cpp:239](d:/TortureRed/Sources/Renderer.cpp:239), never dispatched. `sharcDebug` is only used to gate NRD at line 1738 (`&& frame.sharcDebug == 0`). |
| **Wrong output target** | [SHaRC_Debug.hlsl](d:/TortureRed/Sources/Shaders/SHaRC_Debug.hlsl) writes to `OutputIdx0` only. Its comment says "must point to `RasterIndirectLightingTex`" — removed in §11. |
| **Only one output** | `Lighting.hlsl` reads **two** textures (`FinalDiffuseTex` × diffuseFactor + `FinalSpecularTex` × specularFactor). The debug shader must fill both to appear in the final composite. |

### 12.2 Three Changes Required

| # | File | Change |
|---|------|--------|
| **1** | `Renderer.cpp` — `DispatchRestirGI` | Insert dispatch block **after** SSO Call 2 but **before** NRD. When `sharcDebug != 0` and GI is active: run `m_SharcDebugPSO`, write to `FinalDiffuseTex` + `FinalSpecularTex`, UAV-barrier, transition to SRV, **early-return** (skip NRD). |
| **2** | `SHaRC_Debug.hlsl` | Write to **both** `OutputIdx0` and `OutputIdx1`. Both receive the same debug color so `Lighting.hlsl` renders it via either `diffuseFactor` or `specularFactor`. Update header comment. |
| **3** | (none needed) | `m_SharcDebugPSO` already compiled. `m_SharcIndices` already populated. No new allocations required — `FinalDiffuseTex` / `FinalSpecularTex` already exist as universal interchange. |

### 12.3 SHaRC Debug Dispatch Pseudocode

```cpp
// In DispatchRestirGI, after SSO Call 2 and its UAV barrier on Final*,
// BEFORE the useNrd / NRDDenoise block:

if (frame.sharcDebug != 0 && m_SharcDebugPSO)
{
    // Final* already in UAV state from SSO Call 2 — reuse it.
    // NrdPackNoise and NRD are skipped; debug visualization replaces the normal GI output.

    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    indices = {};
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

    // Transition Final* to SRV so Lighting.hlsl can sample the debug visualization
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // Skip NRD (already gated by sharcDebug==0) and avoid the normal SRV-transition path
    m_CurrentReservoirIndex = previousReservoir;
    return;
}
```

**Placement**: insert **immediately after** the SSO Call 2 UAV barrier (line ~1796) and **before** the `if (useNrd && NRDDenoise(frame))` block (line ~1800). The existing `useNrd` already includes `&& frame.sharcDebug == 0` so NRD won't fire, but we need the **early return** to avoid the "NRD disabled" transition path that follows.

### 12.4 SHaRC Debug Shader Changes

Current shader writes only `OutputIdx0`:
```hlsl
RWTexture2D<float4> outDebug = ResourceDescriptorHeap[g_Indices.OutputIdx0];
// ...
outDebug[screenPos] = float4(debugColor, 1.0f);
```

Updated to write both outputs:
```hlsl
RWTexture2D<float4> outDiffuse  = ResourceDescriptorHeap[g_Indices.OutputIdx0]; // FinalDiffuseTex
RWTexture2D<float4> outSpecular = ResourceDescriptorHeap[g_Indices.OutputIdx1]; // FinalSpecularTex
// ...
float4 result = float4(debugColor, 1.0f);
outDiffuse[screenPos]  = result;
outSpecular[screenPos] = result;
```

Both diffuse and specular receive the same debug color. `Lighting.hlsl` will multiply by `diffuseFactor` for one term and `specularFactor` for the other, then sum them. The resulting visible color will be `debugColor * (diffuseFactor + specularFactor)` — close enough to the intended debug visualization.

### 12.5 Pipeline Flow — SHaRC Debug Active

```mermaid
flowchart TD
    subgraph DI["DispatchRestirDI"]
        DI_Temp["Temporal → Spatial → SplitShade"]
        DI_SSO["SSO Call 1 → FinalDiffuse/FinalSpecular<br/>(overwrite DI base)"]
        DI_Temp --> DI_SSO
    end

    subgraph GI["DispatchRestirGI<br/>(sharcDebug != 0)"]
        GI_SharcU["SHaRC Update"]
        GI_SharcR["SHaRC Resolve"]
        GI_DiffT["Diffuse Temporal + Spatial"]
        GI_SpecT["Specular Temporal + Spatial"]
        GI_SharcU --> GI_SharcR --> GI_DiffT
        GI_SharcR --> GI_SpecT

        GI_Skip["⚠️ GI Resolve Intermediates<br/>AND SSO Call 2<br/>STILL RUN<br/>(need intermediates for debug<br/>if DI+GI merge is desired,<br/>or SKIP if debug-only)"]
        GI_DiffT --> GI_Skip
        GI_SpecT --> GI_Skip

        GI_Debug["🟢 SHaRC Debug PSO<br/>sharcDebug=1: SHaRC Output<br/>sharcDebug=2: Bounce Heatmap<br/>→ FinalDiffuseTex<br/>→ FinalSpecularTex<br/>(same debug color to both)"]
    end

    subgraph FinalPair["FinalDiffuse / FinalSpecular<br/>(overwritten by debug PSO)"]
        FD["FinalDiffuseTex"]
        FS["FinalSpecularTex"]
    end

    subgraph Lighting["Lighting.hlsl"]
        L_Diff["Term 2: FinalDiffuse × diffuseFactor"]
        L_Spec["Term 3: FinalSpecular × specularFactor"]
        L_Sum["finalColor =<br/>dirLight + Σ(debugColor × factors)"]
        L_Diff --> L_Sum
        L_Spec --> L_Sum
    end

    DI_SSO --> FD
    DI_SSO --> FS
    GI_Skip -- "may write to" --> FD
    GI_Skip -- "may write to" --> FS
    GI_Debug -- "overwrite" --> FD
    GI_Debug -- "overwrite" --> FS
    FD --> L_Diff
    FS --> L_Spec

    style DI_SSO fill:#e1f5fe,stroke:#0288d1
    style GI_Debug fill:#ffecb3,stroke:#ff8f00
    style GI_Skip fill:#fff3e0,stroke:#e65100
    style FinalPair fill:#a5d6a7,stroke:#2e7d32
    style Lighting fill:#e8f5e9,stroke:#388e3c

    NRD_Skip["🚫 NRD disabled<br/>(useNrd = false when<br/>sharcDebug != 0)"]
    style NRD_Skip fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray: 5 5
```

### 12.6 Decision: Run or Skip GI Resolve + SSO Call 2 Before Debug?

| Option | Effect | Simpler? |
|--------|--------|:---:|
| **A: Skip** GI ResolveIntermediates + SSO Call 2 when debug active | Debug shader replaces entire GI output. DI contribution (if any) still present from SSO Call 1. | ✅ Simpler |
| **B: Run** then overwrite with debug | Extra work that gets immediately overwritten. No benefit. | ❌ |

**Recommendation: Option A** — insert an early-return check at the very top of `DispatchRestirGI`, before SHaRC Update even runs:

```cpp
// At the TOP of DispatchRestirGI:
if (frame.sharcDebug != 0 && m_SharcDebugPSO)
{
    // Run SHaRC update+resolve (needed for the debug shader to query),
    // then run debug PSO → Final*, skip everything else.
    DispatchSharcUpdateAndResolve();       // Pass 1+2 only
    DispatchSharcDebugToFinal();           // Debug PSO
    m_CurrentReservoirIndex = previousReservoir;
    return;
}
```

But this means **duplicating** the SHaRC Update+Resolve code. The cleanest approach is to put the debug dispatch **after** SHaRC Resolve but **before** the ReSTIR temporal passes:

```cpp
// After SHaRC Resolve barrier (line ~1662):
if (frame.sharcDebug != 0 && m_SharcDebugPSO)
{
    // SHaRC cache is fresh. Run debug visualization directly.
    // Skip all ReSTIR passes, GI intermediate resolve, SSO, and NRD.
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(m_CommandList.Get(), m_FinalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // ... dispatch debug PSO ...
    // ... barrier + transition to SRV ...
    m_CurrentReservoirIndex = previousReservoir;
    return;
}
```

**Why this placement wins:**
- SHaRC Update + Resolve already ran — debug shader can query the cache
- Skips expensive ReSTIR temporal/spatial passes (waste of GPU time when debug is active)
- DI contribution (SSO Call 1) already in `Final*` is preserved unless debug overwrites it — the debug color replaces both
- Minimal code duplication

### 12.7 Implementation Checklist

| Step | Action | Effort |
|------|--------|:---:|
| 1 | Insert early-return debug dispatch block after SHaRC Resolve barrier in `DispatchRestirGI` | Small |
| 2 | In debug dispatch: bind `FinalDiffuseTex` as `OutputIdx0`, `FinalSpecularTex` as `OutputIdx1` | Small |
| 3 | Update `SHaRC_Debug.hlsl` to write to both `OutputIdx0` and `OutputIdx1` | Small |
| 4 | Update `SHaRC_Debug.hlsl` header comment — remove `RasterIndirectLightingTex` reference | Trivial |
| 5 | Test: enable GI, set `sharcDebug=1`, verify SHaRC voxel visualization appears | — |
| 6 | Test: enable GI, set `sharcDebug=2`, verify bounce heatmap appears | — |
| 7 | Test: DI-only mode — debug should still be reachable if GI is enabled, unaffected if GI disabled | — |
