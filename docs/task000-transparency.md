# TortureRed transparency pass

_Comprehensive analysis of the forward alpha-blended rendering path for `AlphaMode::Blend` geometry — model loading, PSO setup, shader design, and known issues._

> **Parent document:** [RenderPipeline.md](RenderPipeline.md) — this document is a deep-dive extracted from the main pipeline reference.
>
> **Related task:** DI/GI bypass fix (see [Known issue: DI/GI bypass](#known-issue-digi-bypass-in-transparent-pass) below)

---

## 📋 Table of contents

- [Pipeline position](#pipeline-position)
- [GPU data flow](#gpu-data-flow)
- [Model loading: AlphaMode classification](#model-loading-alphamode-classification)
- [Where transparent objects are NOT rendered](#where-transparent-objects-are-not-rendered)
- [Transparent PSO setup](#transparent-pso-setup)
- [Forward.hlsl — transparent pixel shader](#forwardhlsl--transparent-pixel-shader)
- [Dispatch sequence](#dispatch-sequence)
- [Resolution considerations](#resolution-considerations)
- [Reference: RTXDI FullSample transparent pipeline](#reference-rtxdi-fullsample-transparent-pipeline)
- [Known issue: DI/GI bypass in transparent pass](#known-issue-digi-bypass-in-transparent-pass)

---

## 📍 Pipeline position

```
Depth Pre-pass → G-Buffer → ReSTIR DI → ReSTIR GI → Lighting → Transparency → TAA → ImGui
                                                                     ↑
                                                    TAA path:  renders into RasterHdrOutputTex
                                                               at internal resolution (HDR, no tonemap)
                                                    Non-TAA:   renders directly to back buffer
                                                               at output resolution (LDR, with tonemap)
```

The transparency pass runs **before TAA** so that transparent geometry is temporally accumulated along with the opaque frame. In the TAA path it renders into `RasterHdrOutputTex` (`R16G16B16A16_FLOAT`) at internal resolution; TAA then resolves the combined opaque+transparent HDR image and copies to the back buffer. In the non-TAA path it renders directly to the back buffer at output resolution.

## 📊 GPU data flow

```
GLTF Model Loading                          GPU Memory Layout
─────────────────                          ──────────────────
cgltf_alpha_mode_blend ──→ AlphaMode::Blend    ┌─────────────────────────┐
                                                  │ m_TransparentCommands   │
All other primitives    ──→ opaque/mask          │   (IndirectDrawCommand[])│
                                                  │ m_TransparentCommandBuf │
At Draw-Time:                                     │   (GPU DEFAULT heap)    │
  m_Model.Render(AlphaMode::Blend)                └──────────┬──────────────┘
    → ExecuteIndirect(m_TransparentCommandBuf)               │
                                                             ▼
                                                  [Forward.hlsl]
                                                  Vertex Shader: vertex pulling
                                                  Pixel Shader: PBR lighting
                                                  Output: Back buffer (alpha-blended)
```

## 📦 Model loading: AlphaMode classification

During GLTF parsing in [`Model::LoadGLTFModel()`](../Sources/Model.cpp), each primitive's material `alpha_mode` is read from cgltf:

| cgltf value | `AlphaMode` | Destination buffer |
|-------------|-------------|-------------------|
| `cgltf_alpha_mode_opaque` | `Opaque` | `m_OpaqueCommands` |
| `cgltf_alpha_mode_mask`   | `Mask`   | `m_OpaqueCommands` |
| `cgltf_alpha_mode_blend`  | `Blend`  | `m_TransparentCommands` |

This partitioning happens at load time during indirect command buffer generation in [`Model::CreateGLTFResources()`](../Sources/Model.cpp):

```cpp
if (prim.alphaMode == AlphaMode::Opaque || prim.alphaMode == AlphaMode::Mask)
    m_OpaqueCommands.push_back(cmd);
else
    m_TransparentCommands.push_back(cmd);
```

Both command buffers are uploaded to **GPU DEFAULT heaps** and transitioned to `D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT` for `ExecuteIndirect`.

## 🚫 Where transparent objects are NOT rendered

| Pass | Renders `AlphaMode::Blend`? | Reason |
|------|:---:|--------|
| Depth pre-pass | ❌ No | Only calls `m_Model.Render(AlphaMode::Opaque)` |
| G-Buffer pass | ❌ No | Only calls `m_Model.Render(AlphaMode::Opaque)` |
| ReSTIR DI (compute) | ❌ No | Operates on G-Buffer pixels only |
| ReSTIR GI (compute) | ❌ No | Operates on G-Buffer pixels only |
| Lighting pass | ❌ No | Fullscreen triangle, reads G-Buffer |
| TAA | ❌ No | Processes opaque lighting output |

Transparent objects **only appear** in their dedicated forward pass.

## ⚙️ Transparent PSO setup

Created in [`Renderer::CreatePipelineStates()`](../Sources/Renderer.cpp) using [`Forward.hlsl`](../Sources/Shaders/Forward.hlsl):

| State | Value | Purpose |
|-------|-------|---------|
| **VS** | `Forward.hlsl` / `VSMain` | Vertex pulling from global buffers |
| **PS** | `Forward.hlsl` / `PSMain` | Full PBR lighting (see below) |
| **Blend** | `SrcBlend=SRC_ALPHA, DestBlend=INV_SRC_ALPHA, BlendOp=ADD` | Standard alpha blending |
| **Cull** | `D3D12_CULL_MODE_NONE` | Double-sided rendering for thin transparent surfaces |
| **Depth** | `DepthEnable=TRUE, DepthWriteMask=ZERO, DepthFunc=LESS_EQUAL` | Read-only depth test against G-Buffer depth |
| **Viewport** | TAA path: internal res (`m_InternalWidth × m_InternalHeight`); Non-TAA: output res (`m_OutputWidth × m_OutputHeight`) | Matches the render target |
| **RTV** | TAA path: `R16G16B16A16_FLOAT` (`RasterHdrOutputTex`); Non-TAA: `R8G8B8A8_UNORM` (back buffer) | PSO format must match |
| **PSO** | TAA path: `m_TransparentHdrPSO`; Non-TAA: `m_TransparentPSO` | HDR PSO skips tonemapping |
| **DSV** | `D32_FLOAT` | Matches G-Buffer depth format |

- **Two PSOs**: `m_TransparentPSO` (LDR, `R8G8B8A8_UNORM`) for the non-TAA path and `m_TransparentHdrPSO` (HDR, `R16G16B16A16_FLOAT`) for the TAA path. Both use the same `Forward.hlsl` shader; the shader checks `FrameCB.taaEnabled` to decide whether to tonemap (matching the pattern in `Lighting.hlsl`)
- **Depth write disabled**: prevents transparent objects from occluding each other incorrectly when rendering order is not sorted
- **Double-sided**: most transparent geometry (glass, foliage) is visible from both sides
- **TAA integration**: in the TAA path, transparency is rendered into `RasterHdrOutputTex` before TAA runs, so blended geometry is temporally accumulated

## 🔍 Forward.hlsl — transparent pixel shader

The shader in [`Forward.hlsl`](../Sources/Shaders/Forward.hlsl) performs a **simplified forward PBR lighting** pass independently of the deferred pipeline:

1. **Alpha discard**: pixels with `albedo.a < 0.01` are discarded (avoids blending near-zero alpha pixels that would be invisible anyway)
2. **Material sampling**: base color texture, normal map, metallic-roughness map
3. **Directional light (index 0)**: analytic `EvaluateBSDF` with the main directional light. **No shadow rays** — shadow factor is always 1.0 for transparent objects (performance trade-off)
4. **Local lights (index 1+)**: loops over all point/spot lights with attenuation (`1/(1 + 0.1*d + 0.01*d²)`) and spot-angle smoothstep falloff
5. **Ambient**: constant `0.03 * albedo` ambient term
6. **Tonemapping**: Reinhard (`exposed / (exposed + 1)`) applied in-shader
7. **Output**: `float4(ldrColor, albedo.a)` — RGB is tonemapped, alpha preserved for blending

**Notable omissions vs opaque Lighting.hlsl:**
- No ReSTIR DI/GI integration (no `FinalDiffuseTex` / `FinalSpecularTex` sampling) — **see [Known issue: DI/GI bypass](#known-issue-digi-bypass-in-transparent-pass) below**
- No ray-traced shadows (shadow factor always 1.0 for main directional light)
- No NRD material factors (needed for `FinalDiffuseTex`/`FinalSpecularTex` remodulation)
- No local-light importance sampling (simple analytic loop with distance attenuation only)

## 🔄 Dispatch sequence

From [`Application.cpp`](../Sources/Application.cpp):

```
// 1. Transition G-Buffer depth to DEPTH_READ (shared with opaque pipeline)
TransitionResource(gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_READ);

// 2a. TAA path: render into RasterHdrOutputTex at internal resolution
//     Viewport is already at internal resolution from the top of Render()
if (rasterTaaActive)
{
    TransitionResource(m_Renderer.GetRasterHdrOutputTex(), D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmdList->OMSetRenderTargets(1, &hdrRtvHandle, FALSE, &dsvHandle);
    cmdList->SetPipelineState(m_Renderer.GetTransparentHdrPSO());
}
// 2b. Non-TAA path: render directly to back buffer at output resolution
else
{
    cmdList->RSSetViewports(1, &outputViewport); // switch to output resolution
    TransitionBackBuffer(D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
    cmdList->SetPipelineState(m_Renderer.GetTransparentPSO());
}

// 3. Execute indirect draw for all AlphaMode::Blend primitives
m_Model.Render(cmdList, &m_Renderer, frustum, AlphaMode::Blend);

// 4. TAA path: run TAA on the combined opaque+transparent HDR image
if (rasterTaaActive)
{
    m_Renderer.GenerateMotionVectors(m_FrameConstants);
    m_Renderer.DispatchNaiveTsr(m_FrameConstants, m_Renderer.GetRasterHdrOutputTex());
    m_Renderer.CopyTextureToBackBuffer(m_Renderer.GetTaaOutputTex());
}
```

The `Model::Render()` call with `AlphaMode::Blend` selects `m_TransparentCommandBuffer` and issues `ExecuteIndirect` for all `AlphaMode::Blend` primitives:

```cpp
if (mode == AlphaMode::Opaque || mode == AlphaMode::Mask)
    cmdBuffer = m_OpaqueCommandBuffer;   // ← used by G-Buffer & depth pre-pass
else
    cmdBuffer = m_TransparentCommandBuffer;  // ← used by transparency pass
```

## 📌 Resolution considerations

| Aspect | Value | Reason |
|--------|-------|--------|
| **TAA path viewport** | Internal res (`m_InternalWidth × m_InternalHeight`) | Matches `RasterHdrOutputTex` and G-buffer depth |
| **Non-TAA path viewport** | Output res (`m_OutputWidth × m_OutputHeight`) | Matches back buffer |
| **G-Buffer depth** | Internal res (may be lower with TAAU) | Depth test compares internal-res fragments against internal-res depth — no mismatch in TAA path |
| **Back buffer** | Output resolution | Matches non-TAA viewport |
| **TAA path** | Before TAA resolve | Blended into `RasterHdrOutputTex`, then TAA accumulates the combined result |
| **Non-TAA path** | After direct-to-backbuffer lighting | Blended on top of the raw lighting output |

## 📚 Reference: RTXDI FullSample transparent pipeline

The [RTXDI FullSample](https://github.com/NVIDIAGameWorks/RTXDI) (NVIDIA's reference implementation for ReSTIR DI/GI) is the closest architectural reference to TortureRed. It uses the same Donut framework, the same NRD denoiser integration, and the same ReSTIR DI/GI pipeline — but it has a fully worked-out solution for transparent/glass geometry that TortureRed's current `Forward.hlsl` does not replicate.

### RTXDI FullSample pipeline order (from `SceneRenderer.cpp`)

```
UpdateAccelerationStructure()     ← TLAS rebuild for animated geometry
    ↓
EnvironmentMap()                  ← procedural sky or HDR env map
    ↓
GBuffer()                         ← ray-traced or rasterized G-buffer
    │   Opaque:  INSTANCE_MASK_OPAQUE  → committed hit → write G-buffer
    │   AlphaTested: INSTANCE_MASK_ALPHA_TESTED → any-hit alpha test → write G-buffer
    │   Transmissive: INSTANCE_MASK_TRANSPARENT → any-hit records minGlassRayT only
    │                                              (does NOT write G-buffer)
    ↓
PrepareLights() + PresampleLights()   ← ReGIR cell build, RIS presampling
    ↓
RenderDirectLighting()            ← ReSTIR DI: initial sample → temporal → spatial → shade
    ↓
RenderIndirectLighting()          ← ReSTIR GI / ReSTIR PT / BRDF path tracing
    ↓
Denoiser()                        ← NRD REBLUR or RELAX on DiffuseLighting + SpecularLighting
    ↓
m_compositingPass->Render()       ← CompositingPass.hlsl:
    │   diffuse_illumination * diffuseAlbedo
    │ + specular_illumination * specularF0
    │ + emissive
    │ → writes HdrColor
    ↓
TransparentGeometry()             ← GlassPass.hlsl (ray-traced, up to 8 bounces)
    │   reads HdrColor (already has full DI+GI for opaque background)
    │   ray-traces through INSTANCE_MASK_TRANSPARENT surfaces
    │   applies Fresnel reflection + transmission per layer
    │   composites result back into HdrColor
    ↓
ResolveAA()                       ← TAA / DLSS / accumulation
    ↓
Bloom() → ToneMapping() → FinalFramebufferOutput()
```

### Material domain taxonomy

RTXDI uses a richer material domain system than TortureRed's simple `AlphaMode::Opaque/Mask/Blend`:

| Domain | Instance mask bit | G-buffer | GlassPass |
|--------|:-----------------:|:--------:|:---------:|
| `MaterialDomain_Opaque` | `INSTANCE_MASK_OPAQUE` (0x01) | ✅ Written | Only if mirror (roughness ≤ 0.01) |
| `MaterialDomain_AlphaTested` | `INSTANCE_MASK_ALPHA_TESTED` (0x02) | ✅ Written (discard if `opacity < alphaCutoff`) | ❌ |
| `MaterialDomain_AlphaBlended` | `INSTANCE_MASK_ALPHA_TESTED` (0x02) | ✅ Written (`clip(opacity - 0.5)` — no real blending) | ❌ |
| `MaterialDomain_Transmissive` | `INSTANCE_MASK_TRANSPARENT` (0x04) | ❌ Skipped | ✅ Full ray-traced glass |
| `MaterialDomain_TransmissiveAlphaTested` | `INSTANCE_MASK_TRANSPARENT` (0x04) | ❌ Skipped | ✅ (if `opacity >= alphaCutoff`) |
| `MaterialDomain_TransmissiveAlphaBlended` | `INSTANCE_MASK_TRANSPARENT` (0x04) | ❌ Skipped | ✅ (opacity-weighted) |

**Key insight**: RTXDI does not use rasterized alpha-blending at all. `AlphaBlended` materials are treated as opaque with a hard `clip(opacity - 0.5)` threshold. True transparency is handled exclusively by the ray-traced `GlassPass`.

### How GlassPass integrates with the full DI+GI pipeline

This is the critical architectural difference from TortureRed's `Forward.hlsl`.

**GlassPass runs AFTER `CompositingPass`**, which means `HdrColor` already contains the fully denoised ReSTIR DI + ReSTIR GI result for all opaque/masked surfaces. The glass pass then:

1. **Reads `HdrColor`** — the already-lit opaque background
2. **Ray-traces through up to 8 transparent layers** using `INSTANCE_MASK_TRANSPARENT`
3. **For each transmissive surface**, computes Fresnel reflectance and traces a secondary reflection ray into `INSTANCE_MASK_OPAQUE` geometry
4. **Accumulates** `overlay += secondaryRadiance * fresnel + emissive` and `throughput *= transmission * (1 - fresnel) * baseColor`
5. **Composites** back: `HdrColor = HdrColor * throughput + overlay`

The secondary reflection ray (`getSecondaryRadiance`) hits opaque geometry and returns `emissive + diffuseAlbedo * ambient` — a simplified shading that does **not** re-run ReSTIR DI/GI for the reflected surface. This is a deliberate performance trade-off: reflections in glass are approximated, not fully re-lit.

```hlsl
// GlassPass.hlsl — final composite
if (any(throughput < 1.0) || any(overlay > 0.0))
{
    float4 previousColor = u_CompositedColor[pixelPosition]; // ← HdrColor after CompositingPass
    float metalness = getMetalness(diffuseAlbedo, specularF0); // from G-buffer
    float3 newColor = previousColor.rgb * (lerp(1.f, throughput, (firstBounceGlass ? 1.f : metalness))) + overlay;
    u_CompositedColor[pixelPosition] = float4(newColor.rgb, previousColor.a);
}
```

Note: the `metalness` modulation on `throughput` prevents glass from attenuating the underlying metal surface's specular highlight — a subtle but important correctness fix.

### G-buffer: transparent geometry records `minGlassRayT`

In `RaytracedGBuffer.hlsl`, when a transparent surface is encountered in the any-hit shader, it does **not** write G-buffer data — instead it records the closest transparent hit distance in `payload.minGlassRayT`. This value is stored in `u_Emissive[pixelPosition].w` (the alpha channel of the emissive texture):

```hlsl
// RaytracedGBuffer.hlsl
const float hitT = payload.committedRayT;       // opaque surface depth
const bool hasGlass = payload.minGlassRayT < hitT;
const float maxGlassHitT = hasGlass ? hitT : 0;
// ...
u_Emissive[pixelPosition] = float4(ms.emissiveColor, maxGlassHitT);
```

This `maxGlassHitT` value tells `GlassPass` how far to trace before stopping at the opaque background — it prevents glass from being rendered behind opaque surfaces.

In the rasterized G-buffer path (`RasterizedGBuffer.hlsl`), the same information is stored as `viewDistance` in `o_emissive.w` for all pixels, enabling glass ray tracing on all pixels.

### TAA integration

The `GlassPass` runs **before** `ResolveAA()` (TAA/DLSS), so glass geometry **is** temporally accumulated. This is a major advantage over TortureRed's current approach where the transparent pass runs after TAA and receives no temporal filtering.

### Comparison: TortureRed vs RTXDI FullSample

| Aspect | TortureRed `Forward.hlsl` (transparent) | TortureRed `Lighting.hlsl` (opaque) | RTXDI `GlassPass.hlsl` (transparent) | RTXDI `CompositingPass.hlsl` (opaque) |
|--------|------------------------------------------|--------------------------------------|---------------------------------------|----------------------------------------|
| **Rendering method** | Rasterized alpha-blend | Fullscreen deferred | Ray-traced (up to 8 bounces) | Fullscreen deferred |
| **Runs before/after TAA** | TAA path: Before TAA ✅; Non-TAA: N/A | Before TAA ✅ | Before TAA ✅ | Before TAA ✅ |
| **ReSTIR DI contribution** | ❌ None | ✅ Full | ✅ Indirect (via opaque background in `HdrColor`) | ✅ Full |
| **ReSTIR GI contribution** | ❌ None | ✅ Full | ✅ Indirect (via opaque background in `HdrColor`) | ✅ Full |
| **Reflection on glass** | ❌ None | N/A | ✅ Ray-traced secondary ray (simplified shading) | N/A |
| **Transmission** | Alpha-blend (raster) | N/A | Ray-traced Fresnel + transmission | N/A |
| **Shadow on transparent** | ❌ None (shadow factor = 1) | ✅ Ray-traced | ✅ Implicit (background already shadowed) | ✅ Ray-traced |
| **Indirect diffuse on transparent** | ❌ None | ✅ NRD-denoised | ✅ Implicit (background already has GI) | ✅ NRD-denoised |
| **Ambient fallback** | `0.03 * albedo` | `0.03 * albedo` | `0.05 * diffuseAlbedo` (secondary hit) | N/A |
| **Multi-layer transparency** | ❌ No (single draw, unsorted) | N/A | ✅ Up to 8 layers | N/A |

### Key architectural lesson from RTXDI

RTXDI's solution to the DI/GI bypass problem is elegant: **run the glass pass after compositing, not before**. Because `CompositingPass` has already written the fully-lit opaque scene into `HdrColor`, the glass pass can simply read that buffer as the "background" seen through the glass. The glass surface itself gets correct DI+GI indirectly — it sees the correctly-lit opaque world behind it, and its Fresnel reflections see the correctly-lit opaque world in front of it.

This avoids the need to re-run ReSTIR DI/GI for transparent surfaces entirely. The trade-off is that the reflected image in glass is simplified (no recursive ReSTIR), but the transmitted image is fully correct.

TortureRed's [Known issue: DI/GI bypass](#known-issue-digi-bypass-in-transparent-pass) could be partially addressed by adopting this same pattern: move the transparent pass to run after the compositing/lighting pass but before TAA, and read the composited `HdrColor` as the background instead of re-computing lighting from scratch in `Forward.hlsl`.

## ⚠️ Known issue: DI/GI bypass in transparent pass

The current `Forward.hlsl` does **not** sample `FinalDiffuseTex` or `FinalSpecularTex`, meaning transparent objects receive **zero** ReSTIR DI contribution and **zero** indirect GI. The forward shader falls back to a simplified analytic lighting loop that:
- Never casts shadow rays
- Never uses ReSTIR importance sampling for local lights
- Has no indirect bounce light at all (only `0.03 * albedo` ambient)

This creates a visible disconnect: an opaque wall and a semi-transparent glass pane in the same scene will have markedly different lighting, even though they sample the same world-space position and the same lights.

**Root cause in the dispatch:**

In [`Application.cpp`](../Sources/Application.cpp), the transparency pass sets up `Model::Render()` and `OMSetRenderTargets` but does **not** bind `FinalDiffuseTex`/`FinalSpecularTex` via `BindlessIndices` (root constant at `b1`). The Forward PSO's root signature has slot `b1` available for `BindlessIndices`, but it is never populated during this pass.

Compare with the lighting pass which explicitly sets:
```cpp
indices.InputIdx0 = m_Renderer.GetFinalDiffuseTex().srvIndex;
indices.InputIdx1 = m_Renderer.GetFinalSpecularTex().srvIndex;
cmdList->SetGraphicsRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
```

**Proposed fix (3-step integration):**

1. **`Application.cpp`** — Before the transparency draw, transition `FinalDiffuseTex`/`FinalSpecularTex` to `D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE` and set `BindlessIndices` root constant `b1` with `InputIdx0`/`InputIdx1` pointing to their SRV descriptor indices.

2. **`Forward.hlsl`** — Add `#include "NRD.hlsli"`, declare `ConstantBuffer<BindlessIndices> g_Indices : register(b1);`, and after the local-light loop, sample and modulate DI/GI:
   ```hlsl
   if (FrameCB.enableRestirDI || FrameCB.enableRasterIndirectGI) {
       float2 screenUV = input.position.xy / float2(FrameCB.outputWidth, FrameCB.outputHeight);
       screenUV.y = 1.0 - screenUV.y; // D3D screen-space to texture-space
       float3 indirectDiffuse  = ResourceDescriptorHeap[g_Indices.InputIdx0]
                                    .SampleLevel(g_LinearSampler, screenUV, 0).rgb;
       float3 indirectSpecular = ResourceDescriptorHeap[g_Indices.InputIdx1]
                                    .SampleLevel(g_LinearSampler, screenUV, 0).rgb;
       float3 F0 = lerp(0.04, albedo.rgb, metallic);
       float3 diffFactor, specFactor;
       NRD_MaterialFactors(N, V, albedo.rgb, F0, roughness, diffFactor, specFactor);
       finalColor += indirectDiffuse * diffFactor + indirectSpecular * specFactor;
   }
   ```

3. **Resolution mismatch handling** — When TAA upsampling is active, `FinalDiffuseTex`/`FinalSpecularTex` are at internal resolution while the transparency viewport is at output resolution. The UV computed from `SV_POSITION` needs to be remapped from output-space `[0,outputSize]` to internal-space `[0,internalSize]`. This can be done by scaling the UV by `(internalWidth/outputWidth, internalHeight/outputHeight)` using additional `FrameConstants` fields or by creating output-resolution variants of the DI/GI textures.

> **Note:** The resolution-mismatch issue is the primary complexity. A pragmatic first-pass approach is to check `FrameCB.taaEnabled` — if TAA is off, the textures are at output resolution and no scaling is needed. If TAA is on, a bilinear upsample from internal to output resolution is required (or the fix can be deferred to a separate PR).
