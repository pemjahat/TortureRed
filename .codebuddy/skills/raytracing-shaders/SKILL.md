---
name: raytracing-shaders
description: This skill should be used when writing, reviewing, or modifying ray tracing shaders (HLSL) in TortureRed, including inline RayQuery (DXR 1.1) TraceRayInline dispatch, Proceed()/CommittedStatus() hit-classification loops, alpha-test candidate handling, shadow/visibility rays, and indirect-bounce rays in PathTracer.hlsl, RestirGI_*.hlsl, RestirDI*.hlsl, SHaRC_*.hlsl, and CommonTracing.hlsl. Triggers on tasks such as adding a new ray type, writing a shadow or visibility ray, adding an indirect GI bounce, wiring a new ray-miss consumer, or debugging ray-miss/hit behavior.
---

# Raytracing Shaders

## Overview

TortureRed's ray tracing pipeline (PathTracer, ReSTIR DI/GI, SHaRC) is built entirely on DXR 1.1 **inline ray tracing** (`RayQuery`) inside compute shaders. There is no classic DXR shader-table pipeline anywhere in this project — no `[shader("raygeneration")]` / `[shader("closesthit")]` / `[shader("miss")]` stages, no shader binding table, no `DispatchRays` call. Use this skill whenever writing, reviewing, or modifying HLSL that traces a ray in TortureRed.

## When this applies

- Adding a new ray query: a shadow ray, a GI bounce ray, a visibility test, or a ray-miss consumer (e.g. sky/environment sampling).
- Modifying `Sources/Shaders/CommonTracing.hlsl` — the shared ray-tracing helper header included by every ray-tracing shader in the project.
- Reviewing or debugging `Sources/Shaders/PathTracer.hlsl`, `RestirGI_Temporal.hlsl`, `RestirGI_RTXDI_Temporal.hlsl`, `RestirGI_Resolve.hlsl`, `RestirGI_RTXDI_Resolve.hlsl`, `SHaRC_Update.hlsl`, `RestirDI*.hlsl`, or any other `.hlsl` file that declares a `RayQuery<...>`.
- Any question about DXR 1.1 `RayQuery` semantics (`Proceed()`, `CommittedStatus()`, `CandidateType()`, ray flags, alpha-test/non-opaque candidate handling).

## How to use

1. Read `references/inline_rayquery_dxr11.md` (in this skill folder) before writing new `RayQuery` code. It documents the full DXR 1.1 inline ray tracing API surface, the project's alpha-test macro pattern (`PROCESS_ALPHA_MASK`), the shadow-ray vs. bounce-ray flag conventions already established in `CommonTracing.hlsl`, and pitfalls specific to this codebase.
2. Reuse the existing helpers in `Sources/Shaders/CommonTracing.hlsl` (`TracePrimarySurface`, `SampleIndirectRay`, `GetDirectLighting*`, `CheckVisibility`) instead of hand-rolling a new `RayQuery` loop — nearly every ray-tracing need in this project is already covered by one of these. (`SampleSky` is a planned addition, not yet implemented — see `docs/task010-skyrenderer.md`.)
3. When a genuinely new ray-tracing pattern is required, follow the conventions in the reference doc (ray flags, `TMin`/`TMax` epsilon biasing, the `PROCESS_ALPHA_MASK` macro inside every `Proceed()` loop) rather than introducing a divergent style.

## Resources

### references/

- `references/inline_rayquery_dxr11.md` — DXR 1.1 inline `RayQuery` reference: API surface, ray flags, hit classification, and TortureRed-specific conventions grounded in `Sources/Shaders/CommonTracing.hlsl`.
