# 🔍 Inline RayQuery (DXR 1.1) Reference

_Reference material for the `raytracing-shaders` skill — DXR 1.1 inline ray tracing API surface and TortureRed-specific conventions_

---

## 📋 Why this reference exists

TortureRed traces every ray — primary camera rays, shadow rays, GI bounce rays, and ray-miss sky sampling — through DXR 1.1 **inline ray tracing** (`RayQuery`) inside compute shaders[^1]. There is no classic DXR shader-table pipeline anywhere in the project: no `[shader("raygeneration")]`, no `[shader("closesthit")]`, no `[shader("miss")]`, no shader binding table, no `DispatchRays` call. This is a deliberate architectural choice, and it means "miss shader" logic is just a plain `if`/`else` branch after `RayQuery::Proceed()` completes — not a separate shader stage.

All conventions below are grounded in `Sources/Shaders/CommonTracing.hlsl`, the shared header `#include`d by every ray-tracing shader in the project (`PathTracer.hlsl`, `RestirGI_*.hlsl`, `RestirDI*.hlsl`, `SHaRC_Update.hlsl`).

---

## 📚 DXR 1.1 inline ray tracing API surface

### Declaring and dispatching a query

```hlsl
RayQuery<RAY_FLAG_NONE> q;
q.TraceRayInline(accelerationStructure, rayFlags, instanceMask, rayDesc);
while (q.Proceed()) {
    // Handle non-opaque / procedural candidates here (see below)
}
```

- The template parameter (`RAY_FLAG_NONE` above) sets **compile-time** ray flags; `TraceRayInline`'s second argument sets **runtime** ray flags. Most call sites in this project only need the runtime argument and leave the template parameter as `RAY_FLAG_NONE`.
- `rayDesc` is a standard `RayDesc` (`Origin`, `Direction`, `TMin`, `TMax`).
- `instanceMask` is an 8-bit mask compared against each instance's `InstanceMask`; TortureRed always passes `0xFF` (no per-instance ray masking is used).

### Ray flags in use in this project

| Flag                                       | Effect                                                         | Used for                          |
| -------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------- |
| `RAY_FLAG_NONE`                              | No special behavior; visits every candidate                     | Primary rays, GI bounce rays        |
| `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`   | Stops traversal at the first accepted hit (no closest-hit search) | Shadow/visibility rays              |

Shadow rays only need a boolean occluded/unoccluded answer, so `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH` is strictly faster — do not use `RAY_FLAG_NONE` for a shadow ray.

### The `Proceed()` loop and candidate classification

`Proceed()` returns `true` once per **non-opaque or procedural candidate** the traversal encounters, pausing traversal so the shader can decide whether to accept it. Fully opaque triangle hits are auto-committed by the driver and never surface through `Proceed()`.

Inside the loop, `CandidateType()` returns either:

- `CANDIDATE_NON_OPAQUE_TRIANGLE` — an alpha-tested/blended triangle; must be explicitly committed or rejected.
- `CANDIDATE_PROCEDURAL_PRIMITIVE` — not used anywhere in TortureRed (no procedural/AABB geometry).

To accept a non-opaque candidate: `q.CommitNonOpaqueTriangleHit()`. To reject it (treat as transparent): do nothing and let the loop continue. TortureRed centralizes this decision in a single macro (see below) rather than duplicating alpha-test logic at every call site.

### Reading the result after `Proceed()` returns `false`

| Accessor                              | Available when                          | Returns                                              |
| ---------------------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| `CommittedStatus()`                      | Always                                     | `COMMITTED_NOTHING`, `COMMITTED_TRIANGLE_HIT`, or `COMMITTED_PROCEDURAL_PRIMITIVE_HIT` |
| `CommittedInstanceID()`                  | `CommittedStatus() == COMMITTED_TRIANGLE_HIT` | Instance ID of the hit instance                      |
| `CommittedPrimitiveIndex()`              | Same                                       | Triangle index within the hit mesh                     |
| `CommittedTriangleBarycentrics()`        | Same                                       | `float2` barycentric coordinates of the hit             |
| `CommittedRayT()`                        | Same                                       | Hit distance along the ray                              |

**`CommittedStatus() != COMMITTED_TRIANGLE_HIT` is the project's "ray missed geometry" condition** — this is the exact check that replaces a `[shader("miss")]` stage everywhere in TortureRed.

---

## 🏗️ TortureRed conventions (`Sources/Shaders/CommonTracing.hlsl`)

### `PROCESS_ALPHA_MASK` — the one true alpha-test macro

`CommonTracing.hlsl:18-44` defines `PROCESS_ALPHA_MASK(q, rng)`, used inside **every** `Proceed()` loop in the project:

```hlsl
#define PROCESS_ALPHA_MASK(q, rng) \
    if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) { \
        /* fetch instance/material/vertex data, resample base color alpha */ \
        if (mat.alphaMode > 0) { \
            /* ALPHA_MODE_MASK: cutoff test against next_float(rng); ALPHA_MODE_BLEND: stochastic accept */ \
            if (next_float(rng) < alpha) { q.CommitNonOpaqueTriangleHit(); } \
        } else { \
            q.CommitNonOpaqueTriangleHit(); \
        } \
    }
```

> ⚠️ **Never write a bare `while (q.Proceed())` loop without calling `PROCESS_ALPHA_MASK(q, rng)` inside it.** Omitting it makes every non-opaque triangle either invisible (never committed) or fully opaque depending on driver defaults — both are wrong, and both are silent (no compile error, no crash — just incorrect alpha-tested geometry in the render).

### Primary ray pattern

`TracePrimarySurface()` (`CommonTracing.hlsl:195-235`) is the canonical primary-ray pattern: build a `RayDesc` from the camera (`GetCameraRayDirection()`), trace with `RAY_FLAG_NONE`, drain `Proceed()` with `PROCESS_ALPHA_MASK`, then branch on `CommittedStatus()`. `TMin = 0.001f`, `TMax = 10000.0f` are the project's standard near/far bounds for all rays (camera, bounce, and shadow rays alike).

### Shadow / visibility ray pattern

`GetDirectLighting()` (`CommonTracing.hlsl:273-297`) and `CheckVisibility()` (`CommonTracing.hlsl:387-405`) show the standard shadow-ray shape:

```hlsl
RayDesc shadowRay;
shadowRay.Origin = P + N * 0.001f;   // Normal-offset bias — avoids self-intersection ("shadow acne")
shadowRay.Direction = L;
shadowRay.TMin = 0.001f;
shadowRay.TMax = dist - 0.002f;      // For point/local lights: stop just short of the light itself

RayQuery<RAY_FLAG_NONE> sq;
sq.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
while (sq.Proceed()) { PROCESS_ALPHA_MASK(sq, rng); }

bool visible = (sq.CommittedStatus() == COMMITTED_NOTHING);
```

For directional lights, `TMax = 10000.0f` (no light-position distance to stop short of). Always bias the ray origin along the surface normal (`+ N * 0.001f`) — skipping this reintroduces self-intersection artifacts.

### Indirect bounce ray pattern and miss handling

Every GI bounce loop (`PathTracer.hlsl`, `RestirGI_Temporal.hlsl`, `RestirGI_RTXDI_Temporal.hlsl`, `SHaRC_Update.hlsl`) follows the same shape: sample a direction via `SampleIndirectRay()`, trace with `RAY_FLAG_NONE`, then branch:

```hlsl
if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    // Resolve the hit surface via ResolveHitSurface(), continue the path
} else {
    // Ray escaped the scene — this is the "miss" branch.
    // TortureRed's convention: call SampleSky(ray.Direction) here.
}
```

`ResolveHitSurface()` (`CommonTracing.hlsl:154-193`) takes a `minRoughness` clamp parameter: `0.01f` for primary/near hits, `0.15f` for deeper indirect bounces (reduces fireflies from sharp specular lobes several bounces deep).

### Russian roulette and throughput

Bounce loops apply Russian roulette after bounce 2 (`if (bounce > 2) { ... }`), terminating the path stochastically based on `max(throughput.r, throughput.g, throughput.b)` and dividing the surviving throughput by the survival probability. This is not DXR-specific, but it is the standard termination pattern paired with every `RayQuery` bounce loop in this project — new bounce loops should replicate it rather than using a fixed bounce count alone.

---

## ⚠️ Common pitfalls

- **Missing `PROCESS_ALPHA_MASK` inside a `Proceed()` loop.** Silent — no error, just wrong alpha-tested geometry. See the callout above.
- **Using `RAY_FLAG_NONE` for a shadow ray.** Works correctly but wastes performance — traversal keeps searching for a closer hit even after finding any occluder. Use `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`.
- **Forgetting the origin bias (`+ N * 0.001f`).** Causes self-intersection ("shadow acne") — the ray immediately re-hits the surface it started from.
- **Confusing `Candidate*()` and `Committed*()` accessors.** `Candidate*()` accessors are only valid *inside* the `Proceed()` loop body (before a decision is made); `Committed*()` accessors are only valid *after* `Proceed()` returns `false`. Calling the wrong set at the wrong time is a common copy-paste mistake when adapting an existing ray-tracing shader.
- **Treating `COMMITTED_PROCEDURAL_PRIMITIVE_HIT` as reachable.** TortureRed has no procedural/AABB geometry — only `COMMITTED_NOTHING` and `COMMITTED_TRIANGLE_HIT` occur in practice. Don't add dead-code branches for the procedural case.

---

## 🔗 References

[^1]: Microsoft. "DirectX Raytracing (DXR) Functional Spec — Inline Raytracing (Tier 1.1, RayQuery)." _DirectX-Specs_. https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#inline-raytracing
