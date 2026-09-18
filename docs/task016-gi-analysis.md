# TortureRed: relightable baked GI analysis

_Analysis of the requested SIGGRAPH talks, papers, and their reference chains for a static-scene, dynamic-TOD, probe-based baked indirect lighting path — September 15, 2026_

---

## 📋 Summary

**Conclusion:** the main line should be "precomputed light transport + geometry-aware probe reconstruction", not simply a DDGI that stops updating.

The most worthwhile combination from these sources:

- **Remedy 2015** — a system architecture that separates sun, sky, and local lights.
- **CoD 2017/2020** — baking optimized against the final reconstruction, and keeping direct light from polluting low-resolution caches.
- **Activision 2021** — sampling strategy, constrained SH, and interpolation-aware compression.
- **DDGI 2019/2021** — a practical baseline for visibility, relocation, and bias.
- **Neural Light Grid 2024** — important follow-up work found along the reference chain, aimed directly at probe influence domains and leaking.
- **The second half of `GI介绍ppt.pdf`** — a very relevant alternative: learning "position + TOD → SH" from multiple bakes.

I verified the original slides, speaker notes, and reference lists, and also reviewed the current code and the existing `docs/task015-baked-gi-brainstorm.md`. Several judgements in the draft need correcting below. No rendering code or research documents were modified in this pass.

The discussion assumes fixed geometry and materials, with sun direction/intensity and sky able to change, and prioritizes indirect diffuse lighting. If the TOD only follows a fixed sun trajectory, the problem shrinks substantially.

---

## 🎯 What is actually baked

All three routes below may be called "baked GI", but their capabilities differ:

| Route | What is precomputed | Dynamic TOD capability |
| --- | --- | --- |
| Fixed-light probes | Irradiance for one lighting state | Only scale, blend, or switch between existing states |
| Environment light transport | The scene's response to different incident lighting | Recombine sun/sky within the representation's capability |
| Geometric transport relationships | Associations between surfels, probes, and receivers | Relight surfaces at runtime, then propagate the lighting |

For your requirements I would prioritize the second route; the third is an important fallback.

Conceptually, each probe can be expressed as:

$$
\mathbf{e}_p(t) = T_p^{\mathrm{sky}}\,\mathbf{l}_{\mathrm{sky}}(t) + \mathbf{r}_p^{\mathrm{sun}}\big(\omega_{\mathrm{sun}}(t)\big) \odot \mathbf{E}_{\mathrm{sun}}(t) + \mathbf{e}_p^{\mathrm{fixed}}
$$

Where:

- $\mathbf{e}_p$ — directional irradiance coefficients that can be sampled at runtime.
- $T_p^{\mathrm{sky}}$ — transport from sky lighting to probe irradiance.
- $\mathbf{r}_p^{\mathrm{sun}}$ — the indirect response per unit of sun lighting, as a function of direction.
- Color bounces off fixed materials are contained in the transport.
- The material response of the receiving surface is applied exactly once, at final shading.

Path ownership must be explicit:

- Direct sunlight on camera-visible surfaces continues to be computed exactly at runtime.
- Contributions that reach a receiving point after the sun bounces off the scene belong to the baked GI.
- Light arriving directly from the sky without bouncing off another surface still needs occlusion. If that term is not handled separately, it must be included in the sky transport.

So the draft's "exclude both the first-bounce sun and the direct sky light" needs correction: excluding direct sun is fine, but the occluded sky term must not be dropped as well.

---

## 📚 Core sources: what is worth adopting

### Remedy 2015 — the system skeleton closest to the requirement

The key content in `SIGGRAPH_2015_Remedy_Notes.pdf` is on pages 29–35, 75–103, and 106–112. It explicitly separates:

- Local-light irradiance.
- Indirect sun-light transport.
- Sky-light transport.
- Separate specular probe data and visibility.[^1]

This is not simply "three probe sets for morning, noon, and evening".

Points worth borrowing:

- **Unified world-space representation.** Static surfaces, dynamic receivers, and volumetric effects can all read the same GI, without depending on unique UVs.
- **Genuine 3D adaptivity.** A voxel tree with $4 \times 4 \times 4$ branching, with 0.5 m, 2 m, and 8 m levels in the example, suited to scenes with very uneven geometric density.
- **Treating hierarchy continuity as part of the design.** The partial dilation on page 100 constrains the interpolation neighborhood to one or two levels, avoiding unbounded recursion. You cannot implement only "sparse brick allocation" and leave the seams for the end.
- **Separating diffuse from specular.** The talk states explicitly that a high angular-resolution specular PRT representation is infeasible due to data size.

**Boundary:** the original supports the direction of "separating the transport", but does not disclose a sun-direction table specification that can be copied directly. You cannot derive "8 sun directions are enough" from it.

### Sloan 2020 — the key is consistency between baking and reconstruction

The most valuable content is not saving a few SH coefficients, but how the low-resolution representation affects the whole baking process.[^2]

#### A. Retracing direct light happens inside the bake

Page 11 explains that their series-expansion baker reuses the runtime lighting data structures. But models have very sparse probes, and caching a sharp direct light like the sun can make a model behave like a "light tunnel" and produce a false bounce.

The fix is to recompute the sun's direct light when a bake ray hits a model, instead of reading it from that coarse cache.

The implication for TortureRed:

> Even if only indirect light is stored in the end, the direct lighting used while computing that indirect light must have sufficiently accurate visibility.

This goes deeper than "subtract the sun's direct contribution during final composition".

#### B. SH ratio encoding is practical, but conditional

Pages 33–34 store DC in a floating-point format and divide the remaining coefficients by DC before quantization. This is worth keeping as a compression candidate, but:

- The theoretical bound relies on appropriate non-negativity assumptions about the signal.
- Interpolating DC and the ratios, then multiplying, is generally not equivalent to linearly interpolating the original coefficients; page 57 discusses this explicitly.
- The encoding, which targets non-negative lighting signals, cannot be applied directly to arbitrary transport matrix columns, which correspond to signed SH input bases.
- It is also not a general compression method for depth moments or visibility data.

#### C. Series-expansion baking is an accelerator, not an unbiased reference

It reuses sub-paths, but every iteration passes through the cached representation, so spatial interpolation error also enters subsequent bounces. Keep an independent path-traced reference; you cannot validate a coarse probe propagation using its own results.

### Silvennoinen 2021 — four contributions that should be understood separately

#### 1. Warped irradiance volumes

Pages 9–12 obtain a height envelope using rays cast up and down, concentrating more volume samples in the height band the scene occupies.[^3]

It suits worlds that spread mainly over a 2D surface, such as terrain and city blocks.

But it is not a substitute for arbitrary 3D scenes. When multi-story interiors, underground spaces, and geometry above and below a bridge all need detail at once, a single height envelope is not necessarily efficient.

**For TortureRed:** worth considering for large outdoor scenes; for complex interiors, compare adaptive bricks first.

#### 2. Visibility-based sample validation

The focus of pages 22–24 is the selection of baked spatial samples, not the runtime probe-to-point visibility test used by DDGI.

The speaker uses a prior over "where the player/camera is more likely to be" to reject some low-visibility spatial samples and reduce their contamination of the volume fit.

One important limitation:

> "Rays are all short" cannot be universally interpreted as an invalid sample. Enterable narrow rooms, alcoves, and corridors may match exactly that condition.

For you, sampling importance should be determined by receiving surfaces, playable areas, and validation viewpoints, rather than unconditionally favoring open space.

#### 3. Constrained SH

Pages 34–50 discuss not only negative-value ringing but also color shifts, using spatial-domain constraints and YCoCg color constraints before projecting back to SH.

It addresses:

- Negative irradiance.
- Overshoot.
- Color shifts in colored bounces.
- Plain windowing over-smoothing the directionality.

But SH deringing does not solve cross-wall interpolation. The two must not be conflated.

#### 4. Moving Basis Decomposition

The most valuable idea in MBD is that spatial interpolation is built into the compression optimization itself, rather than compressing independent blocks first and fixing block boundaries afterwards.[^3]

One important confusion in the draft needs correcting:

- Page 72 shows 324-dimensional transport data: three RGB $9 \times 9$ indirect transport matrices plus one direct transport matrix.
- The 44:1 ratio and 1.09 bytes/voxel on page 76 correspond to a different example: 48 bytes/voxel of linear RGB SH irradiance.
- You cannot claim that "the full 324-dimensional transport has also been proven to compress to about 1 byte/probe".

### Treyarch 2016 — the failure cases are the most valuable part

This talk is very valuable for leak control, but its final irradiance volume is not itself a general TOD transport representation.[^4]

The key material is on pages 30–32 and 51–58:

- Adjusting volume sampling by normal can still let interior details facing outward pick up exterior lighting.
- Once those errors enter multiple bounces, they keep contaminating the entire room.
- They tried per-voxel dividing planes and an SDF, but black splotches appeared on small geometry and in corners.
- The final combination was macroscopic attenuation volumes, sample point adjustment, sample invalidation, and inpainting.

Implications for TortureRed:

- Leaking cannot be covered up only at final shading; incorrect energy must be prevented from entering the baked propagation.
- Macroscopic spatial isolation and microscopic geometric handling are different problems.
- Inpainting must respect geometric and regional constraints; it cannot smooth unconditionally across walls.

Also, the "bake only about 1/27 of the voxels in open areas" on page 54 saves bake time only; it does not reduce the memory or sampling cost of the final volume textures.

### DDGI 2019/2021 — a good geometric baseline, not an exact visibility answer

The local `paper-lowres.pdf` is the 2021 productionization paper, not the original 2019 DDGI.[^5]

The most transferable parts for a static bake are:

- Probe relocation and invalid-probe classification.
- Directional distance moments.
- Combining normal, visibility, and spatial weights.
- Production experience with the self-shadow bias.
- Boundary handling for multiple volumes.

The advantage of a static scene is that this geometric information can be computed offline instead of being re-traced every frame.

But remember:

- **Two depth moments are not exact occlusion.** When a directional texel region mixes a nearby wall with distant open space, the statistical visibility still fails.
- **Relocation cannot fix every coverage gap.** The paper limits displacement to preserve grid indexing properties, and explicitly acknowledges that some probes cannot be moved out of walls.
- **Bias affects both bright and dark leaking.** Too much bias can move the query point to the wrong side; too little can over-occlude.
- **Do not copy the dynamic hysteresis strategy.** Offline transport recombination need not depend on historical convergence, so do not artificially introduce the lighting lag of a dynamic DDGI.

---

## 🔍 Sousa and the three Chinese decks: what to take and what to skip

### Sousa 2025 — borrow the multi-scale architecture; do not treat the reflection refit as full relighting

The core of idTech 8 is dynamic GI. The most valuable lesson is that the world-space cache provides stable coverage while the final gather handles detail near receiving surfaces; high-quality GI is not achieved by shading directly from coarse probes alone.[^6]

This supports keeping two comparable modes in TortureRed:

- Pure probe reconstruction.
- Short-range RT final gather that terminates subsequent paths in the baked cache.

The reflection formula on page 26:

$$
R_{\mathrm{new}} \approx \frac{R_{\mathrm{baked}}}{E_{\mathrm{probe}}}\,E_{\mathrm{frame}}
$$

is a production-valuable brightness and color adaptation method, but it is not a complete dynamic reflection solution:

- It does not move a sun highlight to a new direction.
- It does not create visible reflection content that was absent from the original cubemap.
- It does not restore new occlusion relationships.
- A near-zero denominator must be handled.

So for a high-quality target, keep the existing RT specular path first, rather than making a single-time cubemap plus an irradiance ratio the end state.

### `letianyu_v7.pptx` — separating surface solving from spatial transport

Pages 16–29 distinguish:

- **Surface Probe** near surfaces, which computes the Local Reflection.
- **Transport Probe** in space, which computes the Global Transport.
- Tetrahedral adjacency, which organizes the transport.
- Final resampling into volume textures, which serves shading.

The leak handling on page 77 is normal extrusion plus a local visibility SH.

The most valuable idea:

> The sampling points used to solve GI need not be the same data structure used for final shading.

But "local infinite bounces" does not mean the whole scene's infinite bounces have been solved exactly. Globally, it still goes through a discrete probe graph, iteration, and reconstruction.

It is a good reference for "local precomputation + global runtime propagation", and should not be treated as an equivalent implementation of high-precision offline transport.

### `移动端GI.pptx` — the point is not the absence of precomputation, but the absence of traditional offline lightmaps

This IllusionGI talk contains:

- Voxel color and SDF.
- A local SH transfer matrix.
- Directional-light relighting using a shadow map.
- Global radiance propagation, then projection to SH.
- For an unchanged scene, parts of the local construction run only once.

Notably, the speaker notes on page 20 acknowledge that its physical correctness is weaker than ray-traced lightmaps or PRT.

So for your current target:

- The local/global decomposition and memory access organization are worth borrowing.
- "Fully dynamic and fast on mobile" is not sufficient reason to adopt its propagation approximation as a high-quality baker.

### `GI介绍ppt.pdf` — LUX and the neural TOD section must be read separately

**The first half, LUX:**

- Separates propagation-stage leaking from sampling-stage leaking.
- Interior volumes, depth comparison, and a custom trilinear sampler.
- Pages 56–57 propose a 4-byte SG depth/visibility scheme.
- Page 72 has navmesh- and collider-driven automatic probe placement.

The 4-byte scheme is worth studying as a compression option, but the material does not disclose the encoding or error conditions sufficiently to treat it as a quality-equivalent replacement for full directional depth moments.

**The second half is actually closer to this requirement.** Pages 115–135 describe:

$$
(\text{Probe Position},\ \text{Time of Day}) \longrightarrow \text{SIREN} \longrightarrow \text{SH}
$$

I also verified the diagram: it trains sector-specific models from multiple baked states and predicts indirect SH at runtime.

This is a valid candidate, but it learns a **fixed family of TOD states**:

- If the sun, sky, and light switches follow fixed time rules, it fits very well.
- If sky distribution, sun trajectory, weather parameters, or light states must change independently, those degrees of freedom must be included in training.
- The talk's "error below 1%" and "smaller than MBD" results lack sufficient metric definitions and controlled conditions, so they cannot be converted directly into a quality promise for TortureRed.

It predicts lighting. The Neural Light Grid below represents interpolation weights. The two are entirely different.

---

## 💡 Additional findings along the reference chain

### A. CoD: Infinite Warfare 2017 — should be raised to the highest priority

This is a direct reference of Sloan 2020, and pages 51–58 and 108–109 are especially worth reading.[^8]

It offers two ideas more valuable than "place probes uniformly":

**Layout adapts to both geometry and lighting error.**

- Start from a regular, well-shaped tetrahedral structure.
- Subdivide near geometry, navmesh, and important regions.
- Then adjust resolution based on reconstruction error between fine and coarse levels.
- Do not simply Delaunay-triangulate arbitrary sparse points and consider the layout problem solved.

**Bake the final reconstruction, not just the probe centers.**

Page 109 states explicitly:

- Take multiple reference samples inside a tetrahedron.
- Use the same visibility and interpolation weights as at runtime.
- Solve for probe values that best reconstruct those samples.
- Add a spatial regularization term that accounts for visibility.

This is the quality mechanism I believe the current draft most needs to add.

For TOD, this can be extended further: fit and validate jointly across multiple sun/sky states, instead of optimizing only for noon.

### B. Neural Light Grid 2024 — the production-level leak work most worth adding

This is the authors' follow-up work, not a reference in the older talks. It learns or represents a spatial influence function $\Phi_p(x)$ per probe:[^9]

$$
\hat{L}(x) = \frac{\sum_p \Phi_p(x)\,L_p}{\sum_p \Phi_p(x)}
$$

The difference from a plain visibility test:

- The influence domain should stop at walls.
- It can bend around corners.
- It should not produce an unreasonable "probe shadow" just because a small column blocks the straight probe-to-point line.

The paper uses a small MLP or product fields, and documents real game deployment.

But two reservations are mandatory for TortureRed:

- Its influence domains are generated using not only geometry but also reference lighting differences and lighting clustering. Without validation, weights generated under one fixed lighting state cannot be assumed optimal for arbitrary TOD.
- Page 14 explicitly acknowledges that limited representation capacity still leaks, and relies on production constraints such as a minimum wall thickness.

My suggestion is to borrow the idea of a bendable influence domain that can extend past small occluders first, without introducing the full neural training pipeline in the first phase.

### C. Sparse Radiance Probes 2017 — stronger local reconstruction for static scenes

_Real-time Global Illumination by Precomputed Local Reconstruction from Sparse Radiance Probes_ separates global sampling from a precomputed local reconstruction, and incorporates mutual visibility into the reconstruction operator.[^10]

Its value for you:

- The scene is static, so local geometric relationships can be precomputed.
- Not every detail has to be carried by coarse spatial probe density.
- A small number of global probes can be paired with a finer receiver reconstruction.

This is more worth investigating than simply increasing probe count. However, it is a paper route; this citation alone does not make it a shipped solution fully equivalent to your target.

### D. The Division 2016 — a direct production reference for the second architecture route

This is the work actually cited on page 110 of `GI介绍ppt.pdf`, and must not be confused with the 2012 Far Cry 3 approach.[^11]

It precomputes the surfels and association weights that each probe can see, and at runtime:

- Recomputes surfel lighting.
- Aggregates into bricks.
- Transports to probes.
- Generates the irradiance volume used for shading.

The advantage is that the sun input does not have to be limited to low-order SH or a small number of full-scene keyframes.

But its postmortem explicitly acknowledges:

- Multiple bounces are coarsely approximated.
- Missing probes create dark areas.
- Interior/exterior volume switching has seams.
- Sun shadow map coverage of surfels requires dedicated handling.

So borrow its structure, not its resolution or bounce approximations from that era.

### Reading priority for the other references

| Reference | Source | Suggested use |
| --- | --- | --- |
| Fast Non-uniform Radiance Probe Placement and Tracing, 2019 | DDGI 2021 | Scene skeletons, visibility coverage, non-uniform layout; not a drop-in replacement for tetrahedral interpolation[^12] |
| Ray Guiding for Production Lightmap Baking, 2019 | Sloan 2020 | Bake convergence for dark rooms and small windows |
| UberBake, 2020 | Sloan 2020 | Bake system, cache reuse, iteration workflow |
| ZH3, 2024 | Authors' follow-up work | More compact directional irradiance; does not solve spatial leaking |
| Precomputed GI in Frostbite, 2018 | Sousa 2025 | High-quality baker and representation workflow; the correct attribution is Yuriy O'Donnell, GDC 2018 |
| God of War indirect lighting, 2019 | Sousa 2025 | Reflection normalization and production workflow, useful for later specular work |

These last four should not take priority over transport correctness and spatial reconstruction. Source links are listed at the end of this document.[^13]

---

## 📍 Probe distribution and leak control

### Distribution: guarantee coverage and topology first, then pursue sparsity

| Scheme | Suitable scenes | Main risk |
| --- | --- | --- |
| Uniform grid + relocation | First-version baseline, control experiments | Cubic memory growth, insufficient coverage in complex interiors |
| Adaptive bricks/tree | Mixed indoor/outdoor, highly varying detail density | Hierarchy seams, neighborhood queries |
| Height-warped volume | Terrain, city blocks | Not necessarily efficient for overlapping multi-level spaces |
| Non-uniform tetrahedral probes | Strict probe budgets | Ill-conditioned cells, query cost, cross-wall interpolation |
| Surface + transport layering | Runtime relighting and propagation | Managing the association and error of two sample types |

I would not assume now that a full-scene 2 m grid is a high-quality configuration.

A more reasonable approach:

- Build an error curve across several grid spacings.
- Check coverage separately on both sides of walls, in window openings, in narrow corridors, and at contact regions.
- Subdivide based on reconstruction error across multiple lighting states.
- Preserve sufficient valid support for every important receiver, rather than only maximizing "the number of probes in free space".

"A probe is not embedded in a wall" does not mean "it correctly serves every surrounding surface".

### Leak control: split it into at least four problem classes

| Error class | Typical symptom | Corresponding measure |
| --- | --- | --- |
| Transport error | Light has already passed through a wall during baking | Correct geometry/alpha, accurate direct-light visibility |
| Spatial reconstruction error | An outdoor probe lights an interior | Coverage, visibility, influence domains, reconstruction-aware fitting |
| Angular representation error | Negative values, color shift, blurred directionality | SH constraints, higher angular resolution, a suitable basis |
| Compression/hierarchy error | Block seams, flicker, brightness drift | Interpolation-aware compression, boundary consistency, cross-TOD validation |

My recommended baseline stack:

1. Geometric validity checks and offline relocation.
2. Directional visibility data retained at sufficient precision.
3. Reconstruction weights consistent with runtime.
4. Fitting and validation at receiving positions and across multiple TOD states.
5. Increased resolution or stronger local influence domains in failing regions.
6. Visibility compression only last.

AO cannot repair incorrect cross-wall energy, and blindly multiplying by AO may compute already-baked occlusion a second time.

Likewise, exact probe-to-point visibility is not the final answer: it can reject probes that should legitimately provide similar irradiance, creating dark leaks. The Neural Light Grid discussion of this point is well worth reading.

---

## ⚙️ Recommended architecture and corrected quality expectations

### Main line: sky transport plus an independent sun response

For the first version I suggest:

- **Sky** — transport from a radiance SH input to directional irradiance.
- **Sun** — an independent directional response representation.
- **Diffuse output** — keep an unaggressively compressed SH9 baseline first.
- **Spatial reconstruction** — start with explicit, debuggable geometric weights, then compare reconstruction-aware fitting.
- **Specular** — keep the existing RT path.

**The sun response cannot be assumed smooth by default.**

Diffuse bounces smooth many variations, but narrow windows, light shelves, and thin apertures can still make indirect sunlight change rapidly with direction. Therefore:

- Fixed sun trajectory: a one-dimensional response is viable, with sampling increased according to error.
- Arbitrary azimuth/elevation: two-dimensional directional coverage is required.
- Directions outside the training/sampling set must be validated separately.
- A "32-direction lookup table" is an approximate representation, not an exact reference.

If the sun response data becomes too large, or if many dynamic local lights are required, then seriously compare The Division-style surfel relighting with precomputed geometric transport relationships.

### A memory example

Assuming **100,000 valid probes**, the figures below are a design estimate only, not a recommended configuration:

| Data | Per probe | Total |
| --- | ---: | ---: |
| Sky SH9 → SH9, RGB FP16 | 486 B | 46.4 MiB |
| 32-direction sun response, SH9 RGB FP16 | 1,728 B | 164.8 MiB |
| $16 \times 16$ RG16F visibility, no border | 1,024 B | 97.7 MiB |
| Current relit SH9 RGB FP16 | 54 B | 5.1 MiB |

The total is roughly **314 MiB**, before borders, positions, indices, and influence domains.

This shows that:

- Visibility is not cheap incidental data.
- The sun representation may be much larger than the current lit-probe data.
- Compression should be discussed separately for transport, visibility, and runtime results, instead of applying one ratio to the whole scheme.

### "High-quality baking" still has to be demonstrated

More offline samples mainly reduce Monte Carlo noise. They do not automatically eliminate:

- Insufficient spatial sampling.
- Incorrect probe support domains.
- Under-sampling of sun directions.
- SH truncation.
- Cache propagation bias.
- Compression error.

I therefore retract the following over-strong claims from the draft:

- A small number of sun directions is naturally sufficient.
- A single DXR bake necessarily takes only minutes.
- MBD's 44:1 can be applied directly to full transport.
- The reflection ratio refit is sufficient as a high-quality dynamic specular endpoint.
- The existing full PT can be used as fully accurate ground truth without inspection.

---

## 🔧 Concrete landing points in TortureRed

Several signal conventions in the current implementation must be preserved:

- **SHaRC caches already-shaded surface radiance, not lighting-independent irradiance transport.** Saving the current SHaRC does not produce the bake this work needs.
- **`Sky_ProjectSH9.hlsl` outputs irradiance SH that has already been cosine-convolved.** If a transport matrix expects radiance SH, expose the raw radiance projection, or deconvolve correctly over the supported frequency bands.
- **`Lighting.hlsl` already handles the primary direct light separately.** Baked sun/sky GI must replace the overlapping contribution; it cannot simply be added on top of the full ReSTIR GI.
- **If probe irradiance replaces SHaRC at the secondary hit**, it must be converted into outgoing radiance for that surface, and it must be explicit whether secondary direct lighting is still added.
- **The bounce depth, clamp, and roughness handling in the current `PathTracer.hlsl` need review.** A high-quality bake or reference should not inherit every real-time optimization.

### My recommended validation order

1. **Build a small-scene, uncompressed correctness baseline.** Thin walls on both sides, narrow windows, a two-story room, columns, and colored bounces; no MBD or neural compression yet.
2. **Validate sky and sun separately.** Test occluded sky and sun indirect light independently, then combined; for the sun, check interpolated intermediate states, not only the sampling nodes.
3. **Compare reconstruction methods.** Plain trilinear, DDGI-style visibility, and reconstruction-aware fitting, adding explicit influence domains where necessary.
4. **Then do adaptivity and compression.** Spend probes according to error, rather than picking a compression ratio first and working backwards to quality.
5. **Compare pure probes against RT final gather.** If pure probes require excessive spatial density to handle detail, a short-range final gather using the existing DXR may be more cost-effective than endlessly adding probes.

In the end, the combination I most recommend is:

> **Remedy's transport separation + CoD's reconstruction-aware baking + DDGI's geometric baseline + Activision's interpolation-aware compression; treat the Neural Light Grid influence domain as a key quality investigation, and keep the RT final gather as a comparison path.**

What should be resolved up front is not "SH, SG, or neural", but: **which probes should provide what information to each receiving position, and whether that information remains correct across all target sun-lighting states.**

---

## ⚠️ Evidence limits and TOD scope

Further verification adds an important qualification to the recommendations above: **production-proven components** are not the same as a **production-proven, complete dynamic TOD solution**.

### CoD evidence boundaries

- **Sloan 2020, page 129** explicitly lists dynamic TOD as future work. Its shipped lightsets primarily handle intensity and color changes to predefined light groups; they do not establish support for arbitrary changes in sun direction.[^2]
- **Activision 2021** presents compressed irradiance in its production example, while the 324-dimensional transport representation belongs to a separate experiment. This shows MBD can compress transport data, but does not establish that the production compression ratio applies to a complete dynamic sun-and-sky solution.[^3]
- Consequently, the combination recommended above is an **architecture proposal for TortureRed**, not a complete implementation demonstrated in a single presentation that can simply be copied.

### TOD degrees of freedom should determine the baking route

| Actual requirement | Approaches to prioritize for comparison |
| --- | --- |
| Fixed sun trajectory, with sky, weather, and lights following predefined time curves | High-quality bakes at multiple times → interpolation, low-rank compression, or neural prediction |
| Independently adjustable sun direction, intensity, and sky distribution | Explicit light transport, with the sun response handled separately |
| Arbitrarily moving local lights or changes to some materials | Surfel relighting with precomputed geometric transport relationships |

So if your "Dynamic Time of Day" only follows a predefined day-night curve, the multi-state baking route in `GI介绍ppt.pdf` should be evaluated **alongside full PRT**, rather than assuming the latter is inherently more advanced.[^7] It may preserve richer lighting detail within a narrower domain, but intermediate times that were not part of the bakes must be validated separately.

### Two implementation-level qualifications

- **The evidence for reconstruction-aware baking is strong.** CoD 2017 page 109, and CoD 2020 pages 18 and 31, support solving for probe values from reference samples on receiving surfaces, rather than only measuring lighting independently at probe centers. This remains the quality mechanism I most strongly recommend investigating first.[^2][^8]
- **Sousa does not prove that high-quality probe GI must use an RT final gather.** The implementation does trace one ray per gather pixel, but gather resolution varies by quality setting, and no fixed threshold for a "short-range ray" is specified. The short-range gather suggested earlier is a comparison option for TortureRed, not a conclusion of the presentation.[^6]

**The overall priority is unchanged, but the starting point is clearer: first define the range of lighting changes that must be supported, then use uncompressed reference data to validate spatial reconstruction and TOD interpolation, and only afterwards choose a transport, low-rank, or neural representation. Do not choose the compression technique first and then make the quality target accommodate it.**

---

## 🔗 References

[^1]: Silvennoinen, A., and Timonen, V. (2015). "Multi-Scale Global Illumination in Quantum Break." SIGGRAPH 2015, Advances in Real-Time Rendering in Games. Pages 29–35, 75–103, 106–112. https://arisilvennoinen.github.io/Publications/SIGGRAPH_2015_Remedy_Notes.pdf

[^2]: Sloan, P.-P., and Silvennoinen, A. (2020). "Precomputed Lighting Advances in Call of Duty: Modern Warfare." SIGGRAPH 2020, Advances in Real-Time Rendering in Games. Pages 11–13, 18, 31, 33–34, 57, and 129. https://research.activision.com/publications/2020-09/precomputed-lighting-advances-in-call-of-duty--modern-warfare

[^3]: Silvennoinen, A. (2021). "Large-Scale Global Illumination at Activision." SIGGRAPH 2021, Advances in Real-Time Rendering in Games. Pages 9–12, 22–24, 34–50, and 72–76. https://advances.realtimerendering.com/s2021/index.html . Related MBD paper: https://arisilvennoinen.github.io/Publications/mbd.pdf

[^4]: Hooker, J. (2016). "Volumetric Global Illumination at Treyarch." SIGGRAPH 2016, Advances in Real-Time Rendering in Games. Pages 30–32 and 51–58. https://www.activision.com/cdn/research/Volumetric_Global_Illumination_at_Treyarch.pdf

[^5]: Majercik, Z., Guertin, J.-P., McGuire, M., and Nowrouzezahrai, D. (2019). "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields." JCGT 8(2). https://jcgt.org/published/0008/02/01/ . Productionization: Majercik, Z., Marrs, A., Spjut, J., and McGuire, M. (2021). "Scaling Probe-Based Real-Time Dynamic Global Illumination for Production." JCGT 10(2). https://jcgt.org/published/0010/02/01/ — local copy: `E:/Paper/GI/paper-lowres.pdf`

[^6]: Sousa, T. (2025). "FAST AS HELL: idTech 8 Global Illumination." SIGGRAPH 2025, Advances in Real-Time Rendering in Games. Pages 21 and 26. https://advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf

[^7]: "Best Practice For Cross-platform Realtime GI" (Earth: Revival), pages 115–135; "自研GI方案介绍" (IllusionGI); and "职级晋升答辩" (letianyu / Clash Heroes). Local sources under `E:/Paper/GI/`: `GI介绍ppt.pdf`, `移动端GI.pptx`, and `letianyu_v7.pptx`. No publication years are asserted here.

[^8]: Iwanicki, M., and Sloan, P.-P. (2017). "Precomputed Lighting in Call of Duty: Infinite Warfare." SIGGRAPH 2017, Advances in Real-Time Rendering in Games. Pages 51–58 and 108–109. https://www.ppsloan.org/publications/CoD_IW.pptx

[^9]: Iwanicki, M., Sloan, P.-P., Silvennoinen, A., and Shirley, P. (2024). "The Neural Light Grid: A Scalable Production-Ready Learned Irradiance Volume." Activision Technical Report ATVI-TR-24-03. Page 14 for the stated limitations. https://www.ppsloan.org/publications/Neural_Light_Grid.pdf

[^10]: Silvennoinen, A., and Lehtinen, J. (2017). "Real-time Global Illumination by Precomputed Local Reconstruction from Sparse Radiance Probes." ACM TOG 36(6), SIGGRAPH Asia 2017. https://arisilvennoinen.github.io/Projects/RTGI/index.html

[^11]: Stefanov, N. (2016). "Global Illumination in Tom Clancy's The Division." GDC 2016. https://mrakobes.com/Nikolay.Stefanov.GDC.2016.pdf

[^12]: Wang, Y., Khiat, S., Kry, P. G., and Nowrouzezahrai, D. (2019). "Fast Non-uniform Radiance Probe Placement and Tracing." I3D 2019. https://cs.mcgill.ca/~kry/pubs/i3d2019/NonUniformProbes-paper.pdf

[^13]: Encoding and baking references: [Ray Guiding for Production Lightmap Baking](https://arisilvennoinen.github.io/Publications/ray_guiding_For_production_lightmap_baking_author_version.pdf), [UberBake](https://www.ppsloan.org/publications/seyb20uberbake.pdf), [ZH3](https://www.ppsloan.org/publications/ZH3.pdf), [Precomputed Global Illumination in Frostbite](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-precomputedgiobalilluminationinfrostbite.pdf), and [The Indirect Lighting Pipeline of God of War](https://www.gdcvault.com/play/1026323/The-Indirect-Lighting-Pipeline-of). Citations to the Chinese-language material refer to the corresponding files under `E:/Paper/GI/` and the page numbers given above.
