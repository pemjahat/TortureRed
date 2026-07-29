# Repository Research Instruction

This file provides repository instructions for agentic coding tools. It is intended to be read whenever another repository or workspace is mentioned as a reference. Read this entire file before making any decisions.

## 1. Required agentic working rules

1. Start by inspecting the repository structure.
2. Do not make destructive changes.
3. Prefer minimal, idiomatic changes that match the project's existing render-pass, descriptor, shader-compilation, resource-management, naming, and configuration architecture.
4. Preserve existing material, camera, scene, acceleration structure, denoising, accumulation, UI/config, and debug systems unless a change is required for the new feature.
5. Make new feature optional through a compile-time flag and, when the project has runtime settings, a runtime toggle.
6. Do not introduce large new framework dependencies.
7. Do not hard-code local absolute paths.
8. Work in small logical patches internally, but keep moving through the complete feature integration when implementation was requested.
9. Do not stop after scaffolding, a plan, or only one render pass unless blocked by missing SDK files, unsupported renderer architecture, unavailable build tools, or missing project-specific information that prevents safe code changes.
10. Fix compile and shader-compile errors caused by the new changes when local build tools are available.
11. After implementation, run the project's normal configure/build or shader-compile checks if available.
12. If a build cannot be run locally, still run static checks such as file search, formatting checks, shader compile scripts, or CMake generation where available.
13. In the final response, list changed files, explain the new feature render flow, list new build flags/runtime toggles/shader defines, report build commands run, provide manual validation steps, and call out assumptions and unresolved issues.

## 2. First action: other repository discovery

Begin with repository or workspace discovery and immediately map these project-specific targets.
Use fast search commands similar to these, adjusted for the host shell and available tools:

```bash
pwd
git status --short
find . -maxdepth 3 -type f \( -iname "*path*trace*" -o -iname "*ray*trace*" -o -iname "*shader*" -o -iname "*.hlsl" -o -iname "*.hlsli" -o -iname "*.slang" -o -iname "*.glsl" -o -iname "CMakeLists.txt" \) | sort
rg -n "PathTrace|PathTracer|Pathtracer|TraceRay|DispatchRays|DispatchRay|RayGen|closesthit|miss|bouncesMax|samplesPerPixel" .
rg -n "BindingSet|Descriptor|DescriptorSet|RootSignature|register\(|space[0-9]|UAV|SRV|StructuredBuffer|RWStructuredBuffer|ByteAddressBuffer" .
rg -n "ShaderTable|Pipeline|Permutation|Define|Macro|CompileShader|DXC|Slang|CMake|add_shader|shader compile" .
rg -n "Deferred|Upscaling|TAAA|Meshoptimizer|Meshlet|Occlusion Culling" .
```

Find these project-specific locations:

- build system files
- shader include directories
- shader permutation/define setup
- ray tracing pipeline creation
- shader binding table creation
- render loop or render graph pass scheduling
- output render target writing
- descriptor/register-space definitions
- global/per-frame constant buffer definitions
- existing debug UI, runtime settings, config files, and command-line parsing
- path tracer ray generation shader
- path tracing bounce loop
- material sampling and direct lighting evaluation
- scene or model loading
- model vertices, triangles processing
- denoiser input generation, if present
- GPU resource creation and clear/barrier helpers

## 3. Expected high-level host render loop

Adapt to the engine's render graph or command-list style.

If the engine has a render graph, express the same dependencies in graph edges. Do not rely on implicit ordering if the graph requires explicit UAV/resource dependencies.

## 4. Descriptor/register binding

Adapt register spaces to the target engine. Keep bindings stable across Update, Resolve, and Query passes. Barriers must still make writes visible.

## 5. Constant-buffer additions

Respect packing/alignment rules.

If the project already has camera constants, reuse them when possible.

## 6. Pipeline/permutation integration

Use the project's existing permutation system.

If the engine has shader hot reload or shader reflection, update metadata accordingly.

## 7. Final response requirements

The final response must include:

```text
Summary:
- new investigation render flow and fallback behavior

Assumptions:
- ...

Known issues / follow-ups:
- ...
```