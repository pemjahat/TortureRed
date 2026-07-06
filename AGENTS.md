# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. First action: repository discovery

Before editing, map these project. Use fast search commands similar to these, adjusted for the host shell and available tools:

```bash
pwd
git status --short
find . -maxdepth 3 -type f \( -iname "*path*trace*" -o -iname "*ray*trace*" -o -iname "*shader*" -o -iname "*.hlsl" -o -iname "*.hlsli" -o -iname "CMakeLists.txt" \) | sort
rg -n "PathTrace|PathTracer|Pathtracer|TraceRay|DispatchRays|DispatchRay|RayGen|closesthit|miss|bouncesMax|samplesPerPixel" .
rg -n "BindingSet|Descriptor|DescriptorSet|RootSignature|register\(|space[0-9]|UAV|SRV|StructuredBuffer|RWStructuredBuffer|ByteAddressBuffer" .
rg -n "ShaderTable|Pipeline|Permutation|Define|Macro|CompileShader|DXC|ImGui|CMake|add_shader|shader compile" .
rg -n "SHARC|Sharc|HashGrid|NRC|NRD|RTXGI" .
rg -n "RestirDI|RestirGI|Temporal|Spatial|Resolve" .
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
- denoiser input generation, if present
- GPU resource creation and clear/barrier helpers

## 2. Skills

Skills follow the [Agent Skills Specification](https://agentskills.io/specification). Every skill lives under `agentic/skills/<skill-name>/` and is a self-contained folder.

### Structure

```
agentic/skills/<skill-name>/
├── SKILL.md              # Required: YAML frontmatter (name, description) + instructions
```

**Rules:**
- `SKILL.md` is the entry point — starts with `---` delimited YAML containing `name` and `description`
- All files referenced by `SKILL.md` must reside within the skill's own folder
- The `description` field determines when the skill is matched and loaded

### Installed Skills

| Skill | Path | Purpose |
| ----- | ---- | ------- |
| documentation-writer | [SKILL.md](agentic/skills/documentation-writer/SKILL.md) | Write `.md` documents and Mermaid diagrams following project style guides |

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
