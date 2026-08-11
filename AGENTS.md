# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. First action: repository discovery

Before editing, map the project. **This same process and location list applies to any codebase this project consults — this repository itself, or another workspace/repository referenced as a sample or reference implementation (see [Section 2](#2-referencing-other-repositories)).** Never maintain a separate discovery checklist for other repositories; always come back to this list.

Use fast search commands similar to these, adjusted for the host shell and available tools:

```bash
pwd
git status --short
find . -maxdepth 3 -type f \( -iname "*path*trace*" -o -iname "*ray*trace*" -o -iname "*shader*" -o -iname "*.hlsl" -o -iname "*.hlsli" -o -iname "*.slang" -o -iname "*.glsl" -o -iname "CMakeLists.txt" \) | sort
rg -n "PathTrace|PathTracer|Pathtracer|TraceRay|DispatchRays|DispatchRay|RayGen|closesthit|miss|bouncesMax|samplesPerPixel" .
rg -n "BindingSet|Descriptor|DescriptorSet|RootSignature|register\(|space[0-9]|UAV|SRV|StructuredBuffer|RWStructuredBuffer|ByteAddressBuffer" .
rg -n "ShaderTable|Pipeline|Permutation|Define|Macro|CompileShader|DXC|Slang|ImGui|CMake|add_shader|shader compile" .
rg -n "SHARC|Sharc|HashGrid|NRC|NRD|RTXGI" .
rg -n "RestirDI|RestirGI|Temporal|Spatial|Resolve" .
rg -n "Deferred|Upscaling|TAA|Meshoptimizer|Meshlet|Occlusion Culling" .
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
- model vertices/triangles processing
- denoiser input generation, if present
- GPU resource creation and clear/barrier helpers

## 2. Referencing other repositories

When a task requires research or investigation for the feature currently being prompted, check whether it references another workspace or repository (e.g., a sibling directory with a reference engine or sample). Study the referenced implementation **before** designing or changing code in this repository.

Run the exact same [Section 1](#1-first-action-repository-discovery) discovery process — the same search commands, the same project-specific locations list — against the referenced repository instead of inventing a new one. Section 1 is the single canonical discovery checklist regardless of which repository is being mapped.

Key rules:

- **Locate first** — find the referenced repository/workspace on disk before editing anything
- **Discover the same way** — apply Section 1's search commands and location list to the referenced repository; do not use a separate checklist
- **Cite precisely** — record the file paths, function names, and line ranges used as the reference
- **Document before coding** — write down what is wrong or what will change in `docs/` before making code changes

## 3. Documentation standards

When writing any `.md` document, Mermaid diagram, `README`, or documentation output, follow these rules.

### Style guides — read before creating

| Creating...                  | Read first                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------- |
| Any `.md` document           | [markdown_style_guide.md](agentic/markdown_style_guide.md)                   |
| Any Mermaid diagram          | [mermaid_style_guide.md](agentic/mermaid_style_guide.md), then the specific diagram type in [mermaid_diagrams/](agentic/mermaid_diagrams/) |
| A specific document type     | Check [markdown_templates/](agentic/markdown_templates/) for a template first |

### Key rules

- **Diagrams** — accessibility caption on every diagram: italic Markdown paragraph directly above the Mermaid code block (works in all renderers); `accTitle`/`accDescr` inside the block are a newer-directive alternative but render as broken boxes in older viewers; `classDef` color classes (no inline `style`); emoji on key nodes
- **Structure** — one H1, emoji on H2 only, no H5+, horizontal rules after `</details>` blocks
- **Naming** — all markdown files in `docs/taskNNN-{short-description}.md`

### Workflow

1. Identify whether the task requires a `.md` document, a Mermaid diagram, or both.
2. Read the relevant style guide(s) listed in the table above **before writing anything**.
3. Compose the output following the style guide's formatting, structure, and accessibility rules.
4. Place output files in the correct directory (`docs/` for markdown, or wherever the style guide mandates).

### Available templates

| Template | Use for |
| -------- | ------- |
| [decision_record.md](agentic/markdown_templates/decision_record.md) | Architectural decisions |
| [how_to_guide.md](agentic/markdown_templates/how_to_guide.md) | Step-by-step guides |
| [project_documentation.md](agentic/markdown_templates/project_documentation.md) | Project overview docs |
| [status_report.md](agentic/markdown_templates/status_report.md) | Progress reports |

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation rather than after mistakes, and **all documentation output follows the project style guides from the first draft.**
