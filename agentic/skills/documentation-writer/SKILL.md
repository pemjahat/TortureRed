---
name: documentation-writer
description: >
  Write any markdown document or Mermaid diagram following the project's
  documentation standards. Use when creating .md files, Mermaid diagrams,
  READMEs, docs/, or any documentation output.
---

# Documentation Writer

## Style Guides — Read Before Creating

| Creating...                  | Read first                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------ |
| Any `.md` document           | [markdown_style_guide.md](markdown_style_guide.md)                             |
| Any Mermaid diagram          | [mermaid_style_guide.md](mermaid_style_guide.md), then [specific type file](mermaid_diagrams/) |

## Key Rules Enforced by the Style Guides

- **Diagrams** — `accTitle` + `accDescr` on every diagram, `classDef` color classes (no inline `style`), emoji on key nodes
- **Structure** — one H1, emoji on H2 only, no H5+, horizontal rules after `</details>` blocks
- **Everything is Code** — all markdown files in `docs/taskNNN-{short-descrption}.md`

## Workflow

1. Identify whether the task requires a `.md` document, a Mermaid diagram, or both.
2. Read the relevant style guide(s) listed in the table above **before writing anything**.
3. Compose the output following the style guide's formatting, structure, and accessibility rules.
4. Place output files in the correct directory (`docs/` for markdown, or wherever the style guide mandates).
