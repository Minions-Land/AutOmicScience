# scmas plugin

Multi-stage single-cell annotation pipeline ported from CanChen_MAS into
the PantheonOS-ts plugin shape.

## Layout

- `tools/` — subprocess wrappers around `python -m scmas <subcmd>`
- `agents/` — Stage-2 / Stage-3 / Stage-4 LLM agents (PantheonOS-ts `Agent`s)
- `skills/scmas-annotation.md` — pipeline-overview skill, loadable via `FileSkillLoader`
- `team/ScmasPipeline.ts` — `Sequential` of the three LLM agents
- `prompts/` — system-prompt markdowns lifted from the Python source
- `schemas/` — zod schemas mirroring each LLM call's response contract

## Why a Python subprocess bridge

Stage-1 evaluation, scDesign3 synthetic generation, label transfer, and
UCE IMA all depend on PyTorch / scanpy / sklearn / R, which cannot be
faithfully re-implemented in TypeScript. The bridge lets PantheonOS-ts
orchestrate the pipeline end-to-end while the heavy compute stays where
it already works. Vendored Python source lives at
`_import_scMAS/CanChen_MAS/`.

## Environment

| Variable | Purpose |
|---|---|
| `SCMAS_PYTHON_BIN` | python interpreter (default `python`) |
| `SCMAS_ROOT` | scmas project root (default: vendored `_import_scMAS/CanChen_MAS`) |
| `OPENAI_API_KEY` | required when an LLM stage is in `required` mode |
| `OPENAI_BASE_URL` | optional gateway override |
| `SCMAS_LLM_MODEL` | overrides `OPENAI_MODEL` |

## Quick start

```ts
import { createScmasPipeline, scmasToolSet, loadScmasAnnotationSkill }
  from 'pantheon-ts/plugins/scmas';

const team = await createScmasPipeline({ model: 'gpt-4o-mini' });
const skill = await loadScmasAnnotationSkill();
const tools = scmasToolSet();

console.log(skill.name, tools.size(), team.name);
```

## Plugin seam

`registerScmas(host, opts)` accepts an arbitrary host with optional
`registerTool / registerAgent / registerSkill / registerTeam` callbacks.
The whole plugin can be wired into any host without coupling to a
specific one.

## TODO seams left for the user

- The bridge currently passes all output through stdout/stderr text.
  Wire it to a structured event channel if you want streaming progress.
- The Stage-2 prompt contains the pipeline rules; a richer copy of the
  Python `_render_llm_prompt` payload would tighten parity.
- `validateAllowedLabels` in the Stage-4 schema is provided but not
  invoked automatically. Hook it after the agent run when you need
  client-side enforcement.
