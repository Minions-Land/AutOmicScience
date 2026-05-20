---
name: annotation-pipeline
description: Built-in multi-stage single-cell / spatial-transcriptomics annotation pipeline. Selects (source, model) pairs, executes raw label-transfer adapters, and runs cross-model consensus with optional LLM adjudication of low-consistency cells.
---

# Annotation Pipeline

A four-step no-training annotation workflow for single-cell and spatial
transcriptomics data, built into MedrixAI. Deterministic scientific
compute (anndata, scanpy, sklearn, torch, scDesign3) runs in the bundled
Python runtime at `src/bridge/runtime/` and is invoked through the
MedrixAI `bridge/` seam; LLM-driven planning, adapter generation, and
low-consistency adjudication run as MedrixAI `Agent`s.

## Steps

1. **Capability evaluation** (deterministic): build references, prepare
   sources, run synthetic perturbations, evaluate the model registry,
   produce per-(source, model) capability scores. Tools:
   `bio_*`, `synth_*`, `bench_*`.
2. **Selector** (LLM): given a query profile, the `Selector` agent picks
   `top_k` execution-ready (source, model) pairs from candidate cards,
   honoring family caps and gene-fit / robustness / benchmark evidence.
   Tool: `annotate_select_sources`.
3. **Adapter** (LLM): for each selected pair the `Adapter` agent emits
   an `AdapterSpec` whose actions are drawn from `ALLOWED_ACTIONS`. The
   deterministic reviewer rejects unknown actions, mutated immutable
   fields, missing paths, or unsafe keys. Tool:
   `annotate_adapt_and_execute`.
4. **Consensus + Adjudicator** (deterministic + optional LLM): run
   reference-enhanced consensus fusion across all completed adapters;
   for low-consistency cells the `Adjudicator` agent picks one
   `selected_label` per evidence group from `allowed_labels`. Tool:
   `annotate_run_consensus`.

## Tools used by the skill

All names live in the core toolsets — no plugin namespace:

- `bio_build_label_maps`, `bio_build_reference`, `bio_build_seaad_test`,
  `bio_prepare_sources`, `bio_build_dataset_catalog`
- `synth_write_configs`, `synth_preflight`, `synth_generate`
- `bench_prepare_datasets`, `bench_evaluate`,
  `bench_label_transfer_smoke`, `bench_uce_label_transfer`
- `annotate_profile_query`, `annotate_select_sources`,
  `annotate_run_plan`, `annotate_inspect_contracts`,
  `annotate_adapt_and_execute`, `annotate_run_consensus`

## Required environment

```bash
MEDRIX_PYTHON_BIN=python                  # default
MEDRIX_PYTHON_RUNTIME=src/bridge/runtime  # default: bundled runtime/
OPENAI_API_KEY=...                         # required when llm-mode != off
OPENAI_BASE_URL=https://.../v1             # optional gateway override
MEDRIX_MODEL=gpt-4o-mini                  # default LLM model id
```

## Typical invocation

```ts
import { createAnnotationPipeline } from 'medrix-ai';

const team = await createAnnotationPipeline({
  model: process.env.MEDRIX_MODEL ?? 'gpt-4o-mini',
});
const result = await team.runToText(JSON.stringify({
  query_profile: 'runs/selection/my_query/query_profile.json',
  capability_dir: 'configs/capability',
  prepared_source_root: 'data/prepared_sources',
  artifact_bundle: 'artifacts/capability_eval',
  top_k: 3,
}));
```

The team is a `Sequential` of:
`Selector → Adapter → Adjudicator`.

## Hard rules

- The Selector must not infer from hidden labels, query expression
  values, query paths, or downstream consensus / adjudication results.
  Species mismatch is not a hard filter; the gene/input contract is the
  execution gate.
- The Adapter must use only the actions in `ALLOWED_ACTIONS`. SEA-AD
  140-gene trained-head invocations are forbidden — only raw label-
  transfer / kNN adapters.
- The Adjudicator may not request or infer hidden truth labels. Each
  group gets exactly one `selected_label` from `allowed_labels`,
  falling back to `unknown_label` when evidence is insufficient.
