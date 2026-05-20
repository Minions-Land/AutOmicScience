---
name: scmas-annotation
description: Multi-stage single-cell / spatial-transcriptomics annotation pipeline ported from CanChen_MAS. Selects (source, model) pairs, executes raw label-transfer adapters, and runs cross-model consensus with optional LLM adjudication of low-consistency cells.
---

# scMAS Annotation Pipeline

`scmas` is a four-stage no-training annotation workflow for single-cell and
spatial transcriptomics data. It is ported from the CanChen_MAS Python
project; deterministic scientific compute (anndata, scanpy, sklearn, torch,
scDesign3) stays in vendored Python, exposed through PantheonOS-ts Tools.
LLM-driven planning, adapter generation, and low-consistency adjudication
run as PantheonOS-ts Agents.

## Stages

1. **Stage 1 — capability evaluation** (deterministic): build references,
   prepare sources, run scDesign3 synthetic perturbations, evaluate the model
   registry, produce per-(source, model) capability scores.
2. **Stage 2 — selector** (LLM): given a query profile, the selector agent
   picks `top_k` execution-ready (source, model) pairs from candidate cards,
   honoring family caps and gene-fit / robustness / benchmark evidence.
3. **Stage 3 — adapter** (LLM): for each selected pair the adapter agent
   emits an `AdapterSpec` whose actions are drawn from `ALLOWED_ACTIONS`. A
   deterministic reviewer rejects unknown actions, mutated immutable fields,
   missing paths, or unsafe keys.
4. **Stage 4 — consensus + adjudicator** (deterministic + optional LLM):
   run reference-enhanced consensus fusion across all completed adapters;
   for low-consistency cells the adjudicator agent picks one
   `selected_label` per evidence group from `allowed_labels`.

## Tools used by the skill

Subprocess wrappers around `python -m scmas <subcommand>`:

- `scmas_build_label_maps`, `scmas_build_reference`, `scmas_build_seaad_test`,
  `scmas_prepare_sources`, `scmas_build_dataset_catalog`
- `scmas_write_scdesign3_configs`, `scmas_preflight_scdesign3`,
  `scmas_run_scdesign3`
- `scmas_prepare_eval_datasets`, `scmas_evaluate`,
  `scmas_raw_label_transfer_smoke`, `scmas_run_uce_ima_transfer`
- `scmas_profile_query`, `scmas_select_models`,
  `scmas_run_cross_species_plan`, `scmas_inspect_model_contracts`,
  `scmas_adapt_and_execute`, `scmas_run_consensus`

## Required environment

```bash
SCMAS_PYTHON_BIN=python                    # default
SCMAS_ROOT=/path/to/CanChen_MAS            # default: vendored _import_scMAS
OPENAI_API_KEY=...                         # required when llm-mode != off
OPENAI_BASE_URL=https://.../v1             # optional gateway override
SCMAS_LLM_MODEL=gpt-5.4-mini               # overrides OPENAI_MODEL
```

## Typical invocation

```ts
import { createScmasPipeline } from 'pantheon-ts/plugins/scmas';

const team = await createScmasPipeline({
  model: process.env.SCMAS_LLM_MODEL ?? 'gpt-4o-mini',
});
const result = await team.runToText(JSON.stringify({
  query_profile: 'runs/stage2/my_query/query_profile.json',
  capability_dir: 'configs/capability',
  prepared_source_root: 'data/prepared_sources',
  artifact_bundle: 'artifacts/stage1_full',
  top_k: 3,
}));
```

The team is a `Sequential` of:
`Stage2Selector → Stage3Adapter → Stage4Adjudicator`.

## Hard rules carried over from CanChen_MAS

- Stage-2 must not infer from hidden labels, query expression values, query
  paths, or Stage-4 results. Species mismatch is not a hard filter; the
  gene/input contract is the execution gate.
- Stage-3 must use only the actions in `ALLOWED_ACTIONS`. SEA-AD 140-gene
  trained-head invocations are forbidden — only raw label-transfer / kNN
  adapters.
- Stage-4 adjudication may not request or infer hidden truth labels. Each
  group gets exactly one `selected_label` from `allowed_labels`, falling
  back to `unknown_label` when evidence is insufficient.
