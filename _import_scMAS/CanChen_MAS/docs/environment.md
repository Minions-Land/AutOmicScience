# Environment

CanChen_MAS uses one merged conda environment for the Python MAS/LLM runtime and the R scDesign3 pipeline.

## Create the merged environment

```bash
cd /data1/CanChen_MAS
conda env create -f environment.yml
conda activate canchen_mas
Rscript scripts/setup_scdesign3_r_packages.R
pip install -e .
```

`environment.yml` is a pragmatic merge of the two local environments used during cleanup:

- `fdm_mas`: Python MAS, OpenAI-compatible LLM runtime, LangChain/LangGraph, torch/transformers, scanpy/anndata.
- `scdesign3-pipeline`: R 4.3, scDesign3 runtime dependencies, SingleCellExperiment/Matrix/yaml.

The merged environment uses Python 3.11 because it is the safer shared target for the R/scDesign3 toolchain and the Python MAS stack.

## Check the environment

```bash
PYTHONPATH=src python -m compileall -q src
PYTHONPATH=src pytest -q tests
python -m scmas preflight-scdesign3 --rscript-path Rscript
```

If you prefer not to activate the environment before running scDesign3, pass the conda wrapper explicitly:

```bash
python -m scmas preflight-scdesign3 --rscript-path conda:canchen_mas
```

## LLM gateway

The LLM calls use the OpenAI-compatible chat API. Put credentials in `.env` or export them in the shell:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://your-openai-compatible-gateway/v1
OPENAI_MODEL=
SCMAS_LLM_MODEL=
OPENAI_TIMEOUT=60
OPENAI_MAX_RETRIES=2
OPENAI_TRUST_ENV=true
```

`SCMAS_LLM_MODEL` takes precedence for CanChen_MAS. If it is empty, the code falls back to `OPENAI_MODEL`, then `LLM_MODEL`, then the built-in default.

Use `--llm-mode required` or `--llm-policy-mode required` when the run must fail on LLM/network errors. Use `optional` when you want the pipeline to attempt LLM planning and then fall back to deterministic reviewers if the gateway is unavailable.
