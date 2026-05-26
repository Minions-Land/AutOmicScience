# AutOmicScience

AutOmicScience (AOS) is a bioinformatics multi-agent system for omics analysis. It combines an engineering-oriented agent runtime, local tools, a browser console, and an optional Python bridge for single-cell annotation, model selection, foundation-model adapters, reference transfer, and consensus workflows.

The core code is intentionally lightweight. Large foundation-model weights and biological reference collections are optional assets and are distributed separately from the Git repository.

## Repositories

- Code: https://github.com/PoorOtterBob/AutOmicScience
- Code mirror: https://github.com/Minions-Land/AutOmicScience
- Optional assets: https://huggingface.co/PoorOtterBob

Recommended Hugging Face layout:

```text
PoorOtterBob/AutOmicScience-FoundationModels
PoorOtterBob/AutOmicScience-Reference
```

If those repositories are renamed later, keep the local directory layout below and update the download URLs in your environment setup.

## What Is Included

- `aos` CLI for chat, one-shot runs, structured IO, UI serving, setup, store management, and Bio MAS commands.
- Browser console at `/` for model selection, agent runs, permissions, tasks, plugins, sessions, project instructions, and hooks.
- AOS compatibility frontend at `/aos/` with HTTP and optional NATS RPC endpoints.
- TypeScript agent runtime with tools for files, shell, Python, notebooks, web, task management, skills, plugins, MCP, and bioinformatics workflows.
- Python bridge runtime in `src/bridge/runtime/aos_agent` for Bio MAS workflows.
- Tiny synthetic demo generation for smoke tests.

Not included by default:

- Foundation-model checkpoints.
- Full reference collections.
- Large real query datasets.

## Quick Start

```bash
git clone https://github.com/PoorOtterBob/AutOmicScience.git
cd AutOmicScience
npm install
npm run typecheck
npm test
npm run build
```

Start the browser console:

```bash
npm run dev -- serve --port 3127
```

Open:

```text
http://localhost:3127
http://localhost:3127/aos/
```

Start an interactive CLI session:

```bash
npm run dev -- cli
```

Run one prompt:

```bash
npm run dev -- run "List the available Bio MAS tools and explain when to use each one."
```

After build, the package exposes the `aos` bin from `dist/cli/index.js`.

## Interface Checks

CLI checks:

```bash
node dist/cli/index.js --version
node dist/cli/index.js --help
node dist/cli/index.js annotate bio-mas-preflight
```

Frontend checks:

```bash
npm run dev -- serve --port 3127
```

Open `http://localhost:3127` and `http://localhost:3127/aos/`. The UI uses AutOmicScience/AOS text branding only; the previous company logo image and favicon route are intentionally removed.

## Model Provider Configuration

Run the setup wizard:

```bash
npm run dev -- setup
```

Or create `~/.aos/.env` / `.env` with provider settings.

OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
AOS_MODEL=gpt-5.5
```

AOS-prefixed OpenAI settings are also supported:

```bash
AOS_OPENAI_API_KEY=your_key
AOS_OPENAI_BASE_URL=https://your-compatible-endpoint/v1
AOS_MODEL=gpt-5.5
```

Gemini:

```bash
GOOGLE_API_KEY=your_key
AOS_MODEL=gemini-2.5-flash
```

Anthropic-compatible provider:

```bash
ANTHROPIC_API_KEY=your_key
AOS_MODEL=anthropic/provider-model-id
```

Do not commit real API keys.

## Python Bridge

The Python bridge is required only for Bio MAS workflows. It is not needed for the core TypeScript agent runtime, CLI, or UI.

```bash
cd src/bridge/runtime
uv venv
uv pip install -e ".[llm,foundation,notebook,test]"
cd ../../..
```

Point AOS at the Python environment if needed:

```bash
export AOS_PYTHON_RUNTIME="$PWD/src/bridge/runtime"
export AOS_PYTHON_BIN="$PWD/src/bridge/runtime/.venv/bin/python"
```

On Windows PowerShell:

```powershell
$env:AOS_PYTHON_RUNTIME = "$PWD\src\bridge\runtime"
$env:AOS_PYTHON_BIN = "$PWD\src\bridge\runtime\.venv\Scripts\python.exe"
```

## Bio MAS Smoke Test

The tiny demo uses synthetic data and does not require optional large assets.

```bash
npm run dev -- annotate bio-mas-preflight
npm run dev -- annotate create-tiny-bio-demo -- --output-dir runs/bio_mas_tiny_demo
npm run dev -- annotate run-tiny-bio-mas-demo -- --output-dir runs/bio_mas_tiny_demo --cells-per-label 6 --top-k 1
```

Expected smoke-test properties:

```text
synthetic_tiny_demo: true
scientific_use: smoke_test_only
n_metric_rows >= 1
n_prediction_rows >= 1
n_skips = 0
```

Tiny demo outputs are engineering checks only. They are not scientific results.

## Optional Foundation Models And Reference Assets

Large assets are optional. AOS should run basic CLI/UI/tests without them. Bio MAS production workflows will report blocked or skipped stages when required assets are missing.

Asset publishing and download details are also documented in `HUGGINGFACE_ASSETS.md`.

Recommended local layout:

```text
src/bridge/runtime/checkpoints/foundation_models/
  geneformer/
  scgpt/
    brain/
    human/
  nicheformer/
  uce_4l/
  uce_33l/
  scANVI/
  sklearn_baselines/
  spatial_gnn/

src/bridge/runtime/data/reference/
src/bridge/runtime/data/raw/
src/bridge/runtime/data/query/
src/bridge/runtime/data/prepared_sources/
src/bridge/runtime/external/SEA-AD/MJM/
```

Download examples using the Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download PoorOtterBob/AutOmicScience-FoundationModels \
  --repo-type model \
  --local-dir src/bridge/runtime/checkpoints/foundation_models

hf download PoorOtterBob/AutOmicScience-Reference \
  --repo-type dataset \
  --local-dir src/bridge/runtime
```

Publishing from the prepared server copy:

```bash
hf auth login
node scripts/upload-hf-assets.mjs
```

Equivalent Python download:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PoorOtterBob/AutOmicScience-FoundationModels",
    repo_type="model",
    local_dir="src/bridge/runtime/checkpoints/foundation_models",
)
snapshot_download(
    repo_id="PoorOtterBob/AutOmicScience-Reference",
    repo_type="dataset",
    local_dir="src/bridge/runtime",
)
```

If assets live outside the repository, set environment variables:

```bash
export AOS_MAS_ROOT="$PWD/src/bridge/runtime"
export AOS_MAS_FOUNDATION_MAS_ROOT="$PWD/src/bridge/runtime/vendor/foundation_model_based_mas"
export AOS_MAS_FOUNDATION_CHECKPOINT_ROOT="/path/to/foundation_models"
export AOS_MAS_SEAAD_MJM_ROOT="/path/to/SEA-AD/MJM"
export AOS_MAS_SEAAD_MERFISH_H5AD="/path/to/seaad_merfish.h5ad"
export AOS_MAS_SEAAD_DONOR_H5AD_DIR="/path/to/seaad_donor_h5ad"
export AOS_MAS_KUKANJA_MS_NPZ="/path/to/kukanja_ms.npz"
export AOS_MAS_KUKANJA_EAE_NPZ="/path/to/kukanja_eae.npz"
export AOS_MAS_IMA_REFERENCE_H5AD="/path/to/IMA_sample.h5ad"
export AOS_MAS_GENEFORMER_DIR="/path/to/foundation_models/geneformer"
export AOS_MAS_SCGPT_DIR="/path/to/foundation_models/scgpt"
export AOS_MAS_NICHEFORMER_DIR="/path/to/foundation_models/nicheformer"
export AOS_MAS_UCE_4L_DIR="/path/to/foundation_models/uce_4l"
export AOS_MAS_UCE_33L_DIR="/path/to/foundation_models/uce_33l"
export AOS_MAS_UCE_MODEL_PY="$PWD/src/bridge/runtime/vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/model.py"
```

Then re-run:

```bash
npm run dev -- annotate bio-mas-preflight
```

Correct missing-asset behavior:

- `bio-mas-preflight` reports missing paths.
- The corresponding workflow stages are blocked or skipped.
- Foundation-model stages do not execute without checkpoints.
- AOS does not convert tiny demo results into scientific claims.

## GPU Safety

On the current server, do not use GPU 0. Prefer CPU for smoke tests or explicitly choose another visible GPU.

```bash
export CUDA_VISIBLE_DEVICES=1
```

For lightweight checks, use CPU-only commands and avoid foundation-model workflows unless assets and GPU placement are configured.

## UI And Compatibility API

Start the server:

```bash
npm run dev -- serve --port 3127
```

Optional compatibility data directory:

```bash
npm run dev -- serve --port 3127 --aos-data-dir .aos/aos-compat
```

Readiness endpoint:

```text
GET http://localhost:3127/api/aos/ready
```

RPC endpoint:

```text
POST http://localhost:3127/api/aos/rpc
```

The response includes `service_id`, `service_subject`, and NATS status when NATS is enabled. HTTP compatibility remains available even if NATS is unavailable.

## Common Commands

```bash
npm run typecheck
npm test
npm run build
npm run dev -- serve --port 3127
npm run dev -- cli
npm run dev -- run "Summarize this repository."
npm run dev -- annotate bio-mas-preflight
```

## Development Notes

- User configuration is stored under `~/.aos` by default.
- Project-local compatibility state can be stored under `.aos/aos-compat`.
- Frontend and CLI user-facing names should remain AutOmicScience/AOS.
- The frontend logo image has been removed; the UI uses text branding and no commercial company logo.
- `src/bridge/runtime/data`, `src/bridge/runtime/checkpoints`, `src/bridge/runtime/artifacts`, and `runs` are runtime/asset locations and should not be treated as required code for a lightweight install.
