# Hugging Face Optional Assets

AutOmicScience keeps large assets out of GitHub. The code works for CLI, UI, tests, and tiny Bio MAS smoke checks without these files. Production Bio MAS workflows can download optional assets from Hugging Face.

## Target Repositories

```text
PoorOtterBob/AutOmicScience-FoundationModels  # repo type: model
PoorOtterBob/AutOmicScience-Reference         # repo type: dataset
```

Set `AOS_HF_OWNER` if the repositories are published under another Hugging Face owner.

## What To Upload

Foundation model repository:

```text
src/bridge/runtime/checkpoints/foundation_models/ -> .
```

Reference dataset repository:

```text
src/bridge/runtime/data/ -> data/
src/bridge/runtime/external/SEA-AD/ -> external/SEA-AD/
src/bridge/runtime/vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/data/
  -> vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/data/
```

The server copy used for the rebrand contained approximately 45G of checkpoints, 40G of data, 144M of external SEA-AD assets, and 22G of UCE data.

## Upload

Install the Hub CLI and authenticate:

```bash
pip install -U huggingface_hub
hf auth login
```

Then run from the repository root:

```bash
node scripts/upload-hf-assets.mjs
```

Token-based upload also works:

```bash
HF_TOKEN=hf_xxx node scripts/upload-hf-assets.mjs
```

Preview commands without uploading:

```bash
node scripts/upload-hf-assets.mjs --dry-run
```

## Download

```bash
pip install -U huggingface_hub

hf download PoorOtterBob/AutOmicScience-FoundationModels \
  --repo-type model \
  --local-dir src/bridge/runtime/checkpoints/foundation_models

hf download PoorOtterBob/AutOmicScience-Reference \
  --repo-type dataset \
  --local-dir src/bridge/runtime
```

If assets are stored outside the repository, set the `AOS_MAS_*` paths documented in `README.md`.
