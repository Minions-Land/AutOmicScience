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

The script uses `hf upload-large-folder` for large asset directories so interrupted uploads can be resumed by rerunning the same command. Reference assets are uploaded from `src/bridge/runtime` with include filters so `data/`, `external/SEA-AD/`, and `vendor/.../UCE-main/data/` keep their expected paths.

Token-based upload also works:

```bash
HF_TOKEN=hf_xxx node scripts/upload-hf-assets.mjs
```

Preview commands without uploading:

```bash
node scripts/upload-hf-assets.mjs --dry-run
```

Generate repository cards and file manifests without uploading:

```bash
node scripts/prepare-hf-assets.mjs
```

If `hf` is installed outside your shell `PATH`, point the script at it:

```bash
AOS_HF_CLI=/root/anaconda3/bin/hf node scripts/upload-hf-assets.mjs
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
