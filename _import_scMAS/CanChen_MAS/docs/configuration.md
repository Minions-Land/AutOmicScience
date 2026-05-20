# Configuration

CanChen_MAS 的默认路径都从 `src/scmas/paths.py` 派生，可以通过环境变量覆盖。提交版不会附带数据、权重或实验产物。

## 推荐目录

```text
data/
├── raw/
├── query/
├── reference/
├── prepared_sources/
└── synthetic/

checkpoints/foundation_models/
├── geneformer/
├── scgpt/
├── nicheformer/
├── uce_4l/
└── uce_33l/
```

`data/` 和 `checkpoints/foundation_models/` 默认只提交 `.gitkeep`。

## 环境变量

| Variable | Purpose |
| --- | --- |
| `CANCHEN_MAS_ROOT` | 仓库根目录；默认自动推断。 |
| `CANCHEN_MAS_FOUNDATION_MAS_ROOT` | 精简 vendor 或外部 foundation MAS 根目录。 |
| `CANCHEN_MAS_SEAAD_MJM_ROOT` | SEA-AD MJM wrapper/model code 根目录。 |
| `CANCHEN_MAS_SEAAD_MERFISH_H5AD` | SEA-AD MERFISH 原始 h5ad。 |
| `CANCHEN_MAS_SEAAD_DONOR_H5AD_DIR` | SEA-AD donor h5ad 目录。 |
| `CANCHEN_MAS_KUKANJA_MS_NPZ` | Kukanja MS 查询 npz。 |
| `CANCHEN_MAS_KUKANJA_EAE_NPZ` | Kukanja EAE 查询 npz。 |
| `CANCHEN_MAS_FOUNDATION_CHECKPOINT_ROOT` | Foundation model 权重总目录。 |
| `CANCHEN_MAS_GENEFORMER_DIR` | Geneformer 权重目录。 |
| `CANCHEN_MAS_SCGPT_DIR` | scGPT 权重目录。 |
| `CANCHEN_MAS_NICHEFORMER_DIR` | Nicheformer 权重目录。 |
| `CANCHEN_MAS_UCE_4L_DIR` | UCE 4-layer 权重目录。 |
| `CANCHEN_MAS_UCE_33L_DIR` | UCE 33-layer 权重目录。 |
| `CANCHEN_MAS_UCE_MODEL_PY` | UCE `model.py` 路径。 |
| `OPENAI_API_KEY` | OpenAI-compatible gateway API key。 |
| `OPENAI_BASE_URL` | OpenAI-compatible gateway base URL，例如 `https://your-gateway/v1`。 |
| `OPENAI_MODEL` | Gateway 默认 chat model。 |
| `SCMAS_LLM_MODEL` | CanChen_MAS 优先使用的 chat model；为空时回退到 `OPENAI_MODEL`。 |
| `OPENAI_TIMEOUT` | LLM 请求超时秒数。 |
| `OPENAI_MAX_RETRIES` | LLM client 重试次数。 |
| `OPENAI_TRUST_ENV` | 是否让 httpx 读取系统 proxy 环境变量。 |
| `LANGSMITH_TRACING` | 是否启用 vendor tracing；默认建议 `false`。 |

Stage 2/3 的 LLM 组件由 `--llm-mode required|optional|off` 控制。Stage 4 的 policy planner 用 `--llm-policy-mode required|optional|off` 控制，低一致性 cell adjudication 用 `--llm-cell-adjudication-mode required|optional|off` 控制。

`optional` 模式会调用 LLM；如果 gateway/network/JSON 校验失败，则记录失败原因并使用确定性 reviewer 结果。`required` 模式会把这些错误暴露为运行失败，适合正式检查 LLM 配置。

## 切换方式

### 切换 LLM provider/model

`SCMAS_LLM_MODEL` 或 CLI 的 `--llm-model` 决定 Stage 2/3/4 调用哪个 chat model：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-openai-compatible-gateway/v1
export SCMAS_LLM_MODEL=gpt-5.4-mini
```

也可以每次命令显式覆盖：

```bash
canchen-mas select-models ... --llm-mode required --llm-model gpt-5.4-mini
canchen-mas adapt-and-execute ... --llm-mode required --llm-model gpt-5.4-mini
canchen-mas run-consensus ... --llm-policy-mode required --llm-model gpt-5.4-mini
```

这个开关只改变 MAS planner/reviewer 使用的 LLM，不改变候选生物模型。

### 切换 Stage2 候选执行模型

Stage2 能看到的候选由三个输入共同决定：

- `--capability-dir`：候选模型能力卡目录。
- `--prepared-source-root`：可绑定的 source/reference bundle。
- `--artifact-bundle`：可选 Stage-1/source 明细证据和 source manifest；缺失时使用能力卡内置的组合级 Stage-1 分数。

正式运行用默认能力卡：

```bash
canchen-mas select-models \
  --query-profile runs/stage2/my_query/query_profile.json \
  --prepared-source-root data/prepared_sources \
  --artifact-bundle artifacts/stage1_full \
  --capability-dir configs/capability \
  --num-models 3 \
  --llm-mode optional
```

只做小样本 smoke test 时，可以传一个只包含 toy capability YAML 的目录：

```bash
canchen-mas select-models \
  --query-profile runs/smoke/stage2/query_profile.json \
  --prepared-source-root runs/smoke/prepared_sources \
  --artifact-bundle runs/smoke/artifacts/stage1_empty \
  --capability-dir runs/smoke/capability \
  --num-models 2 \
  --llm-mode required
```

如果要隐藏某个模型，可重复使用 `--exclude-model`：

```bash
canchen-mas select-models ... --exclude-model geneformer_raw_knn --exclude-model uce_33l_raw_knn
```

如果要新增模型，新增或复制一张 `configs/capability/<model_id>.yaml`，并在需要 checkpoint 的情况下补齐 `.env` 中对应路径。

## 配置文件

- `configs/paths.yaml`：路径模板和环境变量说明。
- `configs/model_registry.yaml`：传统模型/直接头注册表，支持 `${ENV_VAR}` 和相对路径。
- `configs/capability/*.yaml`：模型能力卡，保留输入输出契约、依赖、权重占位路径和组合级 Stage-1 能力分数；不包含原始 score table、运行目录、报告或本地绝对 `source_path`。
- `configs/data_plan.yaml`：数据准备计划模板，不包含本机绝对路径。

环境创建和 scDesign3 R 包安装见 [environment.md](environment.md)。
