# CanChen_MAS

CanChen_MAS 是一个面向单细胞/空间转录组注释的多阶段 MAS 仓库，包含代码、配置、轻量 vendor 运行组件、测试、文档，以及 Stage-1 产生的组合级能力分数。数据、权重和运行资源可通过独立资源目录挂载或复制到项目内。

## 项目结构

```text
CanChen_MAS/
├── configs/                         # 模型注册表、路径模板、能力卡
│   └── capability/                  # 每个模型/方法的输入输出契约和 Stage-1 能力分数
├── data/                            # 数据目录
├── checkpoints/foundation_models/   # Foundation model 权重目录
├── docs/                            # 配置和 foundation model 权重说明
├── examples/                        # 可放最小运行样例和用户自己的命令记录
├── scripts/                         # 辅助运行脚本
├── src/scmas/                       # CanChen_MAS 主代码
├── tests/                           # 不依赖真实数据集的单元测试
└── vendor/foundation_model_based_mas/# 精简后的 foundation-MAS 运行组件
```

这种结构参考 Pantheon 一类项目的清晰顶层分区：配置、文档、源码、测试、脚本、外部兼容层分别放置，运行产物不混入仓库。

## 安装

推荐使用合并后的 conda 环境，里面同时包含 Python MAS/LLM 依赖和 R/scDesign3 依赖：

```bash
cd CanChen_MAS
conda env create -f environment.yml
conda activate canchen_mas
Rscript scripts/setup_scdesign3_r_packages.R
pip install -e .
```

更多环境说明见 [docs/environment.md](docs/environment.md)。Geneformer、scGPT、Nicheformer、UCE、scVI/scANVI、torch-geometric 等重依赖只在运行对应模型时需要安装或补齐权重。

## 配置

复制 `.env.example` 为 `.env`，按需填写外部数据和权重路径：

```bash
cp .env.example .env
```

默认情况下，项目会把数据视为 `data/`，把 foundation model 权重视为 `checkpoints/foundation_models/`。也可以用环境变量指向外部挂载路径，例如：

```bash
export CANCHEN_MAS_GENEFORMER_DIR=/path/to/geneformer
export CANCHEN_MAS_SCGPT_DIR=/path/to/scgpt
export CANCHEN_MAS_UCE_4L_DIR=/path/to/uce_4l
```

更多变量见 [docs/configuration.md](docs/configuration.md)。

## 能力卡与 Stage-1 分数

`configs/capability/*.yaml` 是 Stage2 选择模型和参考源的主要依据。每张能力卡包含模型 family、输入输出契约、权重占位路径、执行参数，以及 `stage1_evaluation.source_dataset_scores`。

这里的 `source_dataset_scores` 是组合级能力分数：同一个模型在不同 prepared source 上会有不同分数。例如 `geneformer_raw_knn + allen_mouse_reference_e63beb9b` 和 `geneformer_raw_knn + spinal_gse190442` 分开记录。Stage2 选择的是 `(source, model)` 路由，因此优先使用这种组合级 Stage-1 分数；如果用户另外恢复了 `artifacts/stage1_full/scores/full_model_variant_scores.csv`，CSV 中的 source/model 明细会优先于能力卡内置分数。

能力卡中保留数值、source id、source 描述和相对路径占位。

LLM 使用 OpenAI-compatible chat API。`.env` 或 shell 中需要至少配置：

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://your-openai-compatible-gateway/v1
OPENAI_MODEL=
SCMAS_LLM_MODEL=
OPENAI_TIMEOUT=60
OPENAI_MAX_RETRIES=2
OPENAI_TRUST_ENV=true
```

`SCMAS_LLM_MODEL` 优先于 `OPENAI_MODEL`。`--llm-mode required` 会在 LLM/API 错误时失败；`optional` 会先尝试 LLM，再回退到确定性 reviewer；`off` 完全关闭 LLM。

### 切换模型

切换“调用哪个 LLM”：

```bash
export SCMAS_LLM_MODEL=gpt-5.4-mini

canchen-mas select-models \
  --query-profile runs/stage2/my_query/query_profile.json \
  --prepared-source-root data/prepared_sources \
  --artifact-bundle artifacts/stage1_full \
  --output-dir runs/stage2/my_query \
  --llm-mode required \
  --llm-model gpt-5.4-mini
```

`--llm-model` 只影响 planner/reviewer 使用的 chat model，不改变 Stage2 能选择的生物模型。

切换“Stage2 能看到哪些候选执行模型”：

```bash
canchen-mas select-models \
  --query-profile runs/stage2/my_query/query_profile.json \
  --prepared-source-root data/prepared_sources \
  --artifact-bundle artifacts/stage1_full \
  --capability-dir configs/capability \
  --output-dir runs/stage2/my_query \
  --num-models 3 \
  --llm-mode optional
```

Stage2 的候选来自 `--capability-dir` 下的能力卡、`--prepared-source-root` 下的 prepared sources，以及可选 `--artifact-bundle` 中的 Stage-1/source 明细表。能力卡内置组合级 Stage-1 分数；如果 `artifacts/stage1_full` 不存在，Stage2 会继续使用能力卡分数。要让 Geneformer、scGPT、Nicheformer、UCE 等进入候选，保留对应 `configs/capability/*.yaml`，配置好 checkpoint 环境变量，并准备好 source bundle。要只跑 toy/smoke 模型，则传入一个只包含 toy capability YAML 的目录。

## Foundation Model 权重

推荐放置在：

```text
checkpoints/foundation_models/
├── geneformer/
├── scgpt/
│   ├── brain/
│   └── human/
├── nicheformer/
├── uce_4l/
└── uce_33l/
```

来源按原论文或官方发布仓库下载：Geneformer、scGPT、Nicheformer、Universal Cell Embedding (UCE)。需要的文件名和环境变量见 [docs/foundation_models.md](docs/foundation_models.md)。

## 资源包

数据资源可放在独立目录，例如 `/data1/CanChen_MAS_resources`。该目录包含：

- `data/prepared_sources/`：Stage2/Stage3 使用的参考源。
- `data/query/`：SEA-AD 和 Kukanja query 数据。
- `artifacts/stage1_full/`：与资源包内 prepared sources 对齐的 Stage1 source manifest。
- `resources.env.example`：常用 query 数据路径变量。

使用独立资源目录时，在命令中显式传入：

```bash
--prepared-source-root /data1/CanChen_MAS_resources/data/prepared_sources
--artifact-bundle /data1/CanChen_MAS_resources/artifacts/stage1_full
```

## 常用流程

下面是常规运行骨架。运行前准备 query 数据、prepared source 和需要使用的 foundation model 权重。

准备查询 profile：

```bash
canchen-mas profile-query \
  --dataset-id my_query \
  --input data/query/my_query.h5ad \
  --output-dir runs/stage2/my_query \
  --max-cells 20000
```

选择模型和参考源：

```bash
canchen-mas select-models \
  --query-profile runs/stage2/my_query/query_profile.json \
  --prepared-source-root data/prepared_sources \
  --artifact-bundle artifacts/stage1_full \
  --output-dir runs/stage2/my_query \
  --num-models 3 \
  --llm-mode optional
```

执行 Stage 3 适配与白名单动作：

```bash
canchen-mas adapt-and-execute \
  --plan runs/stage2/my_query/selected_execution_plan.yaml \
  --output-dir runs/stage3/my_query \
  --mode subset \
  --llm-mode optional
```

执行 Stage 4 共识融合：

```bash
canchen-mas run-consensus \
  --stage3-summary runs/stage3/my_query/execution_summary.json \
  --output-dir runs/stage4/my_query \
  --mode subset \
  --llm-policy-mode optional
```

真实数据准备命令包括 `build-label-maps`、`build-reference`、`prepare-sources`、`write-scdesign3-configs`、`evaluate`。这些命令需要用户自行提供 SEA-AD/Kukanja/参考数据路径。

## 论文式全量运行

论文表格对应的是全 query 条件、reference cap 1000、k=15、seed 3028、Stage2 LLM required、Stage3 full mode。全量运行需要 query、prepared sources 和 foundation model 权重，并将路径写入 `.env` 或在命令行中显式传入。

```bash
canchen-mas select-models \
  --query-profile runs/stage2/my_query/query_profile.json \
  --prepared-source-root data/prepared_sources \
  --artifact-bundle artifacts/stage1_full \
  --capability-dir configs/capability \
  --output-dir runs/stage2/my_query \
  --num-models 3 \
  --llm-mode required

canchen-mas adapt-and-execute \
  --plan runs/stage2/my_query/selected_execution_plan.yaml \
  --output-dir runs/stage3/my_query \
  --mode full \
  --llm-mode required

canchen-mas run-consensus \
  --stage3-summary runs/stage3/my_query/execution_summary.json \
  --output-dir runs/stage4/my_query \
  --mode full \
  --llm-policy-mode off \
  --llm-cell-adjudication-mode off
```

如果恢复了论文归档 `artifacts/stage1_full`，Stage2 会优先读取其中的明细 CSV；否则使用能力卡内置的组合级 Stage-1 分数。Stage4 的 route weight 只读取能力卡/Stage-1 预执行分数，拿不到时使用中性权重 `1.0`，不会用当前 query 的 post-execution `macro_f1` 做 fallback。

## 模块说明

- `scmas.paths`：统一管理仓库路径、数据路径、权重路径和环境变量覆盖。
- `scmas.data`：构建标签映射、参考集和标准化 source bundle。
- `scmas.eval`：模型注册表、Stage-1 评估、raw label transfer、UCE-IMA 相关执行。
- `scmas.stage2`：查询侧 gene-only profile、source/model 排序和选择计划。
- `scmas.stage3`：模型契约检查、AdapterSpec 生成、白名单执行。
- `scmas.stage4`：多模型预测归一化、投票/置信度/参考增强共识。
- `vendor/foundation_model_based_mas`：从原 foundation-model MAS 项目精简出的兼容运行组件。

## 验证

```bash
PYTHONPATH=src pytest -q tests
PYTHONPATH=src python -m compileall -q src
python -m scmas preflight-scdesign3 --rscript-path Rscript
```

当前测试不依赖真实数据集或已下载模型权重。
