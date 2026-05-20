# Foundation Models

提交版不包含任何 foundation model 权重。请从原论文或官方发布仓库下载，并通过环境变量指向本地目录。

## Geneformer

来源：Geneformer 论文和官方发布仓库。仓库中只保留期望路径；权重需要用户自行下载。

默认路径：

```text
checkpoints/foundation_models/geneformer/
├── model.safetensors
└── Geneformer_code_only/geneformer/gene_name_id_dict_gc104M.pkl
```

环境变量：

```bash
export CANCHEN_MAS_GENEFORMER_DIR=/path/to/geneformer
```

## scGPT

来源：scGPT 论文和官方发布仓库。仓库中只保留期望路径；权重需要用户自行下载。

默认路径：

```text
checkpoints/foundation_models/scgpt/
├── brain/
│   ├── best_model.pt
│   ├── vocab.json
│   └── args.json
└── human/
    ├── best_model.pt
    ├── vocab.json
    └── args.json
```

环境变量：

```bash
export CANCHEN_MAS_SCGPT_DIR=/path/to/scgpt
```

## Nicheformer

来源：Nicheformer 论文和官方发布仓库。仓库中只保留期望路径；权重需要用户自行下载。

默认路径：

```text
checkpoints/foundation_models/nicheformer/
├── nicheformer.ckpt
├── model.h5ad
├── merfish_mean_script.npy
└── gene_name_id_dict_gc104M.pkl
```

环境变量：

```bash
export CANCHEN_MAS_NICHEFORMER_DIR=/path/to/nicheformer
```

## UCE

来源：Universal Cell Embedding (UCE) 论文和官方发布仓库。仓库中只保留期望路径；权重需要用户自行下载。

默认路径：

```text
checkpoints/foundation_models/uce_4l/
├── 4layer_model.torch
├── all_tokens.torch
├── species_chrom.csv
└── species_offsets.pkl

checkpoints/foundation_models/uce_33l/
├── 33l_8ep_1024t_1280.torch
├── all_tokens.torch
├── species_chrom.csv
└── species_offsets.pkl
```

环境变量：

```bash
export CANCHEN_MAS_UCE_4L_DIR=/path/to/uce_4l
export CANCHEN_MAS_UCE_33L_DIR=/path/to/uce_33l
export CANCHEN_MAS_UCE_MODEL_PY=/path/to/UCE-main/model.py
```

本仓库在 `vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/model.py` 保留了 UCE 模型类兼容文件；如果使用外部 UCE 代码，也可以把 `CANCHEN_MAS_UCE_MODEL_PY` 指向外部文件。

## 提交规则

不要把下载后的权重提交到仓库。默认 `.gitignore` 会忽略 `checkpoints/foundation_models/` 下除 `.gitkeep` 以外的内容。正式运行时用环境变量指向本地权重目录即可。
