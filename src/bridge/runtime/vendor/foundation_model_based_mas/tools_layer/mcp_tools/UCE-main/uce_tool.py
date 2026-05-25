from __future__ import annotations
# [1] 291016
import argparse
import faulthandler
import logging
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
faulthandler.enable()

# 先按你服务器上的常见路径给默认值，必要时可在命令行覆盖
DEFAULT_MODEL_DIR      = "/data1/uce/model"
DEFAULT_TOKEN_FILE     = "/data1/foundation_model_based_mas/model/all_tokens.torch"
DEFAULT_SPEC_CHROM_CSV = "/data1/foundation_model_based_mas/model/species_chrom.csv"
DEFAULT_OFFSET_PKL     = "/data1/foundation_model_based_mas/model/species_offsets.pkl"
DEFAULT_PROTEIN_DIR    = "/data1/foundation_model_based_mas/model/protein_embeddings"
DEFAULT_BEST_CLF_PATH  = "/data1/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/best_model.pth"   # 最佳模型保存路径（.pth），留空则使用 save_dir/best_model.pth
DEFAULT_LOG_PATH       = "/data1/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/log.txt"   # 结果汇总日志路径（.txt），留空则不额外追加
DEFAULT_H5AD_PATH      = "/data1/SEA-AD/MTG/mtg_spatial_data/SEAAD_MTG_MERFISH.2024-12-11.h5ad"   # 用于读取 gene names 的 h5ad 文件
DEFAULT_NPZ_PATH       = "/data1/SEA-AD/MJM/data/global_data.npz"   # 预处理后的 NPZ 数据路径
DEFAULT_RESULT_JSON    = "/data1/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/result.json"   # 测试集全部结果保存路径（.json），留空则不保存

sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

def get_logger(log_dir: str, name: str, log_filename: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, log_filename))
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    print("Log directory:", log_dir)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader  (参照 dataloader.py)
# ──────────────────────────────────────────────────────────────────────────────

class SpatialDataset(Dataset):
    def __init__(
        self,
        X, y_class, y_subclass, y_supertype,
        batch_donor, spatial_coords, cps, confidence,
    ):
        self.X             = torch.from_numpy(X).float()
        self.y_class       = torch.from_numpy(y_class).long()
        self.y_subclass    = torch.from_numpy(y_subclass).long()
        self.y_supertype   = torch.from_numpy(y_supertype).long()
        self.batch_donor   = torch.from_numpy(batch_donor).long()
        self.spatial_coords= torch.from_numpy(spatial_coords).float()
        self.cps           = torch.from_numpy(cps).float()
        self.confidence    = torch.from_numpy(confidence).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "X":          self.X[idx],
            "spatial":    self.spatial_coords[idx],
            "batch_id":   self.batch_donor[idx],
            "y_class":    self.y_class[idx],
            "y_subclass": self.y_subclass[idx],
            "y_supertype":self.y_supertype[idx],
            "cps":        self.cps[idx],
            "confidence": self.confidence[idx],
        }


def build_dataloaders(
    npz_path: str,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 8,
):
    data = np.load(npz_path, allow_pickle=True)

    X_all     = data["X"]
    y_c_all   = data["y_class"]
    y_sc_all  = data["y_subclass"]
    y_st_all  = data["y_supertype"]
    batch_all = data["batch_donor"]
    spatial   = data["spatial"]
    cps_all   = data["cps"]
    conf_all  = data["y_supertype_confidence"]
    meta_info = data["meta"]

    unique_donors = np.unique(batch_all)
    np.random.seed(seed)
    np.random.shuffle(unique_donors)

    test_donors  = unique_donors[:6]
    val_donors   = unique_donors[6:9]
    train_donors = unique_donors[9:]

    mask_train = np.isin(batch_all, train_donors)
    mask_val   = np.isin(batch_all, val_donors)
    mask_test  = np.isin(batch_all, test_donors)

    def _make(mask):
        return SpatialDataset(
            X_all[mask], y_c_all[mask], y_sc_all[mask], y_st_all[mask],
            batch_all[mask], spatial[mask], cps_all[mask], conf_all[mask],
        )

    train_ds = _make(mask_train)
    val_ds   = _make(mask_val)
    test_ds  = _make(mask_test)

    meta       = meta_info.item()
    output_num = [
        int(meta.get("num_class",     int(y_c_all.max())  + 1)),
        int(meta.get("num_subclass",  int(y_sc_all.max()) + 1)),
        int(meta.get("num_supertype", int(y_st_all.max()) + 1)),
    ]

    print(f"[数据] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"[数据] output_num from meta: {output_num}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, output_num


# ──────────────────────────────────────────────────────────────────────────────
# Model  (参照 uce_annotation.py)
# ──────────────────────────────────────────────────────────────────────────────

CLS_TOKEN_IDX      = 3
PAD_TOKEN_IDX      = 0
CHROM_TOKEN_RIGHT  = 2
CHROM_TOKEN_OFFSET = 143574


class ClsDecoder(nn.Module):
    """Two-layer classification head（与 scGPT ClsDecoder 结构一致）。"""
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers += [nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU()]
        layers.append(nn.Linear(d_model, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UCEForAnnotation(nn.Module):
    """
    预训练 UCE（4-layer）微调用于分层细胞类型标注。

    Args:
        model_path:      4layer_model.torch 的完整路径。
        token_file:      all_tokens.torch 的完整路径。
        spec_chrom_csv:  species_chrom.csv 的完整路径。
        offset_pkl:      species_offsets.pkl 的完整路径。
        gene_names:      数据矩阵列对应的基因符号列表。
        output_num:      [n_class, n_subclass, n_supertype]。
        freeze_backbone: 冻结 transformer 权重，只训练分类头。
    """

    def __init__(
        self,
        model_path: str,
        token_file: str,
        spec_chrom_csv: str,
        offset_pkl: str,
        gene_names: list,
        output_num: list = None,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self.gene_names = gene_names
        self.n_genes    = len(gene_names)
        self.d_model    = 1280

        from model import TransformerModel

        # ── backbone ──────────────────────────────────────────────────────────
        self.backbone = TransformerModel(
            token_dim  = 5120,
            d_model    = 1280,
            nhead      = 20,
            d_hid      = 5120,
            nlayers    = 4,
            output_dim = 1280,
            dropout    = 0.05,
        )
        full_state     = torch.load(model_path, map_location="cpu")
        backbone_state = {k: v for k, v in full_state.items() if not k.startswith("pe_embedding")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"[UCE] Loaded backbone from {model_path} | missing: {len(missing)} | unexpected: {len(unexpected)}")

        # ── pe_embedding ──────────────────────────────────────────────────────
        all_tokens = torch.load(token_file, map_location="cpu")
        self.pe_embedding = nn.Embedding.from_pretrained(all_tokens, freeze=True)
        print(f"[UCE] pe_embedding shape: {all_tokens.shape}")

        # ── 基因元信息预计算 ──────────────────────────────────────────────────
        chrom_df = pd.read_csv(spec_chrom_csv)
        # categorical codes 必须在全物种 DataFrame 上计算，以匹配预训练
        chrom_df["spec_chrom"] = pd.Categorical(
            chrom_df["species"] + "_" + chrom_df["chromosome"].astype(str)
        )
        human_df = chrom_df[chrom_df["species"] == "human"].reset_index(drop=True)

        with open(offset_pkl, "rb") as f:
            offsets = pickle.load(f)
        human_offset = offsets["human"]

        gene_to_info = {
            row["gene_symbol"]: {
                "token_idx":  human_offset + i,
                "chrom_code": int(human_df["spec_chrom"].cat.codes[i]),
                "start":      int(row["start"]),
            }
            for i, row in human_df.iterrows()
        }

        token_idxs, chrom_codes, starts, valid_mask = [], [], [], []
        for g in gene_names:
            info = gene_to_info.get(g)
            if info is None:
                token_idxs.append(PAD_TOKEN_IDX); chrom_codes.append(-1)
                starts.append(0);                 valid_mask.append(False)
            else:
                token_idxs.append(info["token_idx"]); chrom_codes.append(info["chrom_code"])
                starts.append(info["start"]);          valid_mask.append(True)

        self.register_buffer("_token_idxs",  torch.tensor(token_idxs,  dtype=torch.long))
        self.register_buffer("_chrom_codes", torch.tensor(chrom_codes, dtype=torch.long))
        self.register_buffer("_starts",      torch.tensor(starts,      dtype=torch.long))
        self.register_buffer("_valid_mask",  torch.tensor(valid_mask,  dtype=torch.bool))
        print(f"[UCE] Gene coverage: {self._valid_mask.sum().item()}/{self.n_genes}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[UCE] Backbone weights frozen.")

        # ── 分层分类头 ────────────────────────────────────────────────────────
        self.cls_head_class     = ClsDecoder(self.d_model, output_num[0])
        self.cls_head_subclass  = ClsDecoder(self.d_model, output_num[1])
        self.cls_head_supertype = ClsDecoder(self.d_model, output_num[2])

    def _build_sentences(self, X: torch.Tensor):
        """按染色体顺序构建基因句子。返回 (seq_len, B) sentences 和 (B, seq_len) mask。"""
        B      = X.shape[0]
        device = X.device

        X_np        = X.detach().cpu().numpy()
        token_idxs  = self._token_idxs.cpu().numpy()
        chrom_codes = self._chrom_codes.cpu().numpy()
        starts_np   = self._starts.cpu().numpy()
        valid_mask  = self._valid_mask.cpu().numpy()

        all_sentences = []
        max_len = 0

        for b in range(B):
            expr    = X_np[b]
            sel     = valid_mask & (expr > 0)
            sel_idx = np.where(sel)[0]
            sent    = [CLS_TOKEN_IDX]

            if len(sel_idx) > 0:
                uq_chroms = np.unique(chrom_codes[sel_idx])
                np.random.shuffle(uq_chroms)   # 与预训练一致：染色体顺序随机打乱
                for chrom in uq_chroms:
                    g_in_chrom = sel_idx[chrom_codes[sel_idx] == chrom]
                    order      = np.argsort(starts_np[g_in_chrom])
                    g_in_chrom = g_in_chrom[order]
                    sent.append(int(chrom) + CHROM_TOKEN_OFFSET)
                    for gi in g_in_chrom:
                        sent.append(int(token_idxs[gi]))
                    sent.append(CHROM_TOKEN_RIGHT)

            all_sentences.append(sent)
            max_len = max(max_len, len(sent))

        sentences_padded = torch.full((B, max_len), PAD_TOKEN_IDX, dtype=torch.long)
        mask_padded      = torch.zeros(B, max_len, dtype=torch.float)
        for b, sent in enumerate(all_sentences):
            n = len(sent)
            sentences_padded[b, :n] = torch.tensor(sent, dtype=torch.long)
            mask_padded[b, :n]      = 1.0

        sentences_padded = sentences_padded.t().contiguous().to(device)
        mask_padded      = mask_padded.to(device)
        return sentences_padded, mask_padded

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, G) 原始基因表达，非负 float32。
        Returns:
            logits:   [logit_class (B, C), logit_subclass (B, SC), logit_supertype (B, ST)]
            cell_emb: (B, 1280)
        """
        sentences, mask = self._build_sentences(X)
        emb = self.pe_embedding(sentences)
        emb = nn.functional.normalize(emb, dim=2)
        _, cell_emb = self.backbone(emb, mask=mask)

        return [
            self.cls_head_class(cell_emb),
            self.cls_head_subclass(cell_emb),
            self.cls_head_supertype(cell_emb),
        ], cell_emb


# ──────────────────────────────────────────────────────────────────────────────
# Trainer Engine  (参照 scgpt_trainer.py)
# ──────────────────────────────────────────────────────────────────────────────

class TrainerEngine:
    def __init__(self, model, optimizer, scheduler, device, logger, args):
        self.model         = model
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.logger        = logger
        self.args          = args
        self.best_val_loss = float("inf")

    # ── loss ─────────────────────────────────────────────────────────────────

    def compute_loss(self, batch_data):
        X          = batch_data["X"].to(self.device)
        y_c        = batch_data["y_class"].to(self.device)
        y_sc       = batch_data["y_subclass"].to(self.device)
        y_st       = batch_data["y_supertype"].to(self.device)
        confidence = batch_data["confidence"].to(self.device)

        logits, cell_emb         = self.model(X)
        logit_c, logit_sc, logit_st = logits

        loss_c  = F.cross_entropy(logit_c,  y_c)
        loss_sc = F.cross_entropy(logit_sc, y_sc)
        # confidence-weighted supertype loss（与 MJM 一致）
        loss_st = (F.cross_entropy(logit_st, y_st, reduction="none") * confidence).mean()
        total   = loss_c + loss_sc + loss_st

        metrics     = {"total": total.item(), "class": loss_c.item(), "subclass": loss_sc.item(), "supertype": loss_st.item()}
        logits_dict = {"class": logit_c, "subclass": logit_sc, "supertype": logit_st}
        return total, metrics, logits_dict, cell_emb

    # ── epoch helpers ─────────────────────────────────────────────────────────

    def train_epoch(self, dataloader):
        self.model.train()
        agg = {"total": 0.0, "class": 0.0, "subclass": 0.0, "supertype": 0.0}
        for batch_data in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            loss, metrics, _, _ = self.compute_loss(batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            for k in agg:
                agg[k] += metrics[k]
        return {k: v / len(dataloader) for k, v in agg.items()}

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        agg = {"total": 0.0, "class": 0.0, "subclass": 0.0, "supertype": 0.0}
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            _, metrics, _, _ = self.compute_loss(batch_data)
            for k in agg:
                agg[k] += metrics[k]
        return {k: v / len(dataloader) for k, v in agg.items()}

    # ── full training loop ────────────────────────────────────────────────────

    def train(self, train_loader, val_loader):
        os.makedirs(self.args.save_dir, exist_ok=True)
        best_model_path = (
            self.args.best_clf_path
            if self.args.best_clf_path
            else os.path.join(self.args.save_dir, "best_model.pth")
        )
        patience_counter = 0
        best_val_metrics = None

        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info(f"[Resume] Loaded checkpoint from {best_model_path}")

        for epoch in range(1, self.args.max_epochs + 1):
            t0      = time.time()
            train_m = self.train_epoch(train_loader)
            val_m   = self.eval_epoch(val_loader)
            self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.args.max_epochs:03d}] | "
                f"Time: {time.time()-t0:.1f}s | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            self.logger.info(
                f"  [Train] Total: {train_m['total']:.4f} | Class: {train_m['class']:.4f} | "
                f"Subclass: {train_m['subclass']:.4f} | Supertype: {train_m['supertype']:.4f}"
            )
            self.logger.info(
                f"  [Val]   Total: {val_m['total']:.4f} | Class: {val_m['class']:.4f} | "
                f"Subclass: {val_m['subclass']:.4f} | Supertype: {val_m['supertype']:.4f}"
            )
            if best_val_metrics is not None:
                self.logger.info(
                    f"  [Best Val] Total: {self.best_val_loss:.4f} | "
                    f"Class: {best_val_metrics['class']:.4f} | "
                    f"Subclass: {best_val_metrics['subclass']:.4f} | "
                    f"Supertype: {best_val_metrics['supertype']:.4f}"
                )

            if val_m["total"] < self.best_val_loss:
                self.best_val_loss = val_m["total"]
                best_val_metrics   = val_m
                patience_counter   = 0
                Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f"Best model saved to {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info("Early stopping triggered!")
                    break

        return best_val_metrics

    # ── test ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("\nLoading best model for test-set evaluation...")
        best_model_path = (
            self.args.best_clf_path
            if self.args.best_clf_path
            else os.path.join(self.args.save_dir, "best_model.pth")
        )
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        res        = {"cell_emb": [], "spatial": [], "batch": [], "supertype": [], "cps": []}
        agg        = {"total": 0.0, "class": 0.0, "subclass": 0.0, "supertype": 0.0}
        all_y_true = {"class": [], "subclass": [], "supertype": []}
        all_y_pred = {"class": [], "subclass": [], "supertype": []}
        all_y_prob = {"class": [], "subclass": [], "supertype": []}

        for batch_data in test_loader:
            _, metrics, logits_dict, cell_emb = self.compute_loss(batch_data)

            res["cell_emb"].append(cell_emb.cpu())
            res["spatial"].append(batch_data["spatial"].cpu())
            res["batch"].append(batch_data["batch_id"].cpu())
            res["supertype"].append(batch_data["y_supertype"].cpu())
            res["cps"].append(batch_data["cps"].cpu())
            for k in agg:
                agg[k] += metrics[k]

            all_y_true["class"].append(batch_data["y_class"].cpu().numpy())
            all_y_true["subclass"].append(batch_data["y_subclass"].cpu().numpy())
            all_y_true["supertype"].append(batch_data["y_supertype"].cpu().numpy())
            for task in ("class", "subclass", "supertype"):
                all_y_prob[task].append(F.softmax(logits_dict[task], dim=-1).cpu().numpy())
                all_y_pred[task].append(logits_dict[task].argmax(dim=-1).cpu().numpy())

        test_m     = {k: v / len(test_loader) for k, v in agg.items()}
        final_true = {k: np.concatenate(v)         for k, v in all_y_true.items()}
        final_pred = {k: np.concatenate(v)         for k, v in all_y_pred.items()}
        final_prob = {k: np.concatenate(v, axis=0) for k, v in all_y_prob.items()}

        self.logger.info(
            f"  [Test] Total: {test_m['total']:.4f} | Class: {test_m['class']:.4f} | "
            f"Subclass: {test_m['subclass']:.4f} | Supertype: {test_m['supertype']:.4f}"
        )
        self._log_classification_metrics(final_true, final_pred, prefix="Test")
        self._log_auc_roc_metrics(
            final_true, final_prob,
            num_classes_dict={
                "class":     self.args.output_num[0],
                "subclass":  self.args.output_num[1],
                "supertype": self.args.output_num[2],
            },
            prefix="Test",
        )

        res_np   = {k: torch.cat(v, dim=0).numpy() for k, v in res.items()}
        out_path = os.path.join(self.args.save_dir, "test_features.npz")
        os.makedirs(self.args.save_dir, exist_ok=True)
        np.savez_compressed(out_path, **res_np)
        self.logger.info(f"Features saved to: {out_path}")

        return final_true, final_pred, final_prob, test_m

    # ── metric helpers ────────────────────────────────────────────────────────

    def _log_classification_metrics(self, y_true_dict, y_pred_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} Classification Metrics ==========")
        for task in ("class", "subclass", "supertype"):
            y_true = y_true_dict[task]
            y_pred = y_pred_dict[task]

            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
            per_p, per_r, _, support      = precision_recall_fscore_support(y_true, y_pred, average=None,    zero_division=0)
            acc = accuracy_score(y_true, y_pred)

            self.logger.info(f"[{task.upper()}] Macro    - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
            self.logger.info(f"[{task.upper()}] Micro    - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")
            self.logger.info(f"[{task.upper()}] Accuracy : {acc:.4f}")
            per_logs = [
                f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}"
                for i in range(len(per_p)) if support[i] > 0 or per_p[i] > 0
            ]
            self.logger.info(f"[{task.upper()}] Per-class: {' | '.join(per_logs)}")
            self.logger.info("-" * 50)

    def _log_auc_roc_metrics(self, y_true_dict, y_prob_dict, num_classes_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} AUC-ROC Metrics ==========")
        for task in ("class", "subclass", "supertype"):
            y_true = y_true_dict[task]
            y_prob = y_prob_dict[task]
            num_c  = num_classes_dict[task]

            y_onehot  = np.zeros_like(y_prob)
            y_onehot[np.arange(len(y_true)), y_true] = 1
            micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())

            valid_aucs = []
            for c in range(num_c):
                binary = (y_true == c).astype(int)
                if len(np.unique(binary)) == 2:
                    valid_aucs.append(roc_auc_score(binary, y_prob[:, c]))
            macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

            self.logger.info(
                f"[{task.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | "
                f"(Valid classes: {len(valid_aucs)}/{num_c})"
            )
            self.logger.info("-" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# Args & seed
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UCE 4-layer 分层细胞类型标注工具（以 NPZ 为输入，按 donor 划分 train/val/test）。"
    )

    # 数据
    parser.add_argument("--npz_path",  type=str, default=DEFAULT_NPZ_PATH,  help="预处理后的 NPZ 数据路径")
    parser.add_argument("--h5ad_path", type=str, default=DEFAULT_H5AD_PATH, help="用于读取 gene names 的 h5ad 文件")

    # 模型相关路径
    parser.add_argument("--model_dir",           type=str, default=DEFAULT_MODEL_DIR,      help="含 4layer_model.torch 的目录")
    parser.add_argument("--token_file",           type=str, default=DEFAULT_TOKEN_FILE,     help="all_tokens.torch 路径")
    parser.add_argument("--spec_chrom_csv_path",  type=str, default=DEFAULT_SPEC_CHROM_CSV, help="species_chrom.csv 路径")
    parser.add_argument("--offset_pkl_path",      type=str, default=DEFAULT_OFFSET_PKL,     help="species_offsets.pkl 路径")

    # 保存/日志路径
    parser.add_argument("--best_clf_path",    type=str, default=DEFAULT_BEST_CLF_PATH, help="最佳模型保存路径（.pth），留空则使用 save_dir/best_model.pth")
    parser.add_argument("--log_path",         type=str, default=DEFAULT_LOG_PATH,      help="结果汇总日志路径（.txt），留空则不额外追加")
    parser.add_argument("--result_json_path", type=str, default=DEFAULT_RESULT_JSON,   help="测试集全部结果保存路径（.json），留空则不保存")
    parser.add_argument("--dataset",          type=str, default="SEA_AD_MTP_ST",       help="数据集名称，用于日志目录命名")

    # 模型架构
    parser.add_argument("--output_num",      type=int, nargs="+", default=None,  help="各层级分类数 [class subclass supertype]，默认从 NPZ meta 读取")
    parser.add_argument("--dropout",         type=float, default=0.2)
    parser.add_argument("--freeze_backbone", action="store_true",                help="冻结 transformer backbone，只训练分类头")

    # 训练
    parser.add_argument("--device",       type=str,   default="cuda",    help="cuda / cpu，默认自动检测")
    parser.add_argument("--seed",         type=int,   default=3028)
    parser.add_argument("--mode",         type=str,   default="train", choices=["train", "test"])
    parser.add_argument("--bs",           type=int,   default=512,   help="batch size")
    parser.add_argument("--max_epochs",   type=int,   default=1)
    parser.add_argument("--patience",     type=int,   default=30,    help="early stopping patience")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=8)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        args = parse_args()
        os.chdir(PROJECT_ROOT)

        ft_tag        = "freeze" if args.freeze_backbone else "full"
        ckpt_tag      = f"{ft_tag}_s{args.seed}"
        args.ckpt_tag = ckpt_tag

        log_dir       = f"./experiments/UCE/{args.dataset}/"
        args.save_dir = os.path.join(log_dir, "checkpoints", ckpt_tag)
        logger        = get_logger(log_dir, __name__, f"record_{ckpt_tag}.log")
        logger.info(args)

        set_seed(args.seed)

        device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        device     = torch.device(device_str)
        logger.info(f"Device: {device}")

        print("====== UCE ======")

        # 1. Dataloaders
        train_loader, val_loader, test_loader, meta_output_num = build_dataloaders(
            npz_path    = args.npz_path,
            batch_size  = args.bs,
            seed        = args.seed,
            num_workers = args.num_workers,
        )
        if args.output_num is None:
            args.output_num = meta_output_num
            logger.info(f"output_num 从 NPZ meta 读取: {args.output_num}")
        else:
            logger.info(f"output_num 从命令行指定: {args.output_num}")

        # 2. Gene names from original h5ad
        import anndata
        adata      = anndata.read_h5ad(args.h5ad_path, backed="r")
        gene_names = [g for g in adata.var_names if not g.startswith("Blank")]
        if getattr(adata, "file", None) is not None:
            adata.file.close()
        logger.info(f"Gene names from h5ad: {len(gene_names)}")

        # 3. Model
        model_path = os.path.join(args.model_dir, "4layer_model.torch")
        model = UCEForAnnotation(
            model_path      = model_path,
            token_file      = args.token_file,
            spec_chrom_csv  = args.spec_chrom_csv_path,
            offset_pkl      = args.offset_pkl_path,
            gene_names      = gene_names,
            output_num      = args.output_num,
            dropout         = args.dropout,
            freeze_backbone = args.freeze_backbone,
        ).to(device)

        # 4. Optimizer & scheduler (trainable params only)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = args.lr,
            weight_decay = args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=1e-6
        )

        # 5. Trainer
        engine = TrainerEngine(model, optimizer, scheduler, device, logger, args)

        # 6. Run
        if args.mode == "train":
            engine.train(train_loader, val_loader)
            final_true, final_pred, final_prob, test_m = engine.test(test_loader)
        else:
            final_true, final_pred, final_prob, test_m = engine.test(test_loader)

        # ── stdout summary ────────────────────────────────────────────────────
        for task in ("class", "subclass", "supertype"):
            macro_f1 = f1_score(final_true[task], final_pred[task], average="macro", zero_division=0)
            acc      = accuracy_score(final_true[task], final_pred[task])
            print(f"测试集 [{task}] macro-F1: {macro_f1:.4f} | 准确率: {acc:.4f}")

        # ── optional summary log file ─────────────────────────────────────────
        if args.log_path:
            log_file = Path(args.log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                for task in ("class", "subclass", "supertype"):
                    macro_f1 = f1_score(final_true[task], final_pred[task], average="macro", zero_division=0)
                    acc      = accuracy_score(final_true[task], final_pred[task])
                    f.write(
                        f"ckpt_tag={ckpt_tag}\ttask={task}\t"
                        f"test_f1={macro_f1:.4f}\ttest_acc={acc:.4f}\n"
                    )
            logger.info(f"结果已追加至日志: {args.log_path}")

        # ── JSON full results ─────────────────────────────────────────────────
        if args.result_json_path:
            import json
            from sklearn.metrics import precision_recall_fscore_support as prf
 
            result = {"metrics": {}}  # "ckpt_tag": ckpt_tag, "test_loss": test_m, 
            for task in ("class", "subclass", "supertype"):
                yt, yp, yb = final_true[task], final_pred[task], final_prob[task]
                macro_p, macro_r, macro_f1, _ = prf(yt, yp, average="macro",  zero_division=0)
                micro_p, micro_r, micro_f1, _ = prf(yt, yp, average="micro",  zero_division=0)
                acc = accuracy_score(yt, yp)

                y_onehot  = np.zeros_like(yb)
                y_onehot[np.arange(len(yt)), yt] = 1
                micro_auc = float(roc_auc_score(y_onehot.ravel(), yb.ravel()))
                valid_aucs = []
                for c in range(yb.shape[1]):
                    binary = (yt == c).astype(int)
                    if len(np.unique(binary)) == 2:
                        valid_aucs.append(roc_auc_score(binary, yb[:, c]))
                macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

                result["metrics"][task] = {
                    "macro_precision": round(float(macro_p),  4),
                    "macro_recall":    round(float(macro_r),  4),
                    "macro_f1":        round(float(macro_f1), 4),
                    "micro_precision": round(float(micro_p),  4),
                    "micro_recall":    round(float(micro_r),  4),
                    "micro_f1":        round(float(micro_f1), 4),
                    "accuracy":        round(float(acc),      4),
                    "macro_auc_roc":   round(macro_auc,       4),
                    "micro_auc_roc":   round(micro_auc,       4),
                }

            json_file = Path(args.result_json_path)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"全部测试结果已保存至: {json_file}")

    except Exception:
        print("\n[uce_tool.py 顶层捕获到异常]")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
