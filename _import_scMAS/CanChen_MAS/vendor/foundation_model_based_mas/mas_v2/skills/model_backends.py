from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data_utils import normalize_gene_symbol, upper_set


PROJECT_ROOT = Path(
    os.environ.get("CANCHEN_MAS_FOUNDATION_MAS_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
CHECKPOINT_ROOT = Path(
    os.environ.get("CANCHEN_MAS_FOUNDATION_CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints" / "foundation_models")
).expanduser().resolve()
MJM_ROOT = Path(
    os.environ.get("CANCHEN_MAS_SEAAD_MJM_ROOT", PROJECT_ROOT / "external" / "SEA-AD" / "MJM")
).expanduser().resolve()
if str(MJM_ROOT) not in sys.path:
    sys.path.insert(0, str(MJM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GF_CODE_DIR = Path(os.environ.get("CANCHEN_MAS_GENEFORMER_CODE_DIR", CHECKPOINT_ROOT / "geneformer" / "Geneformer_code_only"))
GF_WEIGHTS = Path(os.environ.get("CANCHEN_MAS_GENEFORMER_WEIGHTS", CHECKPOINT_ROOT / "geneformer" / "model.safetensors"))
SCGPT_CHECKPOINT_ROOT = Path(os.environ.get("CANCHEN_MAS_SCGPT_DIR", CHECKPOINT_ROOT / "scgpt"))
NICHEFORMER_DIR = Path(os.environ.get("CANCHEN_MAS_NICHEFORMER_DIR", CHECKPOINT_ROOT / "nicheformer"))


def resolve_scgpt_checkpoint_dir(model_id: str) -> Path:
    normalized = str(model_id).strip().lower()
    if normalized in {"scgpt", "scgpt_generic_brain"}:
        variant = "brain"
    elif normalized in {"scgpt_human", "scgpt_generic"}:
        variant = "human"
    else:
        raise ValueError(f"Unsupported scGPT model id: {model_id!r}")
    checkpoint_dir = SCGPT_CHECKPOINT_ROOT / variant
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"scGPT checkpoint directory not found: {checkpoint_dir}")
    return checkpoint_dir


def geneformer_vocab_upper() -> set[str]:
    with open(GF_CODE_DIR / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb") as handle:
        gene_name_to_ens: dict[str, str] = pickle.load(handle)
    return upper_set(list(gene_name_to_ens.keys()))


def nicheformer_vocab_upper() -> set[str]:
    with open(NICHEFORMER_DIR / "gene_name_id_dict_gc104M.pkl", "rb") as handle:
        gene_name_to_ens: dict[str, str] = pickle.load(handle)
    return upper_set(list(gene_name_to_ens.keys()))


def scgpt_vocab_upper(model_id: str = "scgpt_generic") -> set[str]:
    ckpt_dir = resolve_scgpt_checkpoint_dir(model_id)
    with open(ckpt_dir / "vocab.json", "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    return upper_set(list(vocab.keys()))


@dataclass
class CoverageInfo:
    n_mapped: int
    n_total: int
    coverage_ratio: float


class GeneformerEmbeddingEncoder:
    def __init__(self, *, gene_names: list[str], device: str):
        from src.models.geneformer.geneformer_annotation import GeneformerForAnnotation

        self.device = torch.device(device)
        self.model = GeneformerForAnnotation(
            code_dir=str(GF_CODE_DIR),
            weights_path=str(GF_WEIGHTS),
            gene_names=gene_names,
            output_num=[2, 2, 2],
            dropout=0.0,
            freeze_backbone=True,
        ).to(self.device)
        self.model.eval()
        vocab_upper = geneformer_vocab_upper()
        mapped = sum(1 for gene_name in gene_names if normalize_gene_symbol(gene_name) in vocab_upper)
        self.coverage_info = CoverageInfo(
            n_mapped=int(mapped),
            n_total=int(len(gene_names)),
            coverage_ratio=float(mapped / max(1, len(gene_names))),
        )

    @torch.inference_mode()
    def encode(self, matrix: np.ndarray, batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, matrix.shape[0], batch_size):
            end = min(start + batch_size, matrix.shape[0])
            x_tensor = torch.from_numpy(matrix[start:end]).float().to(self.device)
            _, cell_emb = self.model(x_tensor)
            outputs.append(F.normalize(cell_emb, dim=1).cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, self.model.d_model), dtype=np.float32)


class NicheformerEmbeddingEncoder:
    def __init__(self, *, gene_names: list[str], species: str, device: str):
        from src.models.nicheformer.nicheformer_annotation import NicheformerForAnnotation

        self.device = torch.device(device)
        species_l = species.strip().lower()
        if "mouse" in species_l:
            specie_token = 6
        elif "human" in species_l:
            specie_token = 5
        else:
            raise ValueError(f"Nicheformer encoder only supports human/mouse species tokens, got {species!r}")
        self.model = NicheformerForAnnotation(
            checkpoint_path=str(NICHEFORMER_DIR / "nicheformer.ckpt"),
            vocab_path=str(NICHEFORMER_DIR / "model.h5ad"),
            merfish_mean_path=str(NICHEFORMER_DIR / "merfish_mean_script.npy"),
            gene_name_to_ens_path=str(NICHEFORMER_DIR / "gene_name_id_dict_gc104M.pkl"),
            gene_names=gene_names,
            output_num=[2, 2, 2],
            dropout=0.0,
            freeze_backbone=True,
            specie_token=specie_token,
        ).to(self.device)
        self.model.eval()
        vocab_upper = nicheformer_vocab_upper()
        mapped = sum(1 for gene_name in gene_names if normalize_gene_symbol(gene_name) in vocab_upper)
        self.coverage_info = CoverageInfo(
            n_mapped=int(mapped),
            n_total=int(len(gene_names)),
            coverage_ratio=float(mapped / max(1, len(gene_names))),
        )

    @torch.inference_mode()
    def encode(self, matrix: np.ndarray, batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, matrix.shape[0], batch_size):
            end = min(start + batch_size, matrix.shape[0])
            x_tensor = torch.from_numpy(matrix[start:end]).float().to(self.device)
            _, cell_emb = self.model(x_tensor)
            outputs.append(F.normalize(cell_emb, dim=1).cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, self.model.d_model), dtype=np.float32)


class OfficialStyleScGPTEmbeddingEncoder:
    """
    scGPT embedding encoder aligned with the official reference-mapping style.
    """

    def __init__(
        self,
        *,
        gene_names: list[str],
        device: str,
        checkpoint_dir: str | Path,
        random_seed: int = 3028,
    ) -> None:
        from src.models.scGPT.scGPT import TransformerModel

        self.device = torch.device(device)
        self.random_seed = int(random_seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.random_seed)

        checkpoint_dir = Path(checkpoint_dir).resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"scGPT checkpoint directory not found: {checkpoint_dir}")
        self.checkpoint_dir = checkpoint_dir

        with open(checkpoint_dir / "args.json", "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        with open(checkpoint_dir / "vocab.json", "r", encoding="utf-8") as handle:
            vocab_dict = json.load(handle)

        self.pad_token = str(cfg.get("pad_token", "<pad>"))
        self.pad_token_id = int(vocab_dict[self.pad_token])
        self.cls_token_id = int(vocab_dict["<cls>"])
        self.pad_value = float(cfg.get("pad_value", -2))
        self.n_bins = int(cfg.get("n_bins", 51))
        self.max_length = int(cfg.get("max_seq_len", 1200))
        self.d_model = int(cfg["embsize"])

        upper_vocab = {normalize_gene_symbol(key): int(value) for key, value in vocab_dict.items()}
        gene_token_ids: list[int] = []
        mapped = 0
        for gene_name in gene_names:
            token_id = upper_vocab.get(normalize_gene_symbol(gene_name))
            if token_id is None:
                token_id = self.pad_token_id
            else:
                mapped += 1
            gene_token_ids.append(int(token_id))
        self.gene_token_ids = np.asarray(gene_token_ids, dtype=np.int64)
        self.coverage_info = CoverageInfo(
            n_mapped=int(mapped),
            n_total=int(len(gene_names)),
            coverage_ratio=float(mapped / max(1, len(gene_names))),
        )

        self.model = TransformerModel(
            ntoken=len(vocab_dict),
            d_model=int(cfg["embsize"]),
            nhead=int(cfg["nheads"]),
            d_hid=int(cfg["d_hid"]),
            nlayers=int(cfg["nlayers"]),
            nlayers_cls=int(cfg.get("n_layers_cls", 3)),
            n_cls=1,
            vocab=vocab_dict,
            dropout=float(cfg.get("dropout", 0.0)),
            pad_token=self.pad_token,
            pad_value=int(cfg.get("pad_value", -2)),
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style=str(cfg.get("input_emb_style", "continuous")),
            n_input_bins=int(cfg.get("n_bins", 51)),
            cell_emb_style=str(cfg.get("cell_emb_style") or "cls"),
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=bool(cfg.get("pre_norm", False)),
        ).to(self.device)

        pretrained_state = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
        converted_state = {}
        for key, value in pretrained_state.items():
            new_key = key.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight").replace(
                "self_attn.Wqkv.bias",
                "self_attn.in_proj_bias",
            )
            converted_state[new_key] = value
        self.model.load_state_dict(converted_state, strict=False)
        self.model.eval()

    def _sample(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gene_ids.numel() <= max_length:
            return gene_ids, expr_values
        keep_first = 1
        tail_indices = torch.randperm(
            gene_ids.numel() - keep_first,
            generator=self._generator,
            device=gene_ids.device,
        )[: max_length - keep_first]
        keep_indices = torch.cat(
            [
                torch.arange(keep_first, device=gene_ids.device),
                tail_indices + keep_first,
            ],
            dim=0,
        )
        return gene_ids[keep_indices], expr_values[keep_indices]

    def _pad(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gene_ids.numel() >= max_length:
            return gene_ids, expr_values
        pad_genes = torch.full(
            (max_length - gene_ids.numel(),),
            self.pad_token_id,
            dtype=gene_ids.dtype,
            device=gene_ids.device,
        )
        pad_expr = torch.full(
            (max_length - expr_values.numel(),),
            self.pad_value,
            dtype=expr_values.dtype,
            device=expr_values.device,
        )
        return torch.cat([gene_ids, pad_genes], dim=0), torch.cat([expr_values, pad_expr], dim=0)

    def _tokenize_and_pad_batch(self, batch_matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        from src.models.scGPT.binning import scgpt_binning_torch

        batch_tensor = torch.from_numpy(np.asarray(batch_matrix, dtype=np.float32))
        binned_batch = scgpt_binning_torch(batch_tensor, n_bins=self.n_bins).numpy().astype(np.float32, copy=False)

        examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_observed_length = 1
        for row, row_binned in zip(batch_matrix, binned_batch):
            nonzero_indices = np.flatnonzero(row > 0)
            token_ids = self.gene_token_ids[nonzero_indices]
            keep_mask = token_ids != self.pad_token_id
            token_ids = token_ids[keep_mask]
            expr_values = row_binned[nonzero_indices][keep_mask].astype(np.float32, copy=False)
            genes = np.concatenate(
                [
                    np.asarray([self.cls_token_id], dtype=np.int64),
                    token_ids.astype(np.int64, copy=False),
                ]
            )
            expr = np.concatenate(
                [
                    np.asarray([self.pad_value], dtype=np.float32),
                    expr_values,
                ]
            )
            gene_tensor = torch.from_numpy(genes)
            expr_tensor = torch.from_numpy(expr)
            examples.append((gene_tensor, expr_tensor))
            max_observed_length = max(max_observed_length, int(gene_tensor.numel()))

        target_length = min(self.max_length, max_observed_length)
        padded_genes: list[torch.Tensor] = []
        padded_expr: list[torch.Tensor] = []
        for gene_tensor, expr_tensor in examples:
            gene_tensor, expr_tensor = self._sample(gene_tensor, expr_tensor, target_length)
            gene_tensor, expr_tensor = self._pad(gene_tensor, expr_tensor, target_length)
            padded_genes.append(gene_tensor)
            padded_expr.append(expr_tensor)
        return torch.stack(padded_genes, dim=0), torch.stack(padded_expr, dim=0)

    @torch.inference_mode()
    def encode(self, matrix: np.ndarray, batch_size: int) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[0] == 0:
            return np.zeros((0, self.d_model), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, matrix.shape[0], batch_size):
            end = min(start + batch_size, matrix.shape[0])
            gene_ids, expr_values = self._tokenize_and_pad_batch(matrix[start:end])
            gene_ids = gene_ids.to(self.device)
            expr_values = expr_values.to(self.device)
            src_key_padding_mask = gene_ids.eq(self.pad_token_id)
            output = self.model(gene_ids, expr_values, src_key_padding_mask)
            cell_emb = F.normalize(output["cell_emb"], dim=1)
            outputs.append(cell_emb.cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, self.d_model), dtype=np.float32)


class ScGPTGenericEmbeddingEncoder:
    def __init__(self, *, gene_names: list[str], device: str, random_seed: int):
        self._encoder = OfficialStyleScGPTEmbeddingEncoder(
            gene_names=gene_names,
            device=device,
            checkpoint_dir=resolve_scgpt_checkpoint_dir("scgpt_generic"),
            random_seed=random_seed,
        )
        self.coverage_info = self._encoder.coverage_info
        self.d_model = self._encoder.d_model

    @torch.inference_mode()
    def encode(self, matrix: np.ndarray, batch_size: int) -> np.ndarray:
        return self._encoder.encode(matrix, batch_size)
