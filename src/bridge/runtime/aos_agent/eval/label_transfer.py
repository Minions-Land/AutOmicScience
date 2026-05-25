from __future__ import annotations

import json
import pickle
import sys
import traceback
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from aos_agent import paths
from aos_agent.embedding_cache import embedding_cache_key, load_embedding_cache, save_embedding_cache
from aos_agent.data.catalog import build_dataset_catalog
from aos_agent.io import ensure_dir, normalize_gene_name, read_json, read_standard_bundle, stratified_indices, write_json
from aos_agent.scdesign3.discover import discover_existing_seaad_variants, discover_new_synthetic_variants


FINE_LABEL_CANDIDATES = [
    "native_label",
    "cell_type",
    "Supertype",
    "supertype",
    "cluster",
    "subtype_annotation",
    "cell.type",
]
COARSE_LABEL_CANDIDATES = [
    "coarse_label",
    "cell_type_annot",
    "Subclass",
    "subclass",
    "class",
    "supercluster",
]


@dataclass
class MatrixBundle:
    X: sparse.csr_matrix
    obs: pd.DataFrame
    var: pd.DataFrame
    genes: list[str]


def _first_present(columns: pd.Index | list[str], candidates: list[str]) -> str | None:
    present = set(columns)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def _feature_symbols_from_var(var: pd.DataFrame) -> list[str]:
    for column in ("feature_id", "feature_name", "gene_symbol", "gene_name"):
        if column in var.columns:
            return [normalize_gene_name(x) for x in var[column].astype(str)]
    return [normalize_gene_name(x) for x in var.index.astype(str)]


def _feature_symbols_from_adata(adata: ad.AnnData) -> pd.Index:
    for column in ("feature_name", "gene_symbol", "gene_name"):
        if column in adata.var.columns:
            return pd.Index([normalize_gene_name(x) for x in adata.var[column].astype(str)])
    return pd.Index([normalize_gene_name(x) for x in adata.var_names.astype(str)])


def _read_scdesign3_variant(
    variant_path: str | Path,
    *,
    max_cells: int,
    seed: int,
) -> MatrixBundle:
    variant_path = Path(variant_path)
    obs = pd.read_csv(variant_path / "sim_obs.csv")
    var = pd.read_csv(variant_path / "sim_var.csv")
    X = spio.mmread(str(variant_path / "sim_counts.mtx")).tocsr()
    if X.shape[0] == len(var) and X.shape[1] == len(obs):
        X = X.transpose().tocsr()
    elif X.shape[0] != len(obs) or X.shape[1] != len(var):
        raise ValueError(
            f"Cannot infer sim_counts orientation for {variant_path}: "
            f"matrix={X.shape}, obs={len(obs)}, var={len(var)}"
        )

    label_col = _first_present(obs.columns, FINE_LABEL_CANDIDATES) or obs.columns[0]
    idx = stratified_indices(obs[label_col], max_cells=max_cells, min_per_group=5, seed=seed)
    obs = obs.iloc[idx].reset_index(drop=True)
    X = X[idx, :].tocsr()
    if "native_label" not in obs.columns:
        fine_col = _first_present(obs.columns, FINE_LABEL_CANDIDATES)
        if fine_col is not None:
            obs["native_label"] = obs[fine_col].astype(str)
    if "coarse_label" not in obs.columns:
        coarse_col = _first_present(obs.columns, COARSE_LABEL_CANDIDATES)
        if coarse_col is not None:
            obs["coarse_label"] = obs[coarse_col].astype(str)

    genes = _feature_symbols_from_var(var)
    return MatrixBundle(X=X, obs=obs, var=var, genes=genes)


def _mode_map(df: pd.DataFrame, key: str, value: str) -> dict[str, str]:
    if key not in df.columns or value not in df.columns:
        return {}
    out: dict[str, str] = {}
    for k, group in df[[key, value]].dropna().astype(str).groupby(key):
        if group.empty:
            continue
        out[str(k)] = str(group[value].mode().iloc[0])
    return out


def _source_dir_for_variant(variant_path: str | Path) -> Path:
    return Path(variant_path).resolve().parent


def _source_metadata(source_dir: Path) -> dict[str, Any]:
    config_path = source_dir / "resolved_config.json"
    config = read_json(config_path) if config_path.exists() else {}
    prepared_obs_path = source_dir / "_prepared_input" / "obs.csv"
    if not prepared_obs_path.exists():
        raise FileNotFoundError(f"Missing prepared input obs: {prepared_obs_path}")
    prepared_obs = pd.read_csv(prepared_obs_path)

    source_path = ""
    if "source_path" in prepared_obs.columns and prepared_obs["source_path"].notna().any():
        source_path = str(prepared_obs["source_path"].dropna().astype(str).mode().iloc[0])
    species = ""
    if "species" in prepared_obs.columns and prepared_obs["species"].notna().any():
        species = str(prepared_obs["species"].dropna().astype(str).mode().iloc[0])
    source_dataset = ""
    if "source_dataset" in prepared_obs.columns and prepared_obs["source_dataset"].notna().any():
        source_dataset = str(prepared_obs["source_dataset"].dropna().astype(str).mode().iloc[0])
    if not species and {"Class", "Subclass", "Supertype"}.issubset(set(prepared_obs.columns)):
        species = "human"
    if not source_dataset and {"Class", "Subclass", "Supertype"}.issubset(set(prepared_obs.columns)):
        source_dataset = "seaad_anchor_140gene"
    fine_col = _first_present(prepared_obs.columns, FINE_LABEL_CANDIDATES)
    coarse_col = _first_present(prepared_obs.columns, COARSE_LABEL_CANDIDATES)

    return {
        "config": config,
        "prepared_obs": prepared_obs,
        "source_path": source_path,
        "species": species,
        "source_dataset": source_dataset,
        "excluded_cell_ids": set(prepared_obs["cell_id"].astype(str)) if "cell_id" in prepared_obs.columns else set(),
        "native_to_coarse": _mode_map(prepared_obs, fine_col, coarse_col) if fine_col and coarse_col else {},
    }


def _load_reference_from_h5ad(
    source_path: str | Path,
    *,
    genes: list[str],
    excluded_cell_ids: set[str],
    max_cells: int,
    seed: int,
) -> tuple[MatrixBundle, dict[str, str]]:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Reference source h5ad not found: {source_path}")

    backed = ad.read_h5ad(source_path, backed="r")
    try:
        obs = backed.obs.copy()
        fine_col = _first_present(obs.columns, FINE_LABEL_CANDIDATES)
        coarse_col = _first_present(obs.columns, COARSE_LABEL_CANDIDATES)
        if fine_col is None:
            raise ValueError(f"No usable fine label column found in {source_path}")
        obs_names = pd.Index(obs.index.astype(str))
        eligible = np.asarray(~obs_names.isin(excluded_cell_ids), dtype=bool)
        labels = obs[fine_col].astype(str).fillna("unknown")
        eligible &= ~labels.isin(["", "nan", "None", "unknown"]).to_numpy()
        eligible_idx = np.flatnonzero(eligible)
        if len(eligible_idx) == 0:
            raise ValueError("No disjoint reference cells remain after excluding scDesign3 source cells.")

        sampled_rel = stratified_indices(
            labels.iloc[eligible_idx],
            max_cells=max_cells,
            min_per_group=5,
            seed=seed,
        )
        obs_idx = np.sort(eligible_idx[sampled_rel])

        symbols = _feature_symbols_from_adata(backed)
        symbol_to_idx: dict[str, int] = {}
        for idx, symbol in enumerate(symbols):
            symbol_to_idx.setdefault(str(symbol), idx)
        wanted = [normalize_gene_name(g) for g in genes]
        found_pairs = [(gene, symbol_to_idx[gene]) for gene in wanted if gene in symbol_to_idx]
        if not found_pairs:
            raise ValueError(f"No shared genes found between variant panel and {source_path}")
        out_genes = [gene for gene, _ in found_pairs]
        var_idx = np.array([idx for _, idx in found_pairs], dtype=np.int64)
        var_read_idx: slice | np.ndarray
        if len(var_idx) == backed.n_vars and np.array_equal(var_idx, np.arange(backed.n_vars)):
            var_read_idx = slice(None)
        else:
            var_read_idx = var_idx

        if not isinstance(var_read_idx, slice) and len(obs_idx) != backed.n_obs:
            sub = backed[:, var_read_idx].to_memory()[obs_idx, :].copy()
        else:
            sub = backed[obs_idx, var_read_idx].to_memory().copy()
        sub.var_names = pd.Index(out_genes)
        sub.var["feature_id"] = out_genes
        ref_obs = sub.obs.copy()
        ref_obs["native_label"] = ref_obs[fine_col].astype(str)
        if coarse_col is not None:
            ref_obs["coarse_label"] = ref_obs[coarse_col].astype(str)
        else:
            ref_obs["coarse_label"] = ref_obs["native_label"]
        if "cell_id" not in ref_obs.columns:
            ref_obs.insert(0, "cell_id", ref_obs.index.astype(str))

        return (
            MatrixBundle(
                X=sparse.csr_matrix(sub.X),
                obs=ref_obs.reset_index(drop=True),
                var=sub.var.copy(),
                genes=out_genes,
            ),
            {"fine_label_column": fine_col, "coarse_label_column": coarse_col or fine_col},
        )
    finally:
        if getattr(backed, "file", None) is not None:
            backed.file.close()


def _load_reference_from_standard_bundle(
    bundle_dir: str | Path,
    *,
    genes: list[str],
    max_cells: int,
    seed: int,
) -> tuple[MatrixBundle, dict[str, str]]:
    X, obs, var = read_standard_bundle(bundle_dir)
    fine_col = _first_present(obs.columns, FINE_LABEL_CANDIDATES)
    coarse_col = _first_present(obs.columns, COARSE_LABEL_CANDIDATES)
    if fine_col is None:
        raise ValueError(f"No usable fine label column found in {bundle_dir}")

    labels = obs[fine_col].astype(str).fillna("unknown")
    usable = ~labels.isin(["", "nan", "None", "unknown"]).to_numpy()
    if not usable.any():
        raise ValueError(f"No usable reference labels found in {bundle_dir}")
    X = X[usable, :].tocsr()
    obs = obs.iloc[np.flatnonzero(usable)].copy()
    labels = labels.iloc[np.flatnonzero(usable)]

    idx = stratified_indices(labels, max_cells=max_cells, min_per_group=5, seed=seed)
    X = X[idx, :].tocsr()
    obs = obs.iloc[idx].reset_index(drop=True)

    bundle_genes = _feature_symbols_from_var(var)
    gene_to_idx: dict[str, int] = {}
    for idx, gene in enumerate(bundle_genes):
        gene_to_idx.setdefault(gene, idx)
    wanted = [normalize_gene_name(g) for g in genes]
    found_pairs = [(gene, gene_to_idx[gene]) for gene in wanted if gene in gene_to_idx]
    if not found_pairs:
        raise ValueError(f"No shared genes found between variant panel and {bundle_dir}")

    out_genes = [gene for gene, _ in found_pairs]
    var_idx = [idx for _, idx in found_pairs]
    ref_obs = obs.copy()
    ref_obs["native_label"] = ref_obs[fine_col].astype(str)
    ref_obs["coarse_label"] = ref_obs[coarse_col].astype(str) if coarse_col else ref_obs["native_label"]
    if "cell_id" not in ref_obs.columns:
        ref_obs.insert(0, "cell_id", ref_obs.index.astype(str))
    return (
        MatrixBundle(
            X=X[:, var_idx].tocsr(),
            obs=ref_obs,
            var=var.iloc[var_idx].copy() if len(var) == len(bundle_genes) else var.copy(),
            genes=out_genes,
        ),
        {"fine_label_column": fine_col, "coarse_label_column": coarse_col or fine_col},
    )


def _subset_genes(bundle: MatrixBundle, genes: list[str]) -> MatrixBundle:
    gene_to_idx = {gene: idx for idx, gene in enumerate(bundle.genes)}
    idx = [gene_to_idx[gene] for gene in genes]
    return MatrixBundle(
        X=bundle.X[:, idx].tocsr(),
        obs=bundle.obs.copy(),
        var=bundle.var.iloc[idx].copy() if len(bundle.var) == len(bundle.genes) else bundle.var.copy(),
        genes=list(genes),
    )


def _align_reference_and_query(ref: MatrixBundle, query: MatrixBundle, *, min_shared_genes: int) -> tuple[MatrixBundle, MatrixBundle, list[str]]:
    ref_genes = set(ref.genes)
    shared = [gene for gene in query.genes if gene in ref_genes]
    if len(shared) < min_shared_genes:
        raise ValueError(f"Only {len(shared)} shared genes; min_shared_genes={min_shared_genes}")
    return _subset_genes(ref, shared), _subset_genes(query, shared), shared


def _clip_nonnegative(X: sparse.csr_matrix) -> sparse.csr_matrix:
    X = X.copy().tocsr()
    if X.nnz:
        X.data = np.where(np.isfinite(X.data), X.data, 0)
        X.data[X.data < 0] = 0
        X.eliminate_zeros()
    return X


def _library_log1p_dense(X: sparse.csr_matrix) -> np.ndarray:
    X = _clip_nonnegative(X)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    scale = np.divide(1e4, row_sums, out=np.zeros_like(row_sums, dtype=np.float64), where=row_sums > 0)
    X = sparse.diags(scale).dot(X).tocsr()
    if X.nnz:
        X.data = np.log1p(X.data)
    return X.toarray().astype(np.float32, copy=False)


class UCE4LRawEmbedding:
    CLS_TOKEN_IDX = 3
    PAD_TOKEN_IDX = 0
    CHROM_TOKEN_RIGHT = 2
    CHROM_TOKEN_OFFSET = 143574

    def __init__(
        self,
        *,
        model_dir: str | Path,
        gene_names: list[str],
        species: str,
        freeze_backbone: bool = True,
    ):
        import importlib.util
        import torch
        import torch.nn as nn

        model_path = paths.UCE_MODEL_PY
        spec = importlib.util.spec_from_file_location("scmas_uce4_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import UCE model from {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TransformerModel = module.TransformerModel

        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.d_model = 1280
        self.backbone = TransformerModel(
            token_dim=5120,
            d_model=1280,
            nhead=20,
            d_hid=5120,
            nlayers=4,
            output_dim=1280,
            dropout=0.05,
        )

        model_dir = Path(model_dir)
        full_state = torch.load(model_dir / "4layer_model.torch", map_location="cpu")
        backbone_state = {k: v for k, v in full_state.items() if not k.startswith("pe_embedding")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"[UCE-4L raw] Loaded backbone | missing: {len(missing)} | unexpected: {len(unexpected)}")

        all_tokens = torch.load(model_dir / "all_tokens.torch", map_location="cpu")
        self.pe_embedding = nn.Embedding.from_pretrained(all_tokens, freeze=True)

        species_key = {
            "human": "human",
            "homo sapiens": "human",
            "mouse": "mouse",
            "mus musculus": "mouse",
        }.get(species.strip().lower())
        if species_key is None:
            raise ValueError(f"Unsupported UCE species for raw transfer: {species!r}")

        chrom_df = pd.read_csv(model_dir / "species_chrom.csv")
        chrom_df["spec_chrom"] = pd.Categorical(chrom_df["species"] + "_" + chrom_df["chromosome"].astype(str))
        species_df = chrom_df[chrom_df["species"] == species_key].reset_index(drop=True)
        if species_df.empty:
            raise ValueError(f"UCE species_chrom.csv has no rows for species={species_key!r}")
        with (model_dir / "species_offsets.pkl").open("rb") as handle:
            offsets = pickle.load(handle)
        species_offset = int(offsets[species_key])

        gene_to_info = {
            normalize_gene_name(row["gene_symbol"]): {
                "token_idx": species_offset + idx,
                "chrom_code": int(species_df["spec_chrom"].cat.codes[idx]),
                "start": int(row["start"]),
            }
            for idx, row in species_df.iterrows()
        }

        token_idxs: list[int] = []
        chrom_codes: list[int] = []
        starts: list[int] = []
        valid_mask: list[bool] = []
        for gene_name in gene_names:
            info = gene_to_info.get(normalize_gene_name(gene_name))
            if info is None:
                token_idxs.append(self.PAD_TOKEN_IDX)
                chrom_codes.append(-1)
                starts.append(0)
                valid_mask.append(False)
            else:
                token_idxs.append(int(info["token_idx"]))
                chrom_codes.append(int(info["chrom_code"]))
                starts.append(int(info["start"]))
                valid_mask.append(True)

        self._token_idxs = torch.tensor(token_idxs, dtype=torch.long)
        self._chrom_codes = torch.tensor(chrom_codes, dtype=torch.long)
        self._starts = torch.tensor(starts, dtype=torch.long)
        self._valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        print(f"[UCE-4L raw] Gene coverage ({species_key}): {int(self._valid_mask.sum().item())}/{self.n_genes}")

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.device = torch.device("cpu")

    def to(self, device: Any) -> "UCE4LRawEmbedding":
        import torch

        self.device = torch.device(device)
        self.backbone.to(self.device)
        self.pe_embedding.to(self.device)
        self._token_idxs = self._token_idxs.to(self.device)
        self._chrom_codes = self._chrom_codes.to(self.device)
        self._starts = self._starts.to(self.device)
        self._valid_mask = self._valid_mask.to(self.device)
        return self

    def eval(self) -> "UCE4LRawEmbedding":
        self.backbone.eval()
        self.pe_embedding.eval()
        return self

    def __call__(self, x_tensor: Any) -> tuple[list[Any], Any]:
        return self.forward(x_tensor)

    def _build_sentences(self, x_tensor: Any) -> tuple[Any, Any]:
        import torch

        batch_size = x_tensor.shape[0]
        x_np = x_tensor.detach().cpu().numpy()
        token_idxs = self._token_idxs.detach().cpu().numpy()
        chrom_codes = self._chrom_codes.detach().cpu().numpy()
        starts_np = self._starts.detach().cpu().numpy()
        valid_mask = self._valid_mask.detach().cpu().numpy()

        all_sentences: list[list[int]] = []
        max_len = 0
        for batch_idx in range(batch_size):
            expr = x_np[batch_idx]
            selected_idx = np.where(valid_mask & (expr > 0))[0]
            sentence = [self.CLS_TOKEN_IDX]
            if len(selected_idx) > 0:
                unique_chroms = np.unique(chrom_codes[selected_idx])
                for chrom in sorted(unique_chroms):
                    chrom_genes = selected_idx[chrom_codes[selected_idx] == chrom]
                    chrom_genes = chrom_genes[np.argsort(starts_np[chrom_genes])]
                    sentence.append(int(chrom) + self.CHROM_TOKEN_OFFSET)
                    sentence.extend(int(token_idxs[gene_idx]) for gene_idx in chrom_genes)
                    sentence.append(self.CHROM_TOKEN_RIGHT)
            all_sentences.append(sentence)
            max_len = max(max_len, len(sentence))

        sentences_padded = torch.full((batch_size, max_len), self.PAD_TOKEN_IDX, dtype=torch.long)
        mask_padded = torch.zeros(batch_size, max_len, dtype=torch.float32)
        for batch_idx, sentence in enumerate(all_sentences):
            sent_len = len(sentence)
            sentences_padded[batch_idx, :sent_len] = torch.tensor(sentence, dtype=torch.long)
            mask_padded[batch_idx, :sent_len] = 1.0
        return sentences_padded.t().contiguous().to(self.device), mask_padded.to(self.device)

    def forward(self, x_tensor: Any) -> tuple[list[Any], Any]:
        import torch
        import torch.nn as nn

        sentences, mask = self._build_sentences(x_tensor)
        embedding = self.pe_embedding(sentences)
        embedding = nn.functional.normalize(embedding, dim=2)
        _, cell_emb = self.backbone(embedding, mask=mask)
        empty_logits = [torch.empty((x_tensor.shape[0], 0), device=x_tensor.device)] * 3
        return empty_logits, cell_emb


class UCE33LRawEmbedding(UCE4LRawEmbedding):
    def __init__(
        self,
        *,
        model_dir: str | Path,
        gene_names: list[str],
        species: str,
        freeze_backbone: bool = True,
    ):
        import importlib.util
        import torch
        import torch.nn as nn

        model_path = paths.UCE_MODEL_PY
        spec = importlib.util.spec_from_file_location("scmas_uce33_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import UCE model from {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TransformerModel = module.TransformerModel

        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.d_model = 1280
        self.backbone = TransformerModel(
            token_dim=5120,
            d_model=1280,
            nhead=20,
            d_hid=5120,
            nlayers=33,
            output_dim=1280,
            dropout=0.05,
        )

        model_dir = Path(model_dir)
        checkpoint_file = model_dir / "33l_8ep_1024t_1280.torch"
        import tarfile

        if tarfile.is_tarfile(checkpoint_file):
            with tarfile.open(checkpoint_file) as archive:
                member_names = archive.getnames()[:20]
            if any(name.startswith("protein_embeddings/") for name in member_names):
                raise FileNotFoundError(
                    f"UCE-33L model state_dict not found: {checkpoint_file} is a protein_embeddings tar archive."
                )
        full_state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        backbone_state = {k: v for k, v in full_state.items() if not k.startswith("pe_embedding")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        del full_state
        print(f"[UCE-33L raw] Loaded backbone | missing: {len(missing)} | unexpected: {len(unexpected)}")

        all_tokens = torch.load(model_dir / "all_tokens.torch", map_location="cpu")
        self.pe_embedding = nn.Embedding.from_pretrained(all_tokens, freeze=True)

        species_key = {
            "human": "human",
            "homo sapiens": "human",
            "mouse": "mouse",
            "mus musculus": "mouse",
        }.get(species.strip().lower())
        if species_key is None:
            raise ValueError(f"Unsupported UCE species for raw transfer: {species!r}")

        chrom_df = pd.read_csv(model_dir / "species_chrom.csv")
        chrom_df["spec_chrom"] = pd.Categorical(chrom_df["species"] + "_" + chrom_df["chromosome"].astype(str))
        species_df = chrom_df[chrom_df["species"] == species_key].reset_index(drop=True)
        if species_df.empty:
            raise ValueError(f"UCE species_chrom.csv has no rows for species={species_key!r}")
        with (model_dir / "species_offsets.pkl").open("rb") as handle:
            offsets = pickle.load(handle)
        species_offset = int(offsets[species_key])

        gene_to_info = {
            normalize_gene_name(row["gene_symbol"]): {
                "token_idx": species_offset + idx,
                "chrom_code": int(species_df["spec_chrom"].cat.codes[idx]),
                "start": int(row["start"]),
            }
            for idx, row in species_df.iterrows()
        }

        token_idxs: list[int] = []
        chrom_codes: list[int] = []
        starts: list[int] = []
        valid_mask: list[bool] = []
        for gene_name in gene_names:
            info = gene_to_info.get(normalize_gene_name(gene_name))
            if info is None:
                token_idxs.append(self.PAD_TOKEN_IDX)
                chrom_codes.append(-1)
                starts.append(0)
                valid_mask.append(False)
            else:
                token_idxs.append(int(info["token_idx"]))
                chrom_codes.append(int(info["chrom_code"]))
                starts.append(int(info["start"]))
                valid_mask.append(True)

        self._token_idxs = torch.tensor(token_idxs, dtype=torch.long)
        self._chrom_codes = torch.tensor(chrom_codes, dtype=torch.long)
        self._starts = torch.tensor(starts, dtype=torch.long)
        self._valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        print(f"[UCE-33L raw] Gene coverage ({species_key}): {int(self._valid_mask.sum().item())}/{self.n_genes}")

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.device = torch.device("cpu")


class EmbeddingAdapter:
    def __init__(self, *, base_method: str, genes: list[str], species: str, device: str, batch_size: int):
        self.base_method = base_method
        self.genes = genes
        self.species = species
        self.device = device
        self.batch_size = batch_size
        self.model: Any = None
        self.valid_gene_mask: np.ndarray | None = None
        self.cache_hits = 0
        self.cache_misses = 0

    def fit(self) -> None:
        if self.base_method == "expression_log1p":
            return
        if self.base_method == "geneformer_raw":
            self._fit_geneformer()
            return
        if self.base_method in {"scgpt_brain_raw", "scgpt_human_raw"}:
            self._fit_scgpt()
            return
        if self.base_method == "nicheformer_raw":
            self._fit_nicheformer()
            return
        if self.base_method in {"uce_4l_raw", "uce_33l_raw"}:
            self._fit_uce()
            return
        raise ValueError(f"Unknown embedding method: {self.base_method}")

    def transform(self, X: sparse.csr_matrix) -> np.ndarray:
        cache_key, cache_metadata = embedding_cache_key(
            base_method=self.base_method,
            genes=self.genes,
            species=self.species,
            matrix=X,
        )
        cached = load_embedding_cache(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached.astype(np.float32, copy=False)
        self.cache_misses += 1
        if self.base_method == "expression_log1p":
            embedding = _library_log1p_dense(X)
        else:
            embedding = self._torch_transform(X)
        save_embedding_cache(cache_key=cache_key, embedding=embedding, metadata=cache_metadata)
        return embedding.astype(np.float32, copy=False)

    def _fit_geneformer(self) -> None:
        sys.path.insert(0, str(paths.SEA_AD_MJM_ROOT))
        from src.models.geneformer.geneformer_annotation import GeneformerForAnnotation

        ckpt_root = paths.GENEFORMER_CHECKPOINT_DIR
        self.model = GeneformerForAnnotation(
            code_dir=str(ckpt_root / "Geneformer_code_only"),
            weights_path=str(ckpt_root / "model.safetensors"),
            gene_names=self.genes,
            output_num=[1, 1, 1],
            freeze_backbone=True,
        )
        self._finalize_torch_model()

    def _fit_scgpt(self) -> None:
        import json as _json
        import torch

        sys.path.insert(0, str(paths.SEA_AD_MJM_ROOT))
        from src.models.scGPT.scGPT_annotation import scGPTForAnnotation

        model_name = "brain" if self.base_method == "scgpt_brain_raw" else "human"
        ckpt_dir = paths.SCGPT_CHECKPOINT_ROOT / model_name
        with (ckpt_dir / "vocab.json").open("r", encoding="utf-8") as handle:
            vocab = _json.load(handle)
        pad_id = int(vocab.get("<pad>", 0))
        gene_ids = []
        valid = []
        for gene in self.genes:
            token = vocab.get(normalize_gene_name(gene))
            if token is None:
                gene_ids.append(pad_id)
                valid.append(False)
            else:
                gene_ids.append(int(token))
                valid.append(True)
        self.valid_gene_mask = np.asarray(valid, dtype=bool)
        if int(self.valid_gene_mask.sum()) == 0:
            raise ValueError(f"{self.base_method} maps 0/{len(self.genes)} genes")
        self.model = scGPTForAnnotation(
            checkpoint_dir=str(ckpt_dir),
            gene_ids=torch.tensor(gene_ids, dtype=torch.long),
            output_num=[1, 1, 1],
            freeze_backbone=True,
        )
        self._finalize_torch_model()

    def _fit_nicheformer(self) -> None:
        sys.path.insert(0, str(paths.SEA_AD_MJM_ROOT))
        from src.models.nicheformer.nicheformer_annotation import NicheformerForAnnotation

        ckpt_root = paths.NICHEFORMER_CHECKPOINT_DIR
        species_token = 6 if self.species.strip().lower() in {"mouse", "mus musculus"} else 5
        self.model = NicheformerForAnnotation(
            checkpoint_path=str(ckpt_root / "nicheformer.ckpt"),
            vocab_path=str(ckpt_root / "model.h5ad"),
            merfish_mean_path=str(ckpt_root / "merfish_mean_script.npy"),
            gene_name_to_ens_path=str(ckpt_root / "gene_name_id_dict_gc104M.pkl"),
            gene_names=self.genes,
            output_num=[1, 1, 1],
            freeze_backbone=True,
            specie_token=species_token,
        )
        self._finalize_torch_model()

    def _fit_uce(self) -> None:
        species = self.species.strip().lower()
        if self.base_method == "uce_4l_raw":
            self.model = UCE4LRawEmbedding(
                model_dir=paths.UCE_4L_MODEL_DIR,
                gene_names=self.genes,
                species=species or "human",
                freeze_backbone=True,
            )
            self._finalize_torch_model()
            return

        self.model = UCE33LRawEmbedding(
            model_dir=paths.UCE_33L_MODEL_DIR,
            gene_names=self.genes,
            species=species or "human",
            freeze_backbone=True,
        )
        self._finalize_torch_model()

    def _finalize_torch_model(self) -> None:
        import torch

        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(torch.device(self.device))
        self.model.eval()

    def _torch_transform(self, X: sparse.csr_matrix) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("Torch model is not initialized")
        X = _clip_nonnegative(X)
        outputs: list[np.ndarray] = []
        device = torch.device(self.device)
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                batch = X[start : start + self.batch_size].toarray().astype(np.float32, copy=False)
                if self.valid_gene_mask is not None:
                    batch[:, ~self.valid_gene_mask] = 0
                tensor = torch.from_numpy(batch).to(device)
                _, cell_emb = self.model(tensor)
                outputs.append(cell_emb.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(outputs)


def _split_method(method: str) -> tuple[str, str]:
    if method.endswith("_prototype"):
        return method[: -len("_prototype")], "prototype"
    if method.endswith("_knn"):
        return method[: -len("_knn")], "knn"
    raise ValueError(f"Method must end with _knn or _prototype, got: {method}")


def _vote_knn(
    ref_emb: np.ndarray,
    query_emb: np.ndarray,
    ref_labels: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(int(k), ref_emb.shape[0]))
    # The full run evaluates up to 100k query cells per variant. Calling
    # sklearn's kneighbors on the whole matrix can materialize a very large
    # distance block, so compute cosine top-k in bounded chunks instead.
    ref_norm = normalize(ref_emb, norm="l2").astype(np.float32, copy=False)
    query_norm = normalize(query_emb, norm="l2").astype(np.float32, copy=False)
    chunks: list[np.ndarray] = []
    chunk_size = 4096
    ref_t = ref_norm.T
    for start in range(0, query_norm.shape[0], chunk_size):
        sims = query_norm[start : start + chunk_size] @ ref_t
        if k == sims.shape[1]:
            top = np.argsort(-sims, axis=1)
        else:
            part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            row = np.arange(part.shape[0])[:, None]
            order = np.argsort(-sims[row, part], axis=1)
            top = part[row, order]
        chunks.append(top.astype(np.int64, copy=False))
    indices = np.vstack(chunks)
    pred: list[str] = []
    conf: list[float] = []
    for row in indices:
        labels = [str(ref_labels[idx]) for idx in row]
        counts = Counter(labels)
        label, count = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0]
        pred.append(label)
        conf.append(float(count) / float(k))
    return np.asarray(pred, dtype=str), np.asarray(conf, dtype=np.float32)


def _prototype_predict(
    ref_emb: np.ndarray,
    query_emb: np.ndarray,
    ref_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(sorted({str(x) for x in ref_labels}), dtype=str)
    centroids = []
    for label in labels:
        centroids.append(ref_emb[ref_labels.astype(str) == label].mean(axis=0))
    centroid_matrix = normalize(np.vstack(centroids), norm="l2")
    query_norm = normalize(query_emb, norm="l2")
    sims = query_norm @ centroid_matrix.T
    best = np.argmax(sims, axis=1)
    return labels[best], sims[np.arange(sims.shape[0]), best].astype(np.float32)


def _metric_row(
    *,
    dataset_id: str,
    source_id: str,
    variant_id: str,
    method: str,
    embedding_method: str,
    transfer_method: str,
    task: str,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    n_shared_genes: int,
    n_reference_cells: int,
    n_query_cells: int,
) -> dict[str, Any]:
    true_labels = true_labels.astype(str)
    pred_labels = pred_labels.astype(str)
    ref_label_set = set(ref_labels.astype(str))
    true_in_ref = np.asarray([x in ref_label_set for x in true_labels], dtype=bool)
    return {
        "dataset_id": dataset_id,
        "source_id": source_id,
        "variant_id": variant_id,
        "method": method,
        "embedding_method": embedding_method,
        "transfer_method": transfer_method,
        "task": task,
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        "label_overlap_fraction": float(true_in_ref.mean()) if len(true_in_ref) else 0.0,
        "n_reference_cells": int(n_reference_cells),
        "n_query_cells": int(n_query_cells),
        "n_shared_genes": int(n_shared_genes),
        "n_ref_labels": int(len(ref_label_set)),
        "n_true_labels": int(len(set(true_labels))),
    }


def _prediction_rows(
    *,
    dataset_id: str,
    source_id: str,
    variant_id: str,
    method: str,
    task: str,
    obs: pd.DataFrame,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, Any]]:
    if "cell_id" in obs.columns:
        cell_ids = obs["cell_id"].astype(str).to_numpy()
    else:
        cell_ids = np.asarray([f"cell_{idx}" for idx in range(len(obs))], dtype=str)
    sample_ids = obs["sample_id"].astype(str).to_numpy() if "sample_id" in obs.columns else np.asarray([""] * len(obs))
    rows = []
    for i in range(len(obs)):
        rows.append(
            {
                "dataset_id": dataset_id,
                "source_id": source_id,
                "variant_id": variant_id,
                "method": method,
                "task": task,
                "cell_id": cell_ids[i],
                "sample_id": sample_ids[i],
                "true_label": str(true_labels[i]),
                "pred_label": str(pred_labels[i]),
                "confidence": float(confidence[i]),
            }
        )
    return rows


def _write_prediction_rows(
    writer: csv.DictWriter,
    *,
    dataset_id: str,
    source_id: str,
    variant_id: str,
    method: str,
    task: str,
    obs: pd.DataFrame,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidence: np.ndarray,
) -> int:
    rows = _prediction_rows(
        dataset_id=dataset_id,
        source_id=source_id,
        variant_id=variant_id,
        method=method,
        task=task,
        obs=obs,
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidence=confidence,
    )
    writer.writerows(rows)
    return len(rows)


def _variant_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["source_id", "method", "task"], dropna=False):
        baseline = group[group["variant_id"].astype(str).str.startswith("baseline")]
        if baseline.empty:
            continue
        base = float(baseline.iloc[0]["macro_f1"])
        source_id, method, task = keys
        for _, row in group.iterrows():
            score = float(row["macro_f1"])
            rows.append(
                {
                    "source_id": source_id,
                    "method": method,
                    "task": task,
                    "variant_id": row["variant_id"],
                    "baseline_score": base,
                    "variant_score": score,
                    "delta_vs_baseline": score - base,
                    "ratio_vs_baseline": (score / base) if base else np.nan,
                }
            )
    return pd.DataFrame(rows)


def evaluate_raw_label_transfer(
    *,
    output_dir: str | Path = paths.RUNS_DIR / "raw_label_transfer_smoke",
    synthetic_root: str | Path = paths.SYNTHETIC_DIR,
    source_ids: list[str] | None = None,
    variant_ids: list[str] | None = None,
    methods: list[str] | None = None,
    max_reference_cells: int = 500,
    max_query_cells: int = 200,
    min_shared_genes: int = 50,
    k: int = 15,
    device: str = "",
    batch_size: int = 16,
    seed: int = 3028,
    include_existing_seaad: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    methods = methods or ["expression_log1p_knn", "expression_log1p_prototype"]
    wanted_sources = set(source_ids or [])

    metrics_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    prediction_columns = [
        "dataset_id",
        "source_id",
        "variant_id",
        "method",
        "task",
        "cell_id",
        "sample_id",
        "true_label",
        "pred_label",
        "confidence",
    ]
    predictions_path = output_dir / "predictions.csv"
    prediction_count = 0

    variants = discover_new_synthetic_variants(synthetic_root)
    if include_existing_seaad:
        variants = [*variants, *discover_existing_seaad_variants()]
    if wanted_sources:
        variants = [item for item in variants if item["source_id"] in wanted_sources]
    wanted_variants = set(variant_ids or [])
    if wanted_variants:
        variants = [item for item in variants if item["variant_id"] in wanted_variants]

    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in variants:
        by_source.setdefault(item["source_id"], []).append(item)

    with predictions_path.open("w", newline="", encoding="utf-8") as pred_handle:
        pred_writer = csv.DictWriter(pred_handle, fieldnames=prediction_columns)
        pred_writer.writeheader()

        for source_id, source_variants in sorted(by_source.items()):
            adapter_cache: dict[tuple[str, tuple[str, ...]], EmbeddingAdapter] = {}
            ref_embedding_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
            source_dir = _source_dir_for_variant(source_variants[0]["path"])
            try:
                source_meta = _source_metadata(source_dir)
                ref_probe = _read_scdesign3_variant(source_variants[0]["path"], max_cells=1, seed=seed)
                source_path = Path(source_meta["source_path"])
                prepared_ref_dir = source_dir / "_prepared_input"
                if prepared_ref_dir.exists():
                    ref_bundle, ref_info = _load_reference_from_standard_bundle(
                        prepared_ref_dir,
                        genes=ref_probe.genes,
                        max_cells=max_reference_cells,
                        seed=seed,
                    )
                elif source_path.suffix == ".h5ad" and source_path.exists():
                    try:
                        ref_bundle, ref_info = _load_reference_from_h5ad(
                            source_path,
                            genes=ref_probe.genes,
                            excluded_cell_ids=source_meta["excluded_cell_ids"],
                            max_cells=max_reference_cells,
                            seed=seed,
                        )
                    except ValueError as exc:
                        if "No disjoint reference cells remain" not in str(exc):
                            raise
                        ref_bundle, ref_info = _load_reference_from_standard_bundle(
                            source_dir / "_prepared_input",
                            genes=ref_probe.genes,
                            max_cells=max_reference_cells,
                            seed=seed,
                        )
                else:
                    ref_bundle, ref_info = _load_reference_from_standard_bundle(
                        source_dir / "_prepared_input",
                        genes=ref_probe.genes,
                        max_cells=max_reference_cells,
                        seed=seed,
                    )
            except Exception as exc:
                for item in source_variants:
                    skip_rows.append(
                        {
                            "source_id": source_id,
                            "dataset_id": item["dataset_id"],
                            "variant_id": item["variant_id"],
                            "method": "",
                            "stage": "reference_load",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue

            for item in sorted(source_variants, key=lambda x: x["variant_id"]):
                dataset_id = item["dataset_id"]
                variant_id = item["variant_id"]
                try:
                    query_bundle = _read_scdesign3_variant(item["path"], max_cells=max_query_cells, seed=seed)
                    ref_aligned, query_aligned, shared_genes = _align_reference_and_query(
                        ref_bundle,
                        query_bundle,
                        min_shared_genes=min_shared_genes,
                    )
                    if "coarse_label" not in query_aligned.obs.columns and source_meta["native_to_coarse"]:
                        query_aligned.obs["coarse_label"] = (
                            query_aligned.obs["native_label"].astype(str).map(source_meta["native_to_coarse"]).fillna("unknown")
                        )
                except Exception as exc:
                    skip_rows.append(
                        {
                            "source_id": source_id,
                            "dataset_id": dataset_id,
                            "variant_id": variant_id,
                            "method": "",
                            "stage": "query_load",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                task_columns = {
                    "native_label": ("native_label", "native_label"),
                    "coarse_label": ("coarse_label", "coarse_label"),
                }

                for method in methods:
                    try:
                        embedding_method, transfer_method = _split_method(method)
                        cache_key = (embedding_method, tuple(shared_genes))
                        if cache_key not in ref_embedding_cache:
                            adapter = EmbeddingAdapter(
                                base_method=embedding_method,
                                genes=shared_genes,
                                species=source_meta.get("species", ""),
                                device=device,
                                batch_size=batch_size,
                            )
                            adapter.fit()
                            ref_emb = adapter.transform(ref_aligned.X)
                            adapter_cache[cache_key] = adapter
                            ref_embedding_cache[cache_key] = ref_emb
                        adapter = adapter_cache[cache_key]
                        ref_emb = ref_embedding_cache[cache_key]
                        query_emb = adapter.transform(query_aligned.X)
                    except Exception as exc:
                        skip_rows.append(
                            {
                                "source_id": source_id,
                                "dataset_id": dataset_id,
                                "variant_id": variant_id,
                                "method": method,
                                "stage": "embedding",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(limit=3),
                            }
                        )
                        continue

                    for task, (ref_col, query_col) in task_columns.items():
                        if ref_col not in ref_aligned.obs.columns or query_col not in query_aligned.obs.columns:
                            skip_rows.append(
                                {
                                    "source_id": source_id,
                                    "dataset_id": dataset_id,
                                    "variant_id": variant_id,
                                    "method": method,
                                    "task": task,
                                    "stage": "labels",
                                    "reason": f"Missing labels ref={ref_col} query={query_col}",
                                }
                            )
                            continue
                        ref_labels = ref_aligned.obs[ref_col].astype(str).to_numpy()
                        true_labels = query_aligned.obs[query_col].astype(str).to_numpy()
                        usable = ~pd.Series(true_labels).isin(["", "nan", "None", "unknown"]).to_numpy()
                        if int(usable.sum()) == 0:
                            skip_rows.append(
                                {
                                    "source_id": source_id,
                                    "dataset_id": dataset_id,
                                    "variant_id": variant_id,
                                    "method": method,
                                    "task": task,
                                    "stage": "labels",
                                    "reason": "No usable query truth labels.",
                                }
                            )
                            continue
                        if transfer_method == "knn":
                            pred, conf = _vote_knn(ref_emb, query_emb, ref_labels, k=k)
                        elif transfer_method == "prototype":
                            pred, conf = _prototype_predict(ref_emb, query_emb, ref_labels)
                        else:
                            raise ValueError(f"Unsupported transfer method: {transfer_method}")

                        metrics_rows.append(
                            _metric_row(
                                dataset_id=dataset_id,
                                source_id=source_id,
                                variant_id=variant_id,
                                method=method,
                                embedding_method=embedding_method,
                                transfer_method=transfer_method,
                                task=task,
                                true_labels=true_labels[usable],
                                pred_labels=pred[usable],
                                ref_labels=ref_labels,
                                n_shared_genes=len(shared_genes),
                                n_reference_cells=ref_aligned.X.shape[0],
                                n_query_cells=int(usable.sum()),
                            )
                        )
                        prediction_count += _write_prediction_rows(
                            pred_writer,
                            dataset_id=dataset_id,
                            source_id=source_id,
                            variant_id=variant_id,
                            method=method,
                            task=task,
                            obs=query_aligned.obs.iloc[np.flatnonzero(usable)].reset_index(drop=True),
                            true_labels=true_labels[usable],
                            pred_labels=pred[usable],
                            confidence=conf[usable],
                        )

    metric_columns = [
        "dataset_id",
        "source_id",
        "variant_id",
        "method",
        "embedding_method",
        "transfer_method",
        "task",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "label_overlap_fraction",
        "n_reference_cells",
        "n_query_cells",
        "n_shared_genes",
        "n_ref_labels",
        "n_true_labels",
    ]
    skip_columns = ["source_id", "dataset_id", "variant_id", "method", "task", "stage", "reason", "traceback"]
    metrics = pd.DataFrame(metrics_rows, columns=metric_columns)
    skips = pd.DataFrame(skip_rows, columns=skip_columns)
    variant_scores = _variant_scores(metrics)

    metrics_path = output_dir / "metrics.csv"
    skips_path = output_dir / "skips_and_failures.csv"
    variant_scores_path = output_dir / "variant_scores.csv"
    metrics.to_csv(metrics_path, index=False)
    skips.to_csv(skips_path, index=False)
    variant_scores.to_csv(variant_scores_path, index=False)

    catalog = build_dataset_catalog(output_dir=output_dir / "dataset_catalog")
    summary = {
        "output_dir": str(output_dir),
        "n_sources": len(by_source),
        "n_variants": len(variants),
        "methods": methods,
        "max_reference_cells": max_reference_cells,
        "max_query_cells": max_query_cells,
        "min_shared_genes": min_shared_genes,
        "k": k,
        "device": device,
        "batch_size": batch_size,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "skips_path": str(skips_path),
        "variant_scores_path": str(variant_scores_path),
        "dataset_catalog": catalog,
        "n_metric_rows": int(len(metrics)),
        "n_prediction_rows": int(prediction_count),
        "n_skips": int(len(skips)),
    }
    write_json(summary, output_dir / "run_summary.json")
    return summary
