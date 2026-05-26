from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from aos_agent import paths
from aos_agent.io import ensure_dir, write_json


CACHE_VERSION = "embedding-cache-v1"
DEFAULT_CACHE_ROOT = paths.AOS_ROOT / "artifacts" / "embedding_cache"


def embedding_cache_enabled() -> bool:
    value = os.environ.get("AOS_EMBEDDING_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disable", "disabled"}


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def model_fingerprint(base_method: str) -> dict[str, Any]:
    method = str(base_method)
    if method == "expression_log1p":
        files: list[Path] = []
    elif method == "geneformer_raw":
        root = paths.GENEFORMER_CHECKPOINT_DIR
        files = [root / "model.safetensors"]
    elif method in {"scgpt_brain_raw", "scgpt_human_raw"}:
        model_name = "brain" if method == "scgpt_brain_raw" else "human"
        root = paths.SCGPT_CHECKPOINT_ROOT / model_name
        files = [root / "best_model.pt", root / "vocab.json", root / "args.json"]
    elif method == "nicheformer_raw":
        root = paths.NICHEFORMER_CHECKPOINT_DIR
        files = [
            root / "nicheformer.ckpt",
            root / "model.h5ad",
            root / "merfish_mean_script.npy",
            root / "gene_name_id_dict_gc104M.pkl",
        ]
    elif method == "uce_4l_raw":
        root = paths.UCE_4L_MODEL_DIR
        files = [root / "4layer_model.torch", root / "all_tokens.torch", root / "species_chrom.csv", root / "species_offsets.pkl"]
    elif method in {"uce_33l_raw", "uce_33l_ima"}:
        root = paths.UCE_33L_MODEL_DIR
        files = [root / "33l_8ep_1024t_1280.torch", root / "all_tokens.torch", root / "species_chrom.csv", root / "species_offsets.pkl"]
    else:
        files = []
    return {
        "base_method": method,
        "files": [_file_fingerprint(path) for path in files],
    }


def _hash_array(handle: Any, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    handle.update(str(arr.dtype).encode("utf-8"))
    handle.update(str(arr.shape).encode("utf-8"))
    handle.update(memoryview(arr).cast("B"))


def matrix_fingerprint(X: Any) -> dict[str, Any]:
    X = sparse.csr_matrix(X)
    digest = hashlib.blake2b(digest_size=24)
    digest.update(CACHE_VERSION.encode("utf-8"))
    digest.update(str(X.shape).encode("utf-8"))
    digest.update(str(X.nnz).encode("utf-8"))
    _hash_array(digest, X.indptr)
    _hash_array(digest, X.indices)
    _hash_array(digest, X.data)
    return {
        "format": "csr",
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "nnz": int(X.nnz),
        "dtype": str(X.data.dtype),
        "hash": digest.hexdigest(),
    }


def embedding_cache_key(
    *,
    base_method: str,
    genes: list[str],
    species: str,
    matrix: Any,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "version": CACHE_VERSION,
        "base_method": str(base_method),
        "species": str(species or ""),
        "genes": [str(g) for g in genes],
        "matrix": matrix_fingerprint(matrix),
        "model": model_fingerprint(base_method),
        "extra": extra or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=24).hexdigest(), payload


def cache_entry_dir(cache_key: str, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    cache_key = str(cache_key)
    return Path(cache_root) / cache_key[:2] / cache_key


def load_embedding_cache(cache_key: str, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> np.ndarray | None:
    if not embedding_cache_enabled():
        return None
    entry_dir = cache_entry_dir(cache_key, cache_root)
    embedding_path = entry_dir / "embedding.npy"
    metadata_path = entry_dir / "metadata.json"
    if not embedding_path.exists() or not metadata_path.exists():
        return None
    try:
        return np.load(embedding_path, allow_pickle=False)
    except Exception:
        return None


def load_embedding_cache_metadata(cache_key: str, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> dict[str, Any]:
    entry_dir = cache_entry_dir(cache_key, cache_root)
    metadata_path = entry_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_embedding_cache(
    *,
    cache_key: str,
    embedding: np.ndarray,
    metadata: dict[str, Any],
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> Path | None:
    if not embedding_cache_enabled():
        return None
    entry_dir = ensure_dir(cache_entry_dir(cache_key, cache_root))
    embedding_path = entry_dir / "embedding.npy"
    tmp_path = entry_dir / f"embedding.{os.getpid()}.tmp.npy"
    array = np.asarray(embedding, dtype=np.float32)
    np.save(tmp_path, array, allow_pickle=False)
    os.replace(tmp_path, embedding_path)
    write_json(
        {
            **metadata,
            "cache_key": str(cache_key),
            "cache_version": CACHE_VERSION,
            "embedding_path": str(embedding_path),
            "embedding_shape": [int(x) for x in array.shape],
            "embedding_dtype": str(array.dtype),
        },
        entry_dir / "metadata.json",
    )
    return embedding_path
