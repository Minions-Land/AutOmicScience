from __future__ import annotations

import numpy as np
from scipy import sparse

from scmas.embedding_cache import embedding_cache_key, load_embedding_cache, save_embedding_cache


def test_embedding_cache_key_respects_gene_order_and_matrix_content(tmp_path):
    X = sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
    key1, meta1 = embedding_cache_key(base_method="expression_log1p", genes=["A", "B", "C"], species="mouse", matrix=X)
    key2, _ = embedding_cache_key(base_method="expression_log1p", genes=["B", "A", "C"], species="mouse", matrix=X)
    key3, _ = embedding_cache_key(
        base_method="expression_log1p",
        genes=["A", "B", "C"],
        species="mouse",
        matrix=sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 4, 0]], dtype=np.float32)),
    )

    assert key1 != key2
    assert key1 != key3

    embedding = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    save_embedding_cache(cache_key=key1, embedding=embedding, metadata=meta1, cache_root=tmp_path)
    loaded = load_embedding_cache(key1, cache_root=tmp_path)
    assert loaded is not None
    assert np.allclose(loaded, embedding)
