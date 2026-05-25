from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def train_reference_logistic_distribution(
    *,
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    query_embeddings: np.ndarray,
    consensus_labels: list[str],
    unknown_label: str,
    min_cells_per_class: int = 2,
    seed: int = 3028,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Train a balanced multinomial logistic head on reference embeddings only.

    Query labels are never used. The returned distribution includes an explicit
    unknown column derived from low maximum known-class probability.
    """
    ref_emb = normalize(np.asarray(reference_embeddings, dtype=np.float32), norm="l2")
    query_emb = normalize(np.asarray(query_embeddings, dtype=np.float32), norm="l2")
    labels = np.asarray(reference_labels, dtype=object)
    known_labels = [label for label in consensus_labels if label != unknown_label]
    known = np.isin(labels, known_labels)
    counts = Counter(labels[known].astype(str))
    usable_labels = sorted(label for label, count in counts.items() if count >= min_cells_per_class)
    usable = np.isin(labels, usable_labels)
    if len(usable_labels) < 2 or int(usable.sum()) < 4:
        return None

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(ref_emb[usable], labels[usable].astype(str))
    proba = clf.predict_proba(query_emb).astype(np.float32)

    distribution = np.zeros((query_emb.shape[0], len(consensus_labels)), dtype=np.float32)
    label_to_idx = {label: idx for idx, label in enumerate(consensus_labels)}
    for class_idx, label in enumerate(clf.classes_.astype(str)):
        if label in label_to_idx:
            distribution[:, label_to_idx[label]] = proba[:, class_idx]

    known_mass_raw = distribution[:, [label_to_idx[label] for label in known_labels]].sum(axis=1, keepdims=True)
    known_mass_raw = np.clip(known_mass_raw, 1e-12, None)
    known_prob = distribution[:, : len(known_labels)] / known_mass_raw
    max_known = known_prob.max(axis=1, keepdims=True)
    unknown_mass = np.clip(1.0 - max_known, 0.0, 0.75)
    distribution[:, : len(known_labels)] = known_prob * (1.0 - unknown_mass)
    distribution[:, label_to_idx[unknown_label]] = unknown_mass.ravel()
    distribution /= np.clip(distribution.sum(axis=1, keepdims=True), 1e-12, None)

    metadata = {
        "classifier": "LogisticRegression",
        "training_source": "reference_embeddings_only",
        "classes": [str(x) for x in clf.classes_.tolist()],
        "n_train_cells": int(usable.sum()),
        "n_query_cells": int(query_emb.shape[0]),
        "min_cells_per_class": int(min_cells_per_class),
        "class_counts": {str(label): int(counts[label]) for label in usable_labels},
        "unknown_mass_rule": "unknown_mass = clip(1 - max_known_probability, 0, 0.75)",
    }
    return distribution.astype(np.float32), metadata
