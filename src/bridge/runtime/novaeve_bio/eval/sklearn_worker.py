from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from novaeve_bio.eval.registry import ModelSpec
from novaeve_bio.eval.run import run_sklearn_model
from novaeve_bio.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec = ModelSpec(
        model_id=args.model_id,
        family="sklearn",
        evaluator="sklearn_pkl",
        raw={"checkpoint_dir": args.checkpoint_dir},
    )
    dataset = {"dataset_id": args.dataset_id, "npz_path": args.npz_path}
    output_dir = ensure_dir(args.output_dir)
    predictions, metrics = run_sklearn_model(spec, dataset, output_dir)
    pd.DataFrame(predictions).to_csv(Path(output_dir) / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(Path(output_dir) / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
