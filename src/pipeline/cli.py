#!/usr/bin/env python3
"""Run ingest -> clean -> embed (CLIP) -> train (fastai, GPU) -> eval for the image safety classifier demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.clean import (
    build_training_pairs,
    deduplicate_by_content_hash,
    filter_quality,
    summarize,
)
from pipeline.eval import evaluate
from pipeline.ingest import ingest_directory
from pipeline.train import save_classifier, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--target-recall", type=float, default=0.98)
    args = parser.parse_args()

    raw = ingest_directory(args.data_dir)
    deduped = deduplicate_by_content_hash(raw, args.data_dir)
    cleaned = filter_quality(deduped, args.data_dir)
    pairs = build_training_pairs(cleaned, args.data_dir)

    print("Ingest summary:", json.dumps(summarize(cleaned), indent=2))

    result = train_classifier(pairs)
    metrics = evaluate(result.learner, result.df, target_recall=args.target_recall)

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "classifier.pkl"
    save_classifier(result.learner, result.model_version, metrics.threshold, model_path)

    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_version": result.model_version,
                "train_size": result.train_size,
                "target_recall": metrics.target_recall,
                "threshold": metrics.threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "passed_gate": metrics.passed_gate,
                "confusion_matrix": metrics.confusion,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(metrics.report)
    print(f"Artifacts: {model_path}")
    print(f"Metrics: {metrics_path}")
    if not metrics.passed_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
