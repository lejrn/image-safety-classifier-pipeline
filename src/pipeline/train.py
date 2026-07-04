"""Train a linear-probe classifier head on frozen CLIP embeddings via fastai.

fastai/PyTorch pick the local NVIDIA GPU automatically when present
(torch.cuda.is_available()) and fall back to CPU otherwise -- no branching
needed here for CI (no GPU) vs. local dev (RTX 4060).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastai.tabular.all import (
    CategoryBlock,
    Learner,
    Normalize,
    RandomSplitter,
    TabularDataLoaders,
    load_learner,
    tabular_learner,
)

from .embed import EMBED_DIM, embed_batch

EMBED_COLS = [f"e{i}" for i in range(EMBED_DIM)]
LABEL_COL = "is_flagged"


@dataclass
class TrainResult:
    learner: Learner
    df: pd.DataFrame
    train_size: int
    model_version: str


def build_dataframe(pairs: list[tuple[Path, bool]]) -> pd.DataFrame:
    paths = [p[0] for p in pairs]
    labels = [int(p[1]) for p in pairs]
    embeddings = embed_batch(paths)
    df = pd.DataFrame(embeddings, columns=EMBED_COLS)
    df[LABEL_COL] = labels
    return df


def train_classifier(
    pairs: list[tuple[Path, bool]],
    model_version: str = "clip-linear-probe-v1",
    epochs: int = 8,
) -> TrainResult:
    df = build_dataframe(pairs)

    splits = RandomSplitter(valid_pct=0.2, seed=42)(range(len(df)))
    dls = TabularDataLoaders.from_df(
        df,
        y_names=LABEL_COL,
        cont_names=EMBED_COLS,
        y_block=CategoryBlock,
        procs=[Normalize],
        valid_idx=list(splits[1]),
        bs=8,
    )
    learn = tabular_learner(dls, metrics=[])
    learn.fit_one_cycle(epochs)

    return TrainResult(learner=learn, df=df, train_size=len(pairs), model_version=model_version)


def save_classifier(learn: Learner, version: str, threshold: float, model_path: Path) -> None:
    learn.model_version = version
    learn.decision_threshold = threshold
    model_path.parent.mkdir(parents=True, exist_ok=True)
    learn.export(model_path)


def load_classifier(model_path: Path) -> tuple[Learner, str, float]:
    learn = load_learner(model_path)
    version = getattr(learn, "model_version", "unknown")
    threshold = getattr(learn, "decision_threshold", 0.5)
    return learn, version, threshold
