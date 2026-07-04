import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    subprocess.check_call(
        [sys.executable, "-m", "pipeline.cli", "--out", str(out)],
        cwd=ROOT,
    )
    return out


def test_pipeline_produces_model(artifacts):
    assert (artifacts / "classifier.pkl").exists()
    metrics = json.loads((artifacts / "metrics.json").read_text())
    assert metrics["train_size"] >= 32
    assert metrics["passed_gate"] is True
    assert metrics["recall"] >= metrics["target_recall"]
    assert 0.0 <= metrics["threshold"] <= 1.0


def test_ingest_rejects_bad_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    from pipeline.ingest import ingest_directory

    with pytest.raises(ValueError, match="invalid JSON"):
        ingest_directory(tmp_path)


def test_embed_image_is_deterministic(tmp_path):
    from pipeline.embed import embed_image

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 60, 200)).save(img_path)

    vec1 = embed_image(img_path)
    vec2 = embed_image(img_path)
    assert np.allclose(vec1, vec2)
    assert vec1.shape == (512,)


def test_threshold_selection_meets_target_recall():
    from pipeline.eval import _select_operating_point

    y_true = [0] * 10 + [1] * 10
    rng = np.random.default_rng(0)
    scores_neg = rng.uniform(0.0, 0.5, size=10)
    scores_pos = rng.uniform(0.4, 1.0, size=10)
    y_scores = np.concatenate([scores_neg, scores_pos])

    threshold, precision, recall, reachable = _select_operating_point(y_true, y_scores, target_recall=0.9)
    assert reachable
    assert recall >= 0.9
    assert 0.0 <= threshold <= 1.0
