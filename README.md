# Image Safety Classifier Pipeline

Portfolio demo of an **end-to-end image moderation classifier workflow**: human annotation ingest, image QA, CLIP-embedding + fastai linear-probe training on GPU, precision/recall-curve-tuned evaluation, and a small inference API.

Mirrors a production milestone: image review exports from a moderation annotation queue, used to train a **CLIP-based image safety classifier** flagging graphic and sexual content, tuned for high recall via precision-recall curve analysis.

> **Synthetic data only.** All sample images under `data/samples/images/` are procedurally generated abstract color/shape compositions (see `scripts/generate_samples.py`), used solely to exercise the pipeline's shape. They are not photographs and do not depict, resemble, or reference any real graphic or sexual content - category labels are arbitrary color-coded identifiers for pipeline testing, nothing more.

---

## Production workflow (what companies run)

```
Annotators (image moderation queue: Sightengine / Rekognition review UI / internal tool)
        │
        ▼
  Export JSONL  ──►  Ingest + Pydantic validation
        │
        ▼
  Clean (corrupt-file check, dimension QA, content-hash dedup, label balance)
        │
        ▼
  Embed images (CLIP ViT image encoder, frozen)
        │
        ▼
  Train classifier (linear-probe head on embeddings, fastai, GPU)
        │
        ▼
  Eval harness (precision-recall curve sweep, threshold tuned for target recall, category slices)
        │
        ▼
  Registry (Hugging Face Hub / MLflow)  ──►  FastAPI / Triton gate
```

### Typical company stack

| Layer | Tools |
|-------|-------|
| Annotation | Labelbox, Sightengine, Scale AI review queue, internal moderation tools |
| Storage | S3 / GCS, PostgreSQL, Parquet on object store |
| Orchestration | Airflow, Prefect, Dagster, Make/just + CI |
| Training | **PyTorch**, **Hugging Face Transformers** (`CLIPModel`), **fastai**, optional PEFT/LoRA, Accelerate |
| Tracking | ClearML, MLflow, Weights & Biases |
| Eval | Custom precision-recall-curve gates, held-out red-team image sets |
| Serving | **FastAPI**, ONNX Runtime, **Triton Inference Server** |
| Infra | Docker, Kubernetes, GitHub Actions |

### Models

- **Backbone:** `openai/clip-vit-base-patch32`, frozen - this demo actually loads and runs it (not a stand-in), extracting real 512-dim CLIP image embeddings.
- **Head:** a linear-probe classifier trained with **fastai** (`tabular_learner`) on top of the frozen embeddings - the standard "frozen encoder + lightweight head" pattern used because it's far more data-efficient than training a vision model from scratch, and CLIP's embeddings already encode broad semantic content from its web-scale pretraining.
- **Compute:** fastai/PyTorch automatically train on the local NVIDIA GPU when available (`torch.cuda.is_available()`), falling back to CPU transparently in CI runners with no GPU - no branching required in the code.
- **Task:** binary flagged/unflagged classification, with a `category` field retained for slicing (`graphic` / `sexual` / `other` / `safe`).

`configs/clip_train.yaml` documents the **at-scale** production version of this same architecture (larger labeled dataset, optional LoRA-tuning of the CLIP vision tower, ClearML tracking, ONNX export for Triton) - this repo already runs the real, smaller-scale version of it, not a substitute.

---

## Glossary

| Term | Meaning |
|------|---------|
| **CLIP** | **Contrastive Language-Image Pretraining** (OpenAI) - jointly trains an image encoder and text encoder on ~400M image-caption pairs so matching pairs land close together in a shared embedding space. Produces image embeddings that capture rich semantic content, not just low-level pixel statistics. |
| **Linear probe** | Freezing a pretrained encoder and training only a small linear (or logistic) classifier head on top of its embeddings. Cheap, data-efficient, and a common way to evaluate/use foundation-model embeddings. |
| **LoRA** | **Low-Rank Adaptation** - a PEFT method that adds small trainable rank-decomposition matrices to frozen layers, letting you fine-tune part of a large model (e.g. CLIP's vision tower) without updating all its weights. |
| **PEFT** | **Parameter-Efficient Fine-Tuning** - Hugging Face library for LoRA-style and other lightweight fine-tuning methods. |
| **Precision-recall curve / operating point** | A sweep of a classifier's decision threshold showing the precision/recall tradeoff at each point. "Tuning for high recall" means picking the least permissive threshold that still clears a target recall floor, maximizing precision subject to that constraint. |

---

## Quick start

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. Install it once, then let `uv` handle the virtualenv and Python toolchain.

```bash
# Create the venv and install deps (incl. the dev group) from uv.lock
uv sync

# Full pipeline: ingest → clean → embed (CLIP) → train (fastai, GPU) → eval → save artifact
uv run run-pipeline --data-dir data/samples --out artifacts/

# Start inference API (after training)
uv run uvicorn serve.app:app --reload
curl -s -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"image_path": "data/samples/images/img001.png"}' | jq

uv run pytest
```

The first `run-pipeline` invocation downloads the CLIP checkpoint (~600MB, cached by
`transformers` under `~/.cache/huggingface`) - subsequent runs are fast. Training itself
runs on the GPU automatically if one is present.

---

## Engineering work per milestone

1. **Annotation spec** - JSON schema for `image_path`, `category`, `is_flagged`, reviewer notes
2. **Ingest pipeline** - validate exports, reject malformed rows, audit trail
3. **Cleaning** - corrupt-file / dimension QA, exact content-hash dedup, label balance
4. **Embedding + training job** - frozen CLIP forward pass, linear-probe head trained via fastai, reproducible seed, GPU-accelerated
5. **Eval gate** - sweep the precision-recall curve, pick the threshold that maximizes precision subject to a target recall floor (e.g. 0.98); block release if unreachable
6. **Serving** - latency-bounded API applying the tuned threshold in front of the moderation stack

---

## Layout

```
data/samples/          # synthetic abstract placeholder images + JSONL annotation export
scripts/                # one-off synthetic sample-image generator
src/pipeline/          # ingest, clean, embed (CLIP), train (fastai), eval, schemas, cli
serve/                 # FastAPI classifier API
tests/
configs/clip_train.yaml # at-scale production training config (LoRA option, registry, serving)
pyproject.toml         # project metadata, deps, `run-pipeline` entry point
uv.lock                # pinned, reproducible dependency set
```

## License

MIT - demonstration code only.
