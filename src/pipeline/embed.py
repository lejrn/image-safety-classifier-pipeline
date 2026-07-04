"""Frozen CLIP image encoder used to produce classifier-head input features.

Loads openai/clip-vit-base-patch32 (frozen, no gradient) via Hugging Face
transformers and runs only the vision tower. Uses the local NVIDIA GPU when
available (torch.cuda.is_available()), falling back to CPU transparently
(e.g. in CI runners with no GPU) -- no branching required elsewhere in the
pipeline, since train.py/eval.py/serve only ever see a numpy feature vector.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_model() -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = _device()
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device


@torch.inference_mode()
def embed_batch(paths: list[Path]) -> np.ndarray:
    model, processor, device = _load_model()
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    output = model.get_image_features(**inputs)
    # transformers>=4.5x returns a BaseModelOutputWithPooling whose
    # pooler_output has been overwritten with the projected image features.
    features = output.pooler_output if hasattr(output, "pooler_output") else output
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def embed_image(path: Path) -> np.ndarray:
    return embed_batch([path])[0]
