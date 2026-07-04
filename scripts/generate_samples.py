"""Generate abstract, synthetic placeholder images + their JSONL metadata.

IMPORTANT: every image here is a procedurally-drawn color/shape composition.
Category labels (graphic / sexual / other / safe) are arbitrary color-coded
identifiers used only to exercise pipeline shape (ingest -> clean -> embed ->
train -> eval). Nothing here depicts, resembles, or references real graphic
or sexual content of any kind -- these are solid-color blobs/polygons/stripes,
not photographs.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

IMG_SIZE = 64
N_PER_CATEGORY = 8
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"

# Base RGB palette + shape per category -- arbitrary, abstract, non-representational.
# "safe" and "other" deliberately share a muted/cool palette family so the
# resulting feature space isn't trivially separable, giving the PR-curve
# threshold sweep real (if modest) work to do.
PALETTES = {
    "safe": ((60, 130, 190), "ellipse"),
    "graphic": ((205, 45, 30), "polygon"),
    "sexual": ((190, 60, 165), "ellipse"),
    "other": ((90, 120, 140), "stripes"),
}


def _seed_for(id_: str) -> int:
    return int(hashlib.sha256(id_.encode()).hexdigest(), 16) % (2**32)


def _jitter(rng: random.Random, c: int, spread: int = 30) -> int:
    return max(0, min(255, c + rng.randint(-spread, spread)))


def _jittered_bbox(rng: random.Random) -> tuple[int, int, int, int]:
    pad = rng.randint(6, 16)
    x0, y0 = pad, pad
    x1, y1 = IMG_SIZE - pad, IMG_SIZE - pad
    return (x0, y0, x1, y1)


def _jittered_polygon(rng: random.Random) -> list[tuple[int, int]]:
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    n_points = rng.randint(5, 8)
    points = []
    for i in range(n_points):
        angle = (2 * 3.14159 * i / n_points) + rng.uniform(-0.3, 0.3)
        radius = rng.randint(18, 28)
        points.append((cx + int(radius * _cos(angle)), cy + int(radius * _sin(angle))))
    return points


def _cos(angle: float) -> float:
    import math

    return math.cos(angle)


def _sin(angle: float) -> float:
    import math

    return math.sin(angle)


def _draw_stripes(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], rng: random.Random) -> None:
    stripe_w = rng.randint(6, 10)
    for x in range(0, IMG_SIZE, stripe_w * 2):
        draw.rectangle([x, 0, x + stripe_w, IMG_SIZE], fill=color)


def _draw(category: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    base, shape = PALETTES[category]
    color = tuple(_jitter(rng, c) for c in base)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    if shape == "ellipse":
        draw.ellipse(_jittered_bbox(rng), fill=color)
    elif shape == "polygon":
        draw.polygon(_jittered_polygon(rng), fill=color)
    elif shape == "stripes":
        _draw_stripes(draw, color, rng)
    return img.filter(ImageFilter.GaussianBlur(1))


def main() -> None:
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    counter = 1
    for category in ("safe", "graphic", "sexual", "other"):
        for _ in range(N_PER_CATEGORY):
            id_ = f"img{counter:03d}"
            seed = _seed_for(id_)
            img = _draw(category, seed)
            rel_path = f"images/{id_}.png"
            img.save(OUT_DIR / rel_path, format="PNG")
            rows.append(
                {
                    "id": id_,
                    "image_path": rel_path,
                    "category": category,
                    "is_flagged": category != "safe",
                    "annotator_id": f"ann-{(counter % 4) + 1:02d}",
                    "reviewer_approved": True,
                    "width": IMG_SIZE,
                    "height": IMG_SIZE,
                }
            )
            counter += 1

    with (OUT_DIR / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(rows)} synthetic images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
