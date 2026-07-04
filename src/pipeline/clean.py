"""Image QA and cleaning gates."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .schemas import ImageAnnotationRecord

MIN_DIM = 16
MAX_DIM = 4096


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_openable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def deduplicate_by_content_hash(
    records: list[ImageAnnotationRecord], data_dir: Path
) -> list[ImageAnnotationRecord]:
    seen: set[str] = set()
    out: list[ImageAnnotationRecord] = []
    for rec in records:
        path = data_dir / rec.image_path
        if not path.exists():
            continue
        key = _content_hash(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def filter_quality(
    records: list[ImageAnnotationRecord], data_dir: Path
) -> list[ImageAnnotationRecord]:
    out: list[ImageAnnotationRecord] = []
    for rec in records:
        if not rec.reviewer_approved:
            continue
        path = data_dir / rec.image_path
        if not _is_openable(path):
            continue
        with Image.open(path) as img:
            w, h = img.size
        if not (MIN_DIM <= w <= MAX_DIM and MIN_DIM <= h <= MAX_DIM):
            continue
        out.append(rec)
    return out


def build_training_pairs(
    records: list[ImageAnnotationRecord], data_dir: Path
) -> list[tuple[Path, bool]]:
    return [(data_dir / r.image_path, r.is_flagged) for r in records]


def summarize(records: list[ImageAnnotationRecord]) -> dict:
    cats = Counter(r.category.value for r in records)
    return {
        "count": len(records),
        "categories": dict(cats),
        "annotators": len({r.annotator_id for r in records}),
    }
