"""Load and validate image-moderation annotation exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schemas import ImageAnnotationRecord


def load_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def ingest_directory(data_dir: Path) -> list[ImageAnnotationRecord]:
    records: list[ImageAnnotationRecord] = []
    errors: list[str] = []

    for path in sorted(data_dir.glob("*.jsonl")):
        for row in load_jsonl(path):
            try:
                records.append(ImageAnnotationRecord.model_validate(row))
            except Exception as exc:  # noqa: BLE001 - collect validation errors
                errors.append(f"{path}: {exc}")

    if errors:
        raise ValueError("Ingest failed:\n" + "\n".join(errors[:20]))

    return records
