"""Pydantic models for image-moderation annotation exports."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ImageCategory(str, Enum):
    safe = "safe"
    graphic = "graphic"
    sexual = "sexual"
    other = "other"


class ImageAnnotationRecord(BaseModel):
    """One row from an image-moderation queue export."""

    id: str
    image_path: str = Field(min_length=1)
    category: ImageCategory
    is_flagged: bool
    annotator_id: str
    reviewer_approved: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
    notes: Optional[str] = None

    @field_validator("image_path")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class ClassifierRequest(BaseModel):
    image_path: str = Field(min_length=1)


class ClassifierResponse(BaseModel):
    image_path: str
    is_flagged: bool
    flagged_probability: float
    decision_threshold: float
    model_version: str
