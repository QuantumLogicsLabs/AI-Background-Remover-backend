"""
Pydantic response models for the three ML feature endpoints:
  - AI Image Similarity Search
  - AI Image Categorization
  - Duplicate Image Detection
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


# ── Similarity Search ─────────────────────────────────────────────────────────

class SimilarityResult(BaseModel):
    image_id: str
    filename: str
    score: float          # cosine similarity 0–1 (higher = more similar)
    download_url: str


class SimilaritySearchResponse(BaseModel):
    query_image_id: str
    total_indexed: int    # how many images the user has indexed
    results: List[SimilarityResult]


class IndexImageResponse(BaseModel):
    image_id: str
    filename: str
    total_indexed: int
    message: str


# ── Categorization ────────────────────────────────────────────────────────────

class CategorizationResponse(BaseModel):
    image_id: str
    filename: str
    category: str         # e.g. "Portrait"
    confidence: float     # 0–1
    scores: Dict[str, float]   # per-category confidence scores
    method: str           # "clip" | "imagenet_heuristic"


# ── Duplicate Detection ───────────────────────────────────────────────────────

class DuplicateMatch(BaseModel):
    image_id: str
    filename: str
    duplicate_type: str   # "exact" | "resized" | "cropped" | "similar"
    hamming_distance: int
    download_url: str


class DuplicateCheckResponse(BaseModel):
    query_image_id: str
    has_duplicates: bool
    total_found: int
    duplicates: List[DuplicateMatch]
    query_md5: str
    query_phash: str


class IndexForDuplicatesResponse(BaseModel):
    image_id: str
    filename: str
    md5: str
    message: str
