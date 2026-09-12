"""
Pydantic response models for the ML feature endpoints:
  - AI Image Similarity Search        (cross-image library)
  - Intra-Image Similar Object Search (within a single image)
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


class HistoryDuplicateItem(BaseModel):
    """One image from history with its duplicate status."""
    image_id: str
    filename: str
    download_url: str
    operation_type: str
    created_at: str
    is_duplicate: bool
    duplicate_type: Optional[str]        # None when unique
    hamming_distance: Optional[int]      # None when unique
    group_id: Optional[int]              # images in the same duplicate group share an id


class HistoryScanResponse(BaseModel):
    """Result of scanning the user's full history for duplicates."""
    total_scanned: int
    total_duplicates: int
    total_unique: int
    groups_found: int                    # number of distinct duplicate groups
    items: List[HistoryDuplicateItem]    # all history images, annotated


# ── Intra-Image Similar Object Detection ──────────────────────────────────────

class DetectedObjectGroup(BaseModel):
    """One cluster of visually similar regions found inside a single image."""
    group_index:    int           # 0-based, maps to a colour in the UI
    instance_count: int           # how many times this object/pattern was found
    locations:      List[str]     # plain-English positions, e.g. ["top-left", "center"]
    avg_similarity: float         # mean pairwise cosine similarity 0–1
    thumbnail_b64:  str           # base64 JPEG crop of the representative patch
    color_rgb:      List[int]     # [R, G, B] accent colour for this group


class SimilarObjectsResponse(BaseModel):
    image_width:         int
    image_height:        int
    groups:              List[DetectedObjectGroup]   # one entry per similar-object type
    annotated_image_b64: str    # base64 PNG — ONE labelled box per instance, no overlap
    summary:             str    # plain-English one-liner, e.g. "Found 3 types of similar objects"
