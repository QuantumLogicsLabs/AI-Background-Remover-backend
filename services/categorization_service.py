"""
AI Image Categorization Service
=================================
Classifies images into predefined categories using a fine-tuned MobileNetV3
feature extractor plus zero-shot classification via CLIP-style keyword
matching, or a simple heuristic classifier as the fallback.

Categories:
    Portrait · Product · Food · Animal · Landscape · Document · Vehicle · Other

Pipeline:
    Raw bytes → PIL → Feature extraction (MobileNetV3)
                              ↓
                  Zero-shot CLIP matching (if transformers ≥ 4.x available)
                  OR keyword-driven heuristic from ImageNet top-5 labels
                              ↓
                  Category + confidence scores
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Category definitions ───────────────────────────────────────────────────────
CATEGORIES = [
    "Portrait",
    "Product",
    "Food",
    "Animal",
    "Landscape",
    "Document",
    "Vehicle",
    "Other",
]

# CLIP-style text prompts (one per category) for zero-shot classification
_PROMPTS = [
    "a photo of a person or human face portrait",
    "a photo of a product or item for sale on white background",
    "a photo of food or a meal",
    "a photo of an animal or pet",
    "a photo of a landscape or nature scenery",
    "a photo of a document, text, or paper",
    "a photo of a vehicle, car, truck, or transportation",
    "a photo of an object or scene",
]

# Keyword → category mapping used by the heuristic fallback (ImageNet labels)
_KEYWORD_MAP: Dict[str, str] = {
    # Portrait
    "face": "Portrait", "person": "Portrait", "man": "Portrait",
    "woman": "Portrait", "child": "Portrait", "portrait": "Portrait",
    "selfie": "Portrait", "people": "Portrait",
    # Product
    "bottle": "Product", "can": "Product", "box": "Product",
    "package": "Product", "product": "Product", "shirt": "Product",
    "shoe": "Product", "handbag": "Product", "backpack": "Product",
    "laptop": "Product", "keyboard": "Product", "phone": "Product",
    # Food
    "food": "Food", "pizza": "Food", "burger": "Food", "salad": "Food",
    "fruit": "Food", "cake": "Food", "coffee": "Food", "bread": "Food",
    "vegetable": "Food", "sushi": "Food", "ice cream": "Food",
    # Animal
    "cat": "Animal", "dog": "Animal", "bird": "Animal", "horse": "Animal",
    "cow": "Animal", "sheep": "Animal", "elephant": "Animal",
    "bear": "Animal", "animal": "Animal", "fish": "Animal",
    # Landscape
    "mountain": "Landscape", "sky": "Landscape", "beach": "Landscape",
    "forest": "Landscape", "field": "Landscape", "river": "Landscape",
    "ocean": "Landscape", "sunset": "Landscape", "landscape": "Landscape",
    "tree": "Landscape",
    # Document
    "document": "Document", "paper": "Document", "book": "Document",
    "text": "Document", "letter": "Document", "invoice": "Document",
    "receipt": "Document",
    # Vehicle
    "car": "Vehicle", "truck": "Vehicle", "bus": "Vehicle",
    "motorcycle": "Vehicle", "bicycle": "Vehicle", "airplane": "Vehicle",
    "boat": "Vehicle", "train": "Vehicle", "vehicle": "Vehicle",
}

# ── Optional CLIP via transformers ────────────────────────────────────────────
_CLIP_PIPE = None
_CLIP_LOADED = False

try:
    from transformers import pipeline as hf_pipeline, CLIPModel, CLIPProcessor  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# ── Optional torch/torchvision for ImageNet fallback ─────────────────────────
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_IMAGENET_MODEL = None
_IMAGENET_WEIGHTS = None
_IMAGENET_DEVICE = "cpu"


def _load_imagenet_model():
    global _IMAGENET_MODEL, _IMAGENET_WEIGHTS, _IMAGENET_DEVICE
    if _IMAGENET_MODEL is not None:
        return _IMAGENET_MODEL, _IMAGENET_WEIGHTS

    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch/torchvision required for image categorization.")

    _IMAGENET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    model.to(_IMAGENET_DEVICE)
    _IMAGENET_MODEL = model
    _IMAGENET_WEIGHTS = weights
    logger.info(f"[Categorization] MobileNetV3-Small (ImageNet) loaded on {_IMAGENET_DEVICE}.")
    return model, weights


def _load_clip_pipeline():
    global _CLIP_PIPE, _CLIP_LOADED
    if _CLIP_LOADED:
        return _CLIP_PIPE

    _CLIP_LOADED = True
    if not _TRANSFORMERS_AVAILABLE:
        logger.info("[Categorization] transformers not available — using ImageNet heuristic.")
        _CLIP_PIPE = None
        return None

    try:
        _CLIP_PIPE = hf_pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=-1,  # CPU always; set to 0 for GPU
        )
        logger.info("[Categorization] CLIP zero-shot pipeline loaded.")
    except Exception as exc:
        logger.warning(f"[Categorization] CLIP pipeline failed to load ({exc}); falling back to ImageNet heuristic.")
        _CLIP_PIPE = None

    return _CLIP_PIPE


# ─────────────────────────────────────────────────────────────────────────────


def _categorize_clip_sync(image_bytes: bytes) -> Dict:
    """CLIP zero-shot classification → returns category + confidence scores."""
    pipe = _load_clip_pipeline()
    if pipe is None:
        return _categorize_imagenet_sync(image_bytes)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    outputs = pipe(img, candidate_labels=_PROMPTS)

    # Outputs: [{"label": ..., "score": ...}, ...]
    # Map back to clean category names
    scores: Dict[str, float] = {}
    for item in outputs:
        label = item["label"]
        try:
            idx = _PROMPTS.index(label)
            cat = CATEGORIES[idx]
        except ValueError:
            cat = "Other"
        scores[cat] = round(float(item["score"]), 4)

    # Ensure all categories present
    for cat in CATEGORIES:
        scores.setdefault(cat, 0.0)

    top_category = max(scores, key=lambda k: scores[k])
    confidence = scores[top_category]

    return {
        "category": top_category,
        "confidence": confidence,
        "scores": scores,
        "method": "clip",
    }


def _categorize_imagenet_sync(image_bytes: bytes) -> Dict:
    """ImageNet top-5 label keyword matching as fallback."""
    model, weights = _load_imagenet_model()
    preprocess = weights.transforms()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(_IMAGENET_DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top5_probs, top5_indices = torch.topk(probs, 5)

    # Get human-readable labels
    categories_meta = weights.meta["categories"]
    top5_labels = [categories_meta[idx].lower() for idx in top5_indices[0].tolist()]

    # Keyword match
    category_votes: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    for label, prob in zip(top5_labels, top5_probs[0].tolist()):
        matched = "Other"
        for keyword, cat in _KEYWORD_MAP.items():
            if keyword in label:
                matched = cat
                break
        category_votes[matched] += float(prob)

    total = sum(category_votes.values()) or 1.0
    scores = {cat: round(v / total, 4) for cat, v in category_votes.items()}
    top_category = max(scores, key=lambda k: scores[k])

    return {
        "category": top_category,
        "confidence": scores[top_category],
        "scores": scores,
        "method": "imagenet_heuristic",
        "top_labels": top5_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────


class CategorizationService:
    """Classifies images into one of the predefined categories."""

    COLLECTION = "categorization_history"

    async def categorize(self, image_bytes: bytes) -> Dict:
        """
        Classify an image. Returns category, confidence, and per-category scores.

        Tries CLIP zero-shot first; falls back to ImageNet keyword heuristic.
        """
        loop = asyncio.get_event_loop()

        # Prefer CLIP if pipeline loads; otherwise use ImageNet heuristic
        if _TRANSFORMERS_AVAILABLE:
            return await loop.run_in_executor(None, _categorize_clip_sync, image_bytes)
        else:
            return await loop.run_in_executor(None, _categorize_imagenet_sync, image_bytes)

    async def save_result(
        self,
        user_id: str,
        image_id: str,
        filename: str,
        result: Dict,
        download_url: str = "",
    ) -> None:
        """Persist categorization result to MongoDB."""
        from services.database import get_collection
        from datetime import datetime, timezone

        collection = get_collection(self.COLLECTION)
        await collection.insert_one({
            "user_id": user_id,
            "image_id": image_id,
            "filename": filename,
            "category": result["category"],
            "confidence": result["confidence"],
            "scores": result["scores"],
            "method": result.get("method", "unknown"),
            "download_url": download_url,
            "created_at": datetime.now(timezone.utc),
        })
