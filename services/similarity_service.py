"""
AI Image Similarity Search Service
===================================
Generates embeddings for images using a lightweight MobileNet-based feature
extractor (torchvision) and stores them in MongoDB. Similarity is computed
via cosine similarity against all stored embeddings, with optional FAISS
acceleration when faiss-cpu is installed.

Embedding pipeline:
    Raw bytes → PIL → Resize(224×224) → Normalize → MobileNetV3 features
                                                           ↓
                                                    1280-dim vector
                                                           ↓
                                              L2-normalised → stored in MongoDB
                                                           ↓
                                              cosine sim search at query time
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Optional FAISS acceleration ───────────────────────────────────────────────
try:
    import faiss  # type: ignore[import]
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.info("[Similarity] faiss-cpu not installed — falling back to NumPy cosine similarity.")

# ── PyTorch / torchvision (required) ─────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("[Similarity] torch/torchvision not available — service will be non-functional.")

# ── Embedding dimensions ──────────────────────────────────────────────────────
_EMB_DIM = 576  # MobileNetV3-Small avgpool output

_TRANSFORM: Optional[object] = None
_MODEL: Optional[object] = None
_DEVICE: str = "cpu"


def _get_model_and_transform():
    """Lazy-initialise the model once and cache it."""
    global _MODEL, _TRANSFORM, _DEVICE

    if _MODEL is not None:
        return _MODEL, _TRANSFORM

    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch/torchvision is required for similarity search.")

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    # Strip the classifier so we get pure feature vectors
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(_DEVICE)
    _MODEL = model

    _TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info(f"[Similarity] MobileNetV3-Small loaded on {_DEVICE}.")
    return _MODEL, _TRANSFORM


# ─────────────────────────────────────────────────────────────────────────────


def _compute_embedding_sync(image_bytes: bytes) -> List[float]:
    """Synchronous embedding computation (runs inside run_in_executor)."""
    model, transform = _get_model_and_transform()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(_DEVICE)  # type: ignore[operator]

    with torch.no_grad():
        embedding = model(tensor)  # type: ignore[operator]

    # L2-normalise so cosine similarity == dot product
    vec = embedding.cpu().numpy().flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _cosine_similarity_numpy(
    query: List[float],
    candidates: List[List[float]],
) -> List[float]:
    """Pure-NumPy cosine similarity between one query and many candidates."""
    q = np.array(query, dtype=np.float32)
    matrix = np.array(candidates, dtype=np.float32)
    # Both are already L2-normalised, so dot product == cosine similarity
    scores = (matrix @ q).tolist()
    return scores


def _cosine_similarity_faiss(
    query: List[float],
    candidates: List[List[float]],
    top_k: int,
) -> List[tuple[int, float]]:
    """FAISS inner-product search (fastest for large collections)."""
    dim = len(query)
    index = faiss.IndexFlatIP(dim)  # Inner-product on L2-normalised vecs == cosine
    matrix = np.array(candidates, dtype=np.float32)
    index.add(matrix)

    q = np.array([query], dtype=np.float32)
    distances, indices = index.search(q, min(top_k, len(candidates)))
    return list(zip(indices[0].tolist(), distances[0].tolist()))


# ─────────────────────────────────────────────────────────────────────────────


class SimilarityService:
    """Manages image embeddings in MongoDB and performs similarity search."""

    COLLECTION = "image_embeddings"

    async def generate_embedding(self, image_bytes: bytes) -> List[float]:
        """Generate a normalised embedding vector for the given image bytes."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compute_embedding_sync, image_bytes)

    async def store_embedding(
        self,
        user_id: str,
        image_id: str,
        filename: str,
        embedding: List[float],
        download_url: str = "",
    ) -> None:
        """Persist an embedding document to MongoDB."""
        from services.database import get_collection
        from datetime import datetime, timezone

        collection = get_collection(self.COLLECTION)
        await collection.update_one(
            {"image_id": image_id, "user_id": user_id},
            {
                "$set": {
                    "image_id": image_id,
                    "user_id": user_id,
                    "filename": filename,
                    "embedding": embedding,
                    "download_url": download_url,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    async def find_similar(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 10,
        exclude_image_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Find the top-k most similar images for a given user.

        Returns a list of dicts with keys: image_id, filename, score, download_url.
        """
        from services.database import get_collection

        collection = get_collection(self.COLLECTION)

        # Fetch all embeddings for this user (per-user scoped search)
        cursor = collection.find(
            {"user_id": user_id},
            {"image_id": 1, "filename": 1, "embedding": 1, "download_url": 1, "_id": 0},
        )
        docs = await cursor.to_list(length=None)

        if not docs:
            return []

        # Optionally exclude the query image itself
        if exclude_image_id:
            docs = [d for d in docs if d.get("image_id") != exclude_image_id]

        if not docs:
            return []

        candidates = [d["embedding"] for d in docs]

        if _FAISS_AVAILABLE:
            ranked = _cosine_similarity_faiss(query_embedding, candidates, top_k)
            results = []
            for idx, score in ranked:
                if score < 0.1:  # filter noise
                    continue
                doc = docs[idx]
                results.append({
                    "image_id": doc["image_id"],
                    "filename": doc["filename"],
                    "score": round(float(score), 4),
                    "download_url": doc.get("download_url", ""),
                })
            return results[:top_k]
        else:
            scores = _cosine_similarity_numpy(query_embedding, candidates)
            paired = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            results = []
            for score, doc in paired[:top_k]:
                if score < 0.1:
                    continue
                results.append({
                    "image_id": doc["image_id"],
                    "filename": doc["filename"],
                    "score": round(float(score), 4),
                    "download_url": doc.get("download_url", ""),
                })
            return results

    async def get_embedding_count(self, user_id: str) -> int:
        """Return number of indexed images for this user."""
        from services.database import get_collection
        collection = get_collection(self.COLLECTION)
        return await collection.count_documents({"user_id": user_id})
