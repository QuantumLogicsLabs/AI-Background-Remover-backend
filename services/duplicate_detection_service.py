"""
Duplicate Image Detection Service
====================================
Detects four kinds of duplicates:

    1. Exact        — identical bytes (MD5 hash match)
    2. Resized      — same content, different dimensions (pHash distance ≤ 4)
    3. Cropped      — subset of another image (structural similarity + overlap)
    4. Similar      — slightly modified (brightness/contrast/filter tweaks)
                      pHash distance ≤ 10

Uses perceptual hashing (pHash / dHash) from the `imagehash` library if
available, with a pure-Pillow fallback for environments where imagehash is
not installed.

Pipeline:
    Image bytes
        ↓
    MD5 hash         → exact duplicate if matches stored hash
        ↓
    pHash (64-bit)   → compute Hamming distance against stored hashes
        ↓
    Classify by threshold:
        distance == 0  → exact / resized (check dimensions)
        distance ≤ 4   → resized duplicate
        distance ≤ 10  → cropped / similar duplicate
        distance > 10  → unique
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Optional imagehash ────────────────────────────────────────────────────────
try:
    import imagehash  # type: ignore[import]
    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False
    logger.info("[Duplicate] imagehash not installed — using built-in DCT pHash fallback.")

# ── Thresholds ────────────────────────────────────────────────────────────────
EXACT_THRESHOLD = 0       # Hamming distance == 0 → exact/resized
RESIZED_THRESHOLD = 4     # 1–4 bits different → resized
CROPPED_THRESHOLD = 10    # 5–10 bits → cropped / similar modification
# > 10 → unique

# ── Duplicate type labels ─────────────────────────────────────────────────────
DUP_EXACT = "exact"
DUP_RESIZED = "resized"
DUP_CROPPED = "cropped"
DUP_SIMILAR = "similar"
DUP_UNIQUE = "unique"


# ─────────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _phash_imagehash(img: Image.Image) -> str:
    """64-bit pHash as hex string via imagehash library."""
    h = imagehash.phash(img, hash_size=8)
    return str(h)  # hex string like "f8c8e0f8c0e0f8c0"


def _phash_fallback(img: Image.Image) -> str:
    """
    Pure-Pillow pHash fallback (DCT-based, 64-bit).
    Compatible with imagehash output format.
    """
    # Step 1: reduce to 32×32 greyscale
    small = img.convert("L").resize((32, 32), Image.LANCZOS)
    pixels = np.array(small, dtype=np.float32)

    # Step 2: 2D DCT via separable 1D DCTs
    def dct1d(x: np.ndarray) -> np.ndarray:
        N = x.shape[-1]
        n = np.arange(N)
        k = n[:, np.newaxis]
        T = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        return (T @ x.T).T

    dct = dct1d(dct1d(pixels).T).T

    # Step 3: top-left 8×8 low-frequency component
    dct_low = dct[:8, :8]

    # Step 4: threshold at median
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()

    # Step 5: pack bits into hex string
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def _compute_phash(img: Image.Image) -> str:
    if _IMAGEHASH_AVAILABLE:
        return _phash_imagehash(img)
    return _phash_fallback(img)


def _hamming_distance(hash1: str, hash2: str) -> int:
    """Bit-level Hamming distance between two hex hash strings."""
    try:
        if _IMAGEHASH_AVAILABLE:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return int(h1 - h2)
        else:
            # Manual bit counting
            b1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            b2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            min_len = min(len(b1), len(b2))
            return sum(c1 != c2 for c1, c2 in zip(b1[:min_len], b2[:min_len]))
    except Exception:
        return 64  # treat as unique on error


def _classify_duplicate(
    distance: int,
    query_size: Tuple[int, int],
    candidate_size: Tuple[int, int],
) -> str:
    """Map Hamming distance + size comparison to a duplicate type label."""
    if distance > CROPPED_THRESHOLD:
        return DUP_UNIQUE

    if distance <= EXACT_THRESHOLD:
        if query_size == candidate_size:
            return DUP_EXACT
        return DUP_RESIZED

    if distance <= RESIZED_THRESHOLD:
        return DUP_RESIZED

    # distance 5–10: check if one might be a crop of the other
    qw, qh = query_size
    cw, ch = candidate_size
    area_ratio = (qw * qh) / max(cw * ch, 1)
    if 0.2 <= area_ratio <= 0.85 or 0.2 <= (1 / area_ratio) <= 0.85:
        return DUP_CROPPED

    return DUP_SIMILAR


# ─────────────────────────────────────────────────────────────────────────────
# Sync inner function for executor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_hashes_sync(image_bytes: bytes) -> Dict:
    """Extract MD5 + pHash + image size from raw bytes."""
    md5 = _md5(image_bytes)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    phash = _compute_phash(img)
    return {"md5": md5, "phash": phash, "width": img.width, "height": img.height}


# ─────────────────────────────────────────────────────────────────────────────


class DuplicateDetectionService:
    """Detects duplicate images by comparing MD5 and perceptual hashes."""

    COLLECTION = "image_hashes"

    async def extract_hashes(self, image_bytes: bytes) -> Dict:
        """
        Extract MD5 and pHash for the given image (async).
        Returns {"md5": str, "phash": str, "width": int, "height": int}
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_hashes_sync, image_bytes)

    async def store_hashes(
        self,
        user_id: str,
        image_id: str,
        filename: str,
        hashes: Dict,
        download_url: str = "",
    ) -> None:
        """Persist hash document for future comparisons."""
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
                    "md5": hashes["md5"],
                    "phash": hashes["phash"],
                    "width": hashes["width"],
                    "height": hashes["height"],
                    "download_url": download_url,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    async def find_duplicates(
        self,
        query_hashes: Dict,
        user_id: str,
        exclude_image_id: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict]:
        """
        Compare query image hashes against all stored images for this user.

        Returns a list of dicts sorted by duplicate likelihood:
            image_id, filename, duplicate_type, hamming_distance, download_url
        """
        from services.database import get_collection

        collection = get_collection(self.COLLECTION)
        cursor = collection.find(
            {"user_id": user_id},
            {"image_id": 1, "filename": 1, "md5": 1, "phash": 1,
             "width": 1, "height": 1, "download_url": 1, "_id": 0},
        )
        docs = await cursor.to_list(length=None)

        if not docs:
            return []

        if exclude_image_id:
            docs = [d for d in docs if d.get("image_id") != exclude_image_id]

        query_md5 = query_hashes["md5"]
        query_phash = query_hashes["phash"]
        query_size = (query_hashes["width"], query_hashes["height"])

        results: List[Dict] = []

        for doc in docs:
            # Fast path: exact byte match
            if doc.get("md5") == query_md5:
                dup_type = DUP_EXACT
                distance = 0
            else:
                distance = _hamming_distance(query_phash, doc.get("phash", ""))
                candidate_size = (doc.get("width", 0), doc.get("height", 0))
                dup_type = _classify_duplicate(distance, query_size, candidate_size)

            if dup_type == DUP_UNIQUE:
                continue

            results.append({
                "image_id": doc["image_id"],
                "filename": doc.get("filename", ""),
                "duplicate_type": dup_type,
                "hamming_distance": distance,
                "download_url": doc.get("download_url", ""),
            })

        # Sort: exact first, then by ascending Hamming distance
        order = {DUP_EXACT: 0, DUP_RESIZED: 1, DUP_CROPPED: 2, DUP_SIMILAR: 3}
        results.sort(key=lambda x: (order.get(x["duplicate_type"], 99), x["hamming_distance"]))
        return results[:max_results]

    async def check_single(
        self,
        image_bytes: bytes,
        user_id: str,
        exclude_image_id: Optional[str] = None,
    ) -> Dict:
        """
        One-shot: extract hashes + find duplicates, without storing.
        Used for the check-only API endpoint.
        """
        hashes = await self.extract_hashes(image_bytes)
        duplicates = await self.find_duplicates(hashes, user_id, exclude_image_id)
        return {
            "has_duplicates": len(duplicates) > 0,
            "total_found": len(duplicates),
            "duplicates": duplicates,
            "query_hashes": {
                "md5": hashes["md5"],
                "phash": hashes["phash"],
            },
        }
