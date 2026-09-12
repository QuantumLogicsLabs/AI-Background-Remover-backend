"""
Intra-Image Similarity Service
================================
Detects visually similar objects / repeated regions **within a single image**
and returns the results as human-readable **object groups**.

Algorithm
---------
1.  Load MobileNetV3-Small (classifier stripped) → 576-d feature vectors.
2.  Extract patches at multiple scales with a sliding window.
3.  Cluster patches by cosine similarity (DBSCAN or KMeans fallback).
4.  Per cluster, pick the single **best representative patch** (most central).
5.  For each cluster with ≥ 2 spatially-distinct members, produce an
    ObjectGroup with:
      - thumbnail_b64   : cropped image of the representative patch (base64 JPEG)
      - instance_count  : how many times this object/pattern was found
      - locations       : plain-English position descriptions per instance
                          ("top-left", "center", "bottom-right", …)
      - avg_similarity  : mean pairwise cosine similarity within the group (0–1)
      - color_rgb       : UI accent color for this group
6.  Draw a clean annotated image: ONE labelled box per instance (not dozens of
    overlapping rectangles).

All heavy work runs in a thread-pool executor.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Optional scikit-learn ─────────────────────────────────────────────────────
try:
    from sklearn.cluster import DBSCAN  # type: ignore[import]
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ── PyTorch / torchvision ─────────────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("[IntraSimilarity] torch/torchvision not available.")

# ── Config ────────────────────────────────────────────────────────────────────
_WINDOW_CONFIGS: List[Tuple[float, float]] = [
    (0.35, 0.18),   # large
    (0.22, 0.11),   # medium
    (0.13, 0.07),   # small
]
_MAX_PATCHES        = 100
_SIMILARITY_THRESH  = 0.82   # minimum cosine sim to be in the same group
_MIN_SPATIAL_SPREAD = 0.08   # instances must be > 8 % of short-side apart
_MIN_INSTANCES      = 2      # a group needs at least 2 spatially distinct instances
_MAX_GROUPS         = 8      # cap returned groups for readability

# Colour palette — one colour per group
_PALETTE: List[Tuple[int, int, int]] = [
    (99,  102, 241),  # indigo
    (34,  197,  94),  # green
    (249, 115,  22),  # orange
    (236,  72, 153),  # pink
    (20,  184, 166),  # teal
    (234, 179,   8),  # yellow
    (239,  68,  68),  # red
    (59,  130, 246),  # blue
]

# Model cache
_MODEL:     Optional[object] = None
_TRANSFORM: Optional[object] = None
_DEVICE:    str              = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Internal data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Patch:
    x: int; y: int; w: int; h: int
    embedding: List[float] = field(default_factory=list)
    cluster_id: int = -1


@dataclass
class ObjectGroup:
    """One cluster of visually similar regions — returned to the caller."""
    group_index:    int            # 0-based index (for colour mapping)
    instance_count: int            # number of times found
    locations:      List[str]      # e.g. ["top-left", "bottom-right"]
    avg_similarity: float          # mean pairwise cosine sim, 0–1
    thumbnail_b64:  str            # base64 JPEG of representative patch
    color_rgb:      Tuple[int, int, int]
    # raw bounding boxes for annotated image drawing (not exposed to API)
    _boxes:         List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class IntraSimilarityResult:
    image_width:  int
    image_height: int
    groups:       List[ObjectGroup]
    annotated_image_b64: str = ""
    summary: str = ""    # plain-English one-liner


# ─────────────────────────────────────────────────────────────────────────────
# Model init
# ─────────────────────────────────────────────────────────────────────────────

def _get_model():
    global _MODEL, _TRANSFORM, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TRANSFORM
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch/torchvision is required.")
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Identity()
    m.eval().to(_DEVICE)
    _MODEL = m
    _TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    logger.info(f"[IntraSimilarity] model loaded on {_DEVICE}")
    return _MODEL, _TRANSFORM


# ─────────────────────────────────────────────────────────────────────────────
# Patch extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_patches(img: Image.Image) -> List[_Patch]:
    W, H  = img.size
    short = min(W, H)
    patches: List[_Patch] = []
    for win_frac, stride_frac in _WINDOW_CONFIGS:
        wp = max(32, int(short * win_frac))
        sp = max(16, int(short * stride_frac))
        y  = 0
        while y + wp <= H:
            x = 0
            while x + wp <= W:
                patches.append(_Patch(x=x, y=y, w=wp, h=wp))
                x += sp
            y += sp
    if len(patches) > _MAX_PATCHES:
        idx = np.round(np.linspace(0, len(patches) - 1, _MAX_PATCHES)).astype(int)
        patches = [patches[i] for i in idx]
    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _position_label(cx: float, cy: float, W: int, H: int) -> str:
    """Convert normalised centre (0–1) to plain-English position."""
    col = "left" if cx < 0.38 else ("right" if cx > 0.62 else "center")
    row = "top"  if cy < 0.38 else ("bottom" if cy > 0.62 else "middle")
    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def _patch_thumbnail_b64(img: Image.Image, p: _Patch, size: int = 96) -> str:
    crop = img.crop((p.x, p.y, p.x + p.w, p.y + p.h)).resize((size, size), Image.LANCZOS)
    buf  = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _kmeans(matrix: np.ndarray, k: int) -> np.ndarray:
    rng       = np.random.default_rng(42)
    centroids = matrix[rng.choice(len(matrix), min(k, len(matrix)), replace=False)].copy()
    labels    = np.zeros(len(matrix), dtype=int)
    for _ in range(30):
        sims   = matrix @ centroids.T
        new_lb = sims.argmax(axis=1)
        if np.array_equal(new_lb, labels):
            break
        labels = new_lb
        for ci in range(k):
            m = matrix[labels == ci]
            if len(m):
                c = m.mean(0)
                n = np.linalg.norm(c)
                centroids[ci] = c / n if n else c
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_sync(image_bytes: bytes) -> IntraSimilarityResult:
    model, transform = _get_model()
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size

    patches = _extract_patches(img)
    if not patches:
        return IntraSimilarityResult(W, H, [], "", "No regions could be extracted.")

    # ── Batch embed ───────────────────────────────────────────────────────────
    BATCH  = 32
    embeds: List[np.ndarray] = []
    for s in range(0, len(patches), BATCH):
        batch = patches[s: s + BATCH]
        ts    = [transform(img.crop((p.x, p.y, p.x + p.w, p.y + p.h)))
                 for p in batch]
        t     = torch.stack(ts).to(_DEVICE)  # type: ignore[attr-defined]
        with torch.no_grad():
            f = model(t).cpu().numpy()       # type: ignore[operator]
        norms = np.linalg.norm(f, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeds.append(f / norms)

    M = np.vstack(embeds).astype(np.float32)   # (N, 576)
    for i, p in enumerate(patches):
        p.embedding = M[i].tolist()

    # ── Cluster ───────────────────────────────────────────────────────────────
    if _SKLEARN_AVAILABLE and len(patches) >= _MIN_INSTANCES:
        eps    = 1.0 - _SIMILARITY_THRESH
        labels = DBSCAN(eps=eps, min_samples=_MIN_INSTANCES,
                        metric="cosine").fit_predict(M)
    else:
        k      = max(2, int(len(patches) ** 0.5))
        labels = _kmeans(M, k)

    for i, p in enumerate(patches):
        p.cluster_id = int(labels[i])

    # ── Build groups ──────────────────────────────────────────────────────────
    short   = min(W, H)
    groups: List[ObjectGroup] = []
    seen_clusters = sorted(set(labels) - {-1})

    for cid in seen_clusters:
        members = [patches[i] for i, p in enumerate(patches) if p.cluster_id == cid]
        if len(members) < _MIN_INSTANCES:
            continue

        # Deduplicate spatially — keep only members that are far enough apart
        kept: List[_Patch] = []
        for p in members:
            cx_p = p.x + p.w / 2
            cy_p = p.y + p.h / 2
            too_close = any(
                ((cx_p - (k.x + k.w / 2)) ** 2 + (cy_p - (k.y + k.h / 2)) ** 2) ** 0.5
                < _MIN_SPATIAL_SPREAD * short
                for k in kept
            )
            if not too_close:
                kept.append(p)

        if len(kept) < _MIN_INSTANCES:
            continue

        # Representative patch = most central (closest to image centre)
        img_cx, img_cy = W / 2, H / 2
        rep = min(kept, key=lambda p: (
            (p.x + p.w / 2 - img_cx) ** 2 + (p.y + p.h / 2 - img_cy) ** 2
        ))

        # Average pairwise similarity within kept set
        idxs  = [i for i, p in enumerate(patches) if p in kept]
        sims: List[float] = []
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                sims.append(float(np.dot(M[idxs[ai]], M[idxs[bi]])))
        avg_sim = round(float(np.mean(sims)), 3) if sims else _SIMILARITY_THRESH

        # Positions
        locations = [
            _position_label((p.x + p.w / 2) / W, (p.y + p.h / 2) / H, W, H)
            for p in kept
        ]
        # Deduplicate position labels while preserving order
        seen_locs: List[str] = []
        for loc in locations:
            if loc not in seen_locs:
                seen_locs.append(loc)
        locations = seen_locs

        color_rgb = _PALETTE[len(groups) % len(_PALETTE)]
        thumbnail = _patch_thumbnail_b64(img, rep)
        boxes     = [(p.x, p.y, p.x + p.w, p.y + p.h) for p in kept]

        groups.append(ObjectGroup(
            group_index=len(groups),
            instance_count=len(kept),
            locations=locations,
            avg_similarity=avg_sim,
            thumbnail_b64=thumbnail,
            color_rgb=color_rgb,
            _boxes=boxes,
        ))

        if len(groups) >= _MAX_GROUPS:
            break

    # Sort by instance count descending, then by similarity
    groups.sort(key=lambda g: (g.instance_count, g.avg_similarity), reverse=True)
    for i, g in enumerate(groups):
        g.group_index = i

    # ── Annotated image ───────────────────────────────────────────────────────
    annotated_b64 = _draw_annotated(img, groups) if groups else ""

    # ── Plain-English summary ─────────────────────────────────────────────────
    if not groups:
        summary = "No repeated objects or patterns were found in this image."
    elif len(groups) == 1:
        g = groups[0]
        summary = (f"Found 1 type of repeated object — "
                   f"it appears {g.instance_count} times in the image.")
    else:
        total_instances = sum(g.instance_count for g in groups)
        summary = (f"Found {len(groups)} types of similar objects "
                   f"({total_instances} instances total) across the image.")

    return IntraSimilarityResult(
        image_width=W,
        image_height=H,
        groups=groups,
        annotated_image_b64=annotated_b64,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Annotation drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_annotated(img: Image.Image, groups: List[ObjectGroup]) -> str:
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return ""

    base    = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    for g in groups:
        r, gr, b = g.color_rgb
        label    = str(g.group_index + 1)

        for (x1, y1, x2, y2) in g._boxes:
            # Filled semi-transparent box
            draw.rectangle(
                [x1, y1, x2, y2],
                fill=(r, gr, b, 40),
                outline=(r, gr, b, 230),
                width=3,
            )
            # Badge circle in top-left corner
            bsz = 22
            bx, by = x1 + 4, y1 + 4
            draw.ellipse([bx, by, bx + bsz, by + bsz], fill=(r, gr, b, 240))
            draw.text(
                (bx + bsz // 2, by + bsz // 2),
                label,
                fill=(255, 255, 255, 255),
                font=font,
                anchor="mm",
            )

    combined = Image.alpha_composite(base, overlay).convert("RGB")
    buf      = io.BytesIO()
    combined.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────

class IntraSimilarityService:
    async def find_similar_objects(self, image_bytes: bytes) -> IntraSimilarityResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process_sync, image_bytes)
