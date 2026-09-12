"""
ML Features Router
===================
Prefix: /api/ml

Endpoints
---------
POST /api/ml/similarity/index        — embed & store an image for future searches
POST /api/ml/similarity/search       — find similar images to a query upload
GET  /api/ml/similarity/count        — how many images the user has indexed
POST /api/ml/similarity/objects      — find similar objects within a SINGLE image

POST /api/ml/categorize              — classify an image into a category

POST /api/ml/duplicates/index        — hash & store an image for duplicate checks
POST /api/ml/duplicates/check        — check if an upload has duplicates
GET  /api/ml/duplicates/scan-history — scan all history images and report duplicates
"""

from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import httpx

from models.ml import (
    CategorizationResponse,
    DetectedObjectGroup,
    DuplicateCheckResponse,
    DuplicateMatch,
    HistoryDuplicateItem,
    HistoryScanResponse,
    IndexForDuplicatesResponse,
    IndexImageResponse,
    SimilarityResult,
    SimilaritySearchResponse,
    SimilarObjectsResponse,
)
from models.user import UserOut
from services.auth import get_current_user
from services.categorization_service import CategorizationService
from services.database import get_collection
from services.duplicate_detection_service import DuplicateDetectionService
from services.image_service import ImageService
from services.intra_similarity_service import IntraSimilarityService
from services.similarity_service import SimilarityService

router = APIRouter(prefix="/ml", tags=["ML Features"])

# ── Service singletons ────────────────────────────────────────────────────────
similarity_svc = SimilarityService()
intra_similarity_svc = IntraSimilarityService()
categorization_svc = CategorizationService()
duplicate_svc = DuplicateDetectionService()
image_svc = ImageService()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Use JPEG, PNG, or WebP.",
        )
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_SIZE_MB} MB limit.",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Similarity Search
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/similarity/index", response_model=IndexImageResponse)
async def index_image_for_similarity(
    file: UploadFile = File(...),
    image_id: str = Form(""),
    download_url: str = Form(""),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Generate and store an embedding for an image so it can appear in
    future similarity searches. Call this after uploading/processing an image.
    """
    contents = await file.read()
    _validate_upload(file, contents)

    safe_name = file.filename or "upload"
    if not image_id:
        image_id = str(uuid.uuid4())

    try:
        embedding = await similarity_svc.generate_embedding(contents)
        await similarity_svc.store_embedding(
            user_id=current_user.user_id,
            image_id=image_id,
            filename=safe_name,
            embedding=embedding,
            download_url=download_url,
        )
        total = await similarity_svc.get_embedding_count(current_user.user_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        )

    # Persist to history
    try:
        col = get_collection("ml_similarity_history")
        await col.insert_one({
            "user_id": current_user.user_id,
            "image_id": image_id,
            "filename": safe_name,
            "action": "index",
            "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return IndexImageResponse(
        image_id=image_id,
        filename=safe_name,
        total_indexed=total,
        message=f"Image indexed. You now have {total} image(s) in your similarity index.",
    )


@router.post("/similarity/search", response_model=SimilaritySearchResponse)
async def search_similar_images(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Upload an image and retrieve the most visually similar images from the
    user's indexed library. Returns up to top_k results sorted by similarity.
    """
    contents = await file.read()
    _validate_upload(file, contents)

    top_k = max(1, min(top_k, 50))
    query_image_id = str(uuid.uuid4())

    try:
        query_embedding = await similarity_svc.generate_embedding(contents)
        raw_results = await similarity_svc.find_similar(
            query_embedding=query_embedding,
            user_id=current_user.user_id,
            top_k=top_k,
        )
        total = await similarity_svc.get_embedding_count(current_user.user_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {exc}",
        )

    results = [
        SimilarityResult(
            image_id=r["image_id"],
            filename=r["filename"],
            score=r["score"],
            download_url=r.get("download_url", ""),
        )
        for r in raw_results
    ]

    return SimilaritySearchResponse(
        query_image_id=query_image_id,
        total_indexed=total,
        results=results,
    )


@router.get("/similarity/count")
async def get_similarity_index_count(
    current_user: UserOut = Depends(get_current_user),
):
    """Return how many images the authenticated user has indexed."""
    total = await similarity_svc.get_embedding_count(current_user.user_id)
    return {"user_id": current_user.user_id, "total_indexed": total}


@router.post("/similarity/objects", response_model=SimilarObjectsResponse)
async def find_similar_objects_in_image(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Upload a **single** image and detect visually similar objects / repeated
    regions **within that image**.

    Returns plain-English object groups — each group has a thumbnail of the
    detected object, how many times it appears, where it appears (e.g.
    "top-left", "bottom-right"), and a similarity score.

    No indexing or database storage required — fully stateless analysis.
    """
    contents = await file.read()
    _validate_upload(file, contents)

    try:
        result = await intra_similarity_svc.find_similar_objects(contents)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar object detection failed: {exc}",
        )

    groups = [
        DetectedObjectGroup(
            group_index=g.group_index,
            instance_count=g.instance_count,
            locations=g.locations,
            avg_similarity=g.avg_similarity,
            thumbnail_b64=g.thumbnail_b64,
            color_rgb=list(g.color_rgb),
        )
        for g in result.groups
    ]

    return SimilarObjectsResponse(
        image_width=result.image_width,
        image_height=result.image_height,
        groups=groups,
        annotated_image_b64=result.annotated_image_b64,
        summary=result.summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Image Categorization
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/categorize", response_model=CategorizationResponse)
async def categorize_image(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Classify an uploaded image into one of the predefined categories:
    Portrait · Product · Food · Animal · Landscape · Document · Vehicle · Other.

    Uses CLIP zero-shot classification when available, otherwise falls back
    to ImageNet keyword matching.
    """
    contents = await file.read()
    _validate_upload(file, contents)

    safe_name = file.filename or "upload"
    image_id = str(uuid.uuid4())

    try:
        result = await categorization_svc.categorize(contents)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Categorization failed: {exc}",
        )

    # Persist history
    try:
        await categorization_svc.save_result(
            user_id=current_user.user_id,
            image_id=image_id,
            filename=safe_name,
            result=result,
        )
    except Exception:
        pass

    return CategorizationResponse(
        image_id=image_id,
        filename=safe_name,
        category=result["category"],
        confidence=result["confidence"],
        scores=result["scores"],
        method=result.get("method", "unknown"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Duplicate Detection
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/duplicates/index", response_model=IndexForDuplicatesResponse)
async def index_image_for_duplicates(
    file: UploadFile = File(...),
    image_id: str = Form(""),
    download_url: str = Form(""),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Compute and store MD5 + pHash for an image so it participates in future
    duplicate checks. Call this after uploading/processing an image.
    """
    contents = await file.read()
    _validate_upload(file, contents)

    safe_name = file.filename or "upload"
    if not image_id:
        image_id = str(uuid.uuid4())

    try:
        hashes = await duplicate_svc.extract_hashes(contents)
        await duplicate_svc.store_hashes(
            user_id=current_user.user_id,
            image_id=image_id,
            filename=safe_name,
            hashes=hashes,
            download_url=download_url,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hash extraction failed: {exc}",
        )

    return IndexForDuplicatesResponse(
        image_id=image_id,
        filename=safe_name,
        md5=hashes["md5"],
        message="Image indexed for duplicate detection.",
    )


@router.post("/duplicates/check", response_model=DuplicateCheckResponse)
async def check_for_duplicates(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Upload an image and check whether it duplicates any previously indexed
    image in the user's library.

    Returns duplicates grouped by type:
      • exact    — byte-identical (same MD5)
      • resized  — same content, different dimensions (pHash distance ≤ 4)
      • cropped  — looks like a crop of an existing image (distance ≤ 10)
      • similar  — slight modifications (brightness / filter / compression)
    """
    contents = await file.read()
    _validate_upload(file, contents)

    query_image_id = str(uuid.uuid4())

    try:
        hashes = await duplicate_svc.extract_hashes(contents)
        duplicates = await duplicate_svc.find_duplicates(
            query_hashes=hashes,
            user_id=current_user.user_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Duplicate check failed: {exc}",
        )

    dup_models = [
        DuplicateMatch(
            image_id=d["image_id"],
            filename=d["filename"],
            duplicate_type=d["duplicate_type"],
            hamming_distance=d["hamming_distance"],
            download_url=d.get("download_url", ""),
        )
        for d in duplicates
    ]

    # Log to history
    try:
        col = get_collection("ml_duplicate_history")
        await col.insert_one({
            "user_id": current_user.user_id,
            "query_image_id": query_image_id,
            "has_duplicates": len(duplicates) > 0,
            "total_found": len(duplicates),
            "md5": hashes["md5"],
            "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return DuplicateCheckResponse(
        query_image_id=query_image_id,
        has_duplicates=len(duplicates) > 0,
        total_found=len(duplicates),
        duplicates=dup_models,
        query_md5=hashes["md5"],
        query_phash=hashes["phash"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Duplicate Detection — History Auto-Scan
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/duplicates/scan-history", response_model=HistoryScanResponse)
async def scan_history_for_duplicates(
    current_user: UserOut = Depends(get_current_user),
):
    """
    Automatically scan all images in the user's processing history and report
    which ones are duplicates of each other — no upload required.

    Algorithm:
      1. Fetch every record from the user's history (all operation types).
      2. For each record, download the output image and compute MD5 + pHash.
      3. Compare every image against every other (O(n²) with early-exit on MD5).
      4. Union-Find groups connected duplicates together into numbered groups.
      5. Return every history image annotated with is_duplicate / group_id.
    """
    uid = current_user.user_id

    # ── 1. Gather history records ──────────────────────────────────────────────
    sources = [
        ("history",            "remove_bg",  "upload_id",  "original_name"),
        ("enhance_history",    "enhance",    "upload_id",  "original_name"),
        ("replace_bg_history", "replace_bg", "result_id",  "fg_filename"),
        ("smart_crop_history", "smart_crop", "upload_id",  "original_name"),
        ("recolor_history",    "recolor",    "upload_id",  "original_name"),
    ]

    raw_records: list[dict] = []
    for col_name, op_type, id_field, name_field in sources:
        try:
            collection = get_collection(col_name)
            cursor = (
                collection.find({"user_id": uid}, {"_id": 0})
                .sort("created_at", -1)
                .limit(50)
            )
            docs = await cursor.to_list(length=50)
        except Exception:
            continue

        for doc in docs:
            if id_field != "upload_id" and id_field in doc:
                doc.setdefault("upload_id", doc[id_field])
            if name_field != "original_name" and name_field in doc:
                doc.setdefault("original_name", doc[name_field])
            if "created_at" in doc:
                ts = doc["created_at"]
                doc["created_at"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            doc["operation_type"] = op_type
            raw_records.append(doc)

    # Sort newest-first, cap at 200
    raw_records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    raw_records = raw_records[:200]

    if not raw_records:
        return HistoryScanResponse(
            total_scanned=0,
            total_duplicates=0,
            total_unique=0,
            groups_found=0,
            items=[],
        )

    # ── 2. Download each image and extract hashes ──────────────────────────────
    # Build the base URL for internal requests (same host/port as this service).
    # We reuse the output filenames which are served at /api/download/<filename>.
    async def _fetch_and_hash(record: dict) -> dict | None:
        """Return the record enriched with hash info, or None on failure."""
        filename = record.get("output_filename") or record.get("filename", "")
        if not filename:
            return None
        download_path = f"/api/download/{filename}"
        try:
            # Use an internal httpx call to the same server
            async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
                resp = await client.get(download_path)
            if resp.status_code != 200:
                return None
            image_bytes = resp.content
            hashes = await duplicate_svc.extract_hashes(image_bytes)
            return {**record, "_hashes": hashes}
        except Exception:
            return None

    # Run all fetches concurrently (bounded to 10 at a time to avoid overload)
    semaphore = asyncio.Semaphore(10)

    async def _bounded_fetch(record: dict) -> dict | None:
        async with semaphore:
            return await _fetch_and_hash(record)

    results = await asyncio.gather(*[_bounded_fetch(r) for r in raw_records])
    enriched = [r for r in results if r is not None]

    if not enriched:
        return HistoryScanResponse(
            total_scanned=len(raw_records),
            total_duplicates=0,
            total_unique=0,
            groups_found=0,
            items=[
                HistoryDuplicateItem(
                    image_id=r.get("upload_id", ""),
                    filename=r.get("original_name", r.get("output_filename", "")),
                    download_url=f"/api/download/{r.get('output_filename', '')}",
                    operation_type=r.get("operation_type", ""),
                    created_at=r.get("created_at", ""),
                    is_duplicate=False,
                    duplicate_type=None,
                    hamming_distance=None,
                    group_id=None,
                )
                for r in raw_records
            ],
        )

    # ── 3. Pairwise comparison with Union-Find ────────────────────────────────
    n = len(enriched)
    parent = list(range(n))          # Union-Find parent array
    best_edge: list[dict | None] = [None] * n  # best duplicate info per node

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        hi = enriched[i]["_hashes"]
        for j in range(i + 1, n):
            hj = enriched[j]["_hashes"]

            # Fast-path: exact MD5 match
            if hi["md5"] == hj["md5"]:
                dup_type = "exact"
                distance = 0
            else:
                from services.duplicate_detection_service import (
                    _hamming_distance,
                    _classify_duplicate,
                    DUP_UNIQUE,
                )
                distance = _hamming_distance(hi["phash"], hj["phash"])
                dup_type = _classify_duplicate(
                    distance,
                    (hi["width"], hi["height"]),
                    (hj["width"], hj["height"]),
                )
                if dup_type == DUP_UNIQUE:
                    continue

            _union(i, j)

            # Keep the strongest (lowest distance) duplicate info for each node
            for idx, other_idx in [(i, j), (j, i)]:
                existing = best_edge[idx]
                if existing is None or distance < existing["hamming_distance"]:
                    best_edge[idx] = {
                        "duplicate_type": dup_type,
                        "hamming_distance": distance,
                        "matched_with": enriched[other_idx].get("upload_id", ""),
                    }

    # ── 4. Assign group IDs ───────────────────────────────────────────────────
    # Only roots that have more than one member in their component get a group.
    from collections import defaultdict
    component_members: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        component_members[_find(idx)].append(idx)

    # Roots with ≥2 members are real duplicate groups; assign sequential IDs.
    group_id_map: dict[int, int] = {}
    next_gid = 1
    for root, members in component_members.items():
        if len(members) >= 2:
            group_id_map[root] = next_gid
            next_gid += 1

    # ── 5. Build response items ───────────────────────────────────────────────
    items: list[HistoryDuplicateItem] = []
    for idx, record in enumerate(enriched):
        root = _find(idx)
        gid = group_id_map.get(root)
        edge = best_edge[idx] if gid is not None else None

        items.append(HistoryDuplicateItem(
            image_id=record.get("upload_id", ""),
            filename=record.get("original_name", record.get("output_filename", "")),
            download_url=f"/api/download/{record.get('output_filename', '')}",
            operation_type=record.get("operation_type", ""),
            created_at=record.get("created_at", ""),
            is_duplicate=gid is not None,
            duplicate_type=edge["duplicate_type"] if edge else None,
            hamming_distance=edge["hamming_distance"] if edge else None,
            group_id=gid,
        ))

    # Append any records that failed to download as "unique / unknown"
    fetched_ids = {r.get("upload_id") for r in enriched}
    for record in raw_records:
        if record.get("upload_id") not in fetched_ids:
            items.append(HistoryDuplicateItem(
                image_id=record.get("upload_id", ""),
                filename=record.get("original_name", record.get("output_filename", "")),
                download_url=f"/api/download/{record.get('output_filename', '')}",
                operation_type=record.get("operation_type", ""),
                created_at=record.get("created_at", ""),
                is_duplicate=False,
                duplicate_type=None,
                hamming_distance=None,
                group_id=None,
            ))

    total_duplicates = sum(1 for it in items if it.is_duplicate)

    # Log scan to history
    try:
        col = get_collection("ml_duplicate_history")
        await col.insert_one({
            "user_id": uid,
            "scan_type": "history_auto_scan",
            "total_scanned": len(items),
            "total_duplicates": total_duplicates,
            "groups_found": next_gid - 1,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return HistoryScanResponse(
        total_scanned=len(items),
        total_duplicates=total_duplicates,
        total_unique=len(items) - total_duplicates,
        groups_found=next_gid - 1,
        items=items,
    )
