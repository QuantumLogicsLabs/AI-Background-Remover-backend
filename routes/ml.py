"""
ML Features Router
===================
Prefix: /api/ml

Endpoints
---------
POST /api/ml/similarity/index        — embed & store an image for future searches
POST /api/ml/similarity/search       — find similar images to a query upload
GET  /api/ml/similarity/count        — how many images the user has indexed

POST /api/ml/categorize              — classify an image into a category

POST /api/ml/duplicates/index        — hash & store an image for duplicate checks
POST /api/ml/duplicates/check        — check if an upload has duplicates
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from models.ml import (
    CategorizationResponse,
    DuplicateCheckResponse,
    DuplicateMatch,
    IndexForDuplicatesResponse,
    IndexImageResponse,
    SimilarityResult,
    SimilaritySearchResponse,
)
from models.user import UserOut
from services.auth import get_current_user
from services.categorization_service import CategorizationService
from services.database import get_collection
from services.duplicate_detection_service import DuplicateDetectionService
from services.image_service import ImageService
from services.similarity_service import SimilarityService

router = APIRouter(prefix="/ml", tags=["ML Features"])

# ── Service singletons ────────────────────────────────────────────────────────
similarity_svc = SimilarityService()
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
