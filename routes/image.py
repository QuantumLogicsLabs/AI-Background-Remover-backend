import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from models.ai import (
    ImageAnalysisResponse,
    CaptionResponse,
    CaptionsResponse,
    BackgroundSuggestionsResponse,
    AdvancedAnalysisResponse,
    BatchAdvancedAnalysisResponse,
    BatchAdvancedAnalysisItem
)
from services.ai_service import AIService
from services.image_service import ImageService
from services.upload_validation import read_upload_limited
from services.auth import get_current_user
from models.user import UserOut

router = APIRouter(prefix="/image", tags=["image"])
ai_service = AIService()
image_service = ImageService()

MAX_SIZE_MB = 10


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user)
):
    contents = await read_upload_limited(file, MAX_SIZE_MB)
    if not image_service.validate(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
        )

    preprocessed_bytes = image_service.preprocess(contents)

    try:
        analysis = await ai_service.analyze_image(preprocessed_bytes)
        return ImageAnalysisResponse(**analysis)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(
    file: UploadFile = File(...),
    style: str = Form("casual"),
    current_user: UserOut = Depends(get_current_user)
):
    contents = await read_upload_limited(file, MAX_SIZE_MB)
    if not image_service.validate(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
        )

    preprocessed_bytes = image_service.preprocess(contents)

    try:
        caption = await ai_service.generate_caption(preprocessed_bytes, style)
        return CaptionResponse(caption=caption, style=style)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/captions", response_model=CaptionsResponse)
async def generate_captions(
    file: UploadFile = File(...),
    style: str = Form("casual"),
    current_user: UserOut = Depends(get_current_user)
):
    contents = await read_upload_limited(file, MAX_SIZE_MB)
    if not image_service.validate(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
        )

    preprocessed_bytes = image_service.preprocess(contents)

    try:
        captions = await ai_service.generate_captions(preprocessed_bytes, style)
        return CaptionsResponse(captions=captions, style=style)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/suggestions", response_model=BackgroundSuggestionsResponse)
async def background_suggestions(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user)
):
    contents = await read_upload_limited(file, MAX_SIZE_MB)
    if not image_service.validate(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
        )

    preprocessed_bytes = image_service.preprocess(contents)

    try:
        suggestions = await ai_service.suggest_backgrounds(preprocessed_bytes)
        return BackgroundSuggestionsResponse(suggestions=suggestions)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/analyze-advanced", response_model=AdvancedAnalysisResponse)
async def analyze_image_advanced(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user)
):
    contents = await read_upload_limited(file, MAX_SIZE_MB)
    if not image_service.validate(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
        )

    preprocessed_bytes = image_service.preprocess(contents)

    try:
        analysis = await ai_service.analyze_image_advanced(preprocessed_bytes)
        return AdvancedAnalysisResponse(**analysis)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/analyze-advanced-batch", response_model=BatchAdvancedAnalysisResponse)
async def analyze_image_advanced_batch(
    files: list[UploadFile] = File(...),
    current_user: UserOut = Depends(get_current_user)
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided."
        )
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files per batch analysis."
        )

    async def process_single_file(upload: UploadFile) -> BatchAdvancedAnalysisItem:
        try:
            contents = await read_upload_limited(upload, MAX_SIZE_MB)
            if not image_service.validate(contents):
                return BatchAdvancedAnalysisItem(
                    filename=upload.filename or "unknown",
                    status="error",
                    error="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
                )
            preprocessed = image_service.preprocess(contents)
            analysis = await ai_service.analyze_image_advanced(preprocessed)
            return BatchAdvancedAnalysisItem(
                filename=upload.filename or "unknown",
                status="success",
                analysis=AdvancedAnalysisResponse(**analysis)
            )
        except HTTPException as exc:
            # e.g. a file over the size limit: report it for this file only
            return BatchAdvancedAnalysisItem(
                filename=upload.filename or "unknown",
                status="error",
                error=str(exc.detail)
            )
        except Exception as e:
            return BatchAdvancedAnalysisItem(
                filename=upload.filename or "unknown",
                status="error",
                error=str(e)
            )

    results = await asyncio.gather(*(process_single_file(f) for f in files))
    return BatchAdvancedAnalysisResponse(results=list(results))