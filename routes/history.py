from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
from services.database import get_collection
from services.auth     import get_current_user
from models.user       import UserOut

router = APIRouter(tags=["History"])


@router.get(
    "/history",
    response_model=List[dict],
    deprecated=True,
    summary="[Deprecated] Get remove-bg history",
    description=(
        "**Deprecated** — use `GET /api/history/all` instead, which returns history "
        "across all four operation types (remove_bg, enhance, replace_bg, smart_crop).\n\n"
        "This endpoint returns only the `remove_bg` collection and is kept for "
        "backward compatibility only. It will be removed in a future version."
    ),
)
async def get_history(current_user: UserOut = Depends(get_current_user)):
    """[Deprecated] Returns only remove-bg history. Use /api/history/all for full history."""
    try:
        collection = get_collection("history")
        cursor  = (
            collection
            .find({"user_id": current_user.user_id}, {"_id": 0})
            .sort("created_at", -1)
            .limit(50)
        )
        results = await cursor.to_list(length=50)
        for record in results:
            if "created_at" in record:
                record["created_at"] = record["created_at"].isoformat()
        return JSONResponse(
            content=results,
            headers={
                "Deprecation": "true",
                "Link": '</api/history/all>; rel="successor-version"',
                "Warning": '299 - "This endpoint is deprecated. Use /api/history/all instead."',
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch history: {exc}")
