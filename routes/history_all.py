from fastapi import APIRouter, HTTPException, Depends
from typing import List
from services.database import get_collection
from services.auth     import get_current_user
from models.user       import UserOut

router = APIRouter(tags=["History"])


@router.get("/history/all", response_model=List[dict])
async def get_all_history(current_user: UserOut = Depends(get_current_user)):
    """
    Returns the current user's history across all 4 operation types,
    merged and sorted newest-first (capped at 200 total records).

    Each record gains an ``operation_type`` field:
      - ``"remove_bg"``   — background removal
      - ``"enhance"``     — image enhancement
      - ``"replace_bg"``  — background replacement
      - ``"smart_crop"``  — smart crop
      - ``"recolor"``     — magic recolor
    """
    uid = current_user.user_id

    sources = [
        # (collection_name, operation_type, id_field, name_field)
        ("history",            "remove_bg",   "upload_id",  "original_name"),
        ("enhance_history",    "enhance",     "upload_id",  "original_name"),
        ("replace_bg_history", "replace_bg",  "result_id",  "fg_filename"),
        ("smart_crop_history", "smart_crop",  "upload_id",  "original_name"),
        ("recolor_history",    "recolor",     "upload_id",  "original_name"),
    ]

    records: list[dict] = []

    for col_name, op_type, id_field, name_field in sources:
        try:
            collection = get_collection(col_name)
            cursor = (
                collection
                .find({"user_id": uid}, {"_id": 0})
                .sort("created_at", -1)
                .limit(50)
            )
            docs = await cursor.to_list(length=50)
        except Exception:
            # If one collection is unavailable, skip it rather than failing the
            # whole request — history is best-effort.
            continue

        for doc in docs:
            # Normalise created_at to ISO-8601 string
            if "created_at" in doc:
                ts = doc["created_at"]
                doc["created_at"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

            # Expose a stable ``upload_id`` key regardless of source collection
            if id_field != "upload_id" and id_field in doc:
                doc.setdefault("upload_id", doc[id_field])

            # Expose a stable ``original_name`` key regardless of source collection
            if name_field != "original_name" and name_field in doc:
                doc.setdefault("original_name", doc[name_field])

            doc["operation_type"] = op_type
            records.append(doc)

    # Sort all merged records newest-first and cap at 200
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records[:200]
