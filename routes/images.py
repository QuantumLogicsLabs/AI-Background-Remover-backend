"""
images.py — DELETE /api/image/{image_id}

Ownership note
──────────────
This file (images.py) owns the DELETE /api/image/{id} endpoint only.
The POST /api/image/analyze, /api/image/caption, /api/image/captions, and
/api/image/suggestions endpoints live in routes/image.py (singular) and are
unrelated despite the similar module name. Both are registered under the same
/api prefix in app.py but their URL paths never overlap.
"""

from fastapi import APIRouter, HTTPException, Depends
from services.database import get_collection
from services.auth     import get_current_user
from models.user       import UserOut
import os

router = APIRouter(tags=["Images"])

OUTPUT_DIR = "output"

# ── Collection map ─────────────────────────────────────────────────────────
# Each entry:  collection_name  →  (id_field, [output filenames template(s)])
# Templates use {id} as placeholder for the record's primary identifier.
#
# "history"            remove_bg  :  {id}_result.png
# "enhance_history"    enhance    :  {id}_enhanced.png
# "replace_bg_history" replace_bg :  {result_id}_composited.png   (id_field = result_id)
# "smart_crop_history" smart_crop :  {id}_removed.png  +  {id}_cropped.png

_COLLECTION_MAP: list[tuple[str, str, list[str]]] = [
    # (collection,           id_field,    [filename templates])
    ("history",            "upload_id",  ["{id}_result.png"]),
    ("enhance_history",    "upload_id",  ["{id}_enhanced.png"]),
    ("replace_bg_history", "result_id",  ["{id}_composited.png"]),
    ("smart_crop_history", "upload_id",  ["{id}_removed.png", "{id}_cropped.png"]),
]


@router.delete("/image/{image_id}")
async def delete_image(
    image_id:     str,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Delete a processed image record (and its output file(s) from disk) by ID.

    The endpoint searches all four history collections — history (remove-bg),
    enhance_history, replace_bg_history, and smart_crop_history — so it works
    regardless of which operation produced the image. For smart_crop records
    both the _removed.png and _cropped.png files are cleaned up.

    Returns 403 if the record belongs to a different user.
    Returns 200 even if no DB record is found (idempotent delete).
    """
    safe_id = os.path.basename(image_id)
    uid     = current_user.user_id

    deleted_any = False

    for col_name, id_field, filename_templates in _COLLECTION_MAP:
        try:
            collection = get_collection(col_name)
            record     = await collection.find_one({id_field: safe_id})
        except Exception:
            record = None

        if record is None:
            continue

        # Authorisation: record must belong to the requesting user
        if record.get("user_id") != uid:
            raise HTTPException(
                status_code=403,
                detail="Not authorised to delete this image.",
            )

        # Delete every output file associated with this record
        for template in filename_templates:
            filename  = template.replace("{id}", safe_id)
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # already gone or permission issue — non-fatal

        # Remove the DB record
        try:
            await collection.delete_one({id_field: safe_id, "user_id": uid})
        except Exception:
            pass

        deleted_any = True
        # A given ID can only live in one collection, so stop after first match
        break

    return {"message": f"Image {safe_id} deleted successfully.", "found": deleted_any}
