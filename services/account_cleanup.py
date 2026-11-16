"""Remove everything a user owns when their account is deleted."""

from __future__ import annotations

import logging
import os

from services.database import get_collection

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"

# Collections whose documents carry a ``user_id`` field and belong to one user.
# Shared-workspace collections (chat_messages, image_comments) are left alone on purpose.
USER_COLLECTIONS: tuple[str, ...] = (
    "history",
    "enhance_history",
    "replace_bg_history",
    "smart_crop_history",
    "recolor_history",
    "inpaint_history",
    "ml_similarity_history",
    "ml_duplicate_history",
    "conversations",
    "favorites",
    "action_history",
    "usage_events",
    "cost_logs",
    "feedback",
    "batch_jobs",
    "email_verifications",
    "password_reset_tokens",
)

# Fields that name a processed file in the output folder, per collection.
FILE_FIELDS: dict[str, tuple[str, ...]] = {
    "history":            ("output_filename",),
    "enhance_history":    ("output_filename",),
    "replace_bg_history": ("output_filename",),
    "smart_crop_history": ("removed_filename", "cropped_filename"),
    "recolor_history":    ("output_filename",),
    "inpaint_history":    ("output_filename",),
}


def _remove_output_file(filename: object) -> int:
    """Delete one file from the output folder. Returns 1 if removed, else 0."""
    if not isinstance(filename, str) or not filename:
        return 0
    # basename() stops a stored value like "../x" from escaping the output folder
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    try:
        os.remove(path)
        return 1
    except FileNotFoundError:
        return 0
    except OSError as exc:
        logger.warning("[Account] Could not delete %s: %s", path, exc)
        return 0


async def purge_user_data(user_id: str) -> dict[str, int]:
    """
    Delete the user's documents (and their output files) from every personal
    collection. A failure in one collection is logged and never stops the rest.

    Returns {"documents": <deleted documents>, "files": <deleted files>}.
    """
    documents = 0
    files = 0

    for name in USER_COLLECTIONS:
        try:
            collection = get_collection(name)

            fields = FILE_FIELDS.get(name, ())
            if fields:
                projection = {"_id": 0, **{field: 1 for field in fields}}
                docs = await collection.find({"user_id": user_id}, projection).to_list(length=None)
                for doc in docs:
                    for field in fields:
                        files += _remove_output_file(doc.get(field))

            result = await collection.delete_many({"user_id": user_id})
            documents += result.deleted_count
        except Exception as exc:
            logger.warning("[Account] Could not purge %s for user %s: %s", name, user_id, exc)

    return {"documents": documents, "files": files}