"""Health check for hosting platforms and uptime monitors."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.database import is_db_connected

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    """
    200 when the database connection from startup is up, 503 when it is not.
    (Reflects the startup ping; it does not re-check MongoDB on every call.)
    """
    connected = is_db_connected()
    return JSONResponse(
        status_code=200 if connected else 503,
        content={
            "status":   "ok" if connected else "degraded",
            "database": "connected" if connected else "disconnected",
        },
    )