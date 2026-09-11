import os
import sys
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from services.database   import connect_db, close_db
from services.cleanup    import start_cleanup_task, stop_cleanup_task
from services.job_queue  import job_queue
from services.bg_removal import warm_up
from routes.auth        import router as auth_router
from routes.remove_bg   import router as remove_bg_router
from routes.download    import router as download_router
from routes.history     import router as history_router
from routes.history_all import router as history_all_router
from routes.images      import router as images_router
from routes.enhance     import router as enhance_router
from routes.replace_bg  import router as replace_bg_router
from routes.smart_crop  import router as smart_crop_router
from routes.batch       import router as batch_router
from routes.chat        import router as chat_router
from routes.image       import router as image_router
from routes.recolor     import router as recolor_router
from routes.stats       import router as stats_router
from routes.action_history import router as action_history_router
from routes.collab         import router as collab_router
from routes.analytics      import router as analytics_router
from routes.favorites       import router as favorites_router
from routes.inpaint        import router as inpaint_router
from routes.vectorize      import router as vectorize_router
from routes.ml             import router as ml_router

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    start_cleanup_task()
    await job_queue.start()
    # Pre-load AI model sessions so the first request isn't slow.
    # Non-fatal: auth and other routes still work if warm-up fails.
    try:
        await warm_up()
    except BaseException as exc:
        print(f"[AI] Model warm-up skipped/failed (will initialize on first request): {exc}")
    yield
    await job_queue.stop()
    stop_cleanup_task()
    await close_db()


app = FastAPI(
    title="AI Background Remover API",
    description="REST API for AI-powered background removal with auth and cloud storage.",
    version="2.0.0",
    lifespan=lifespan,
)

_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router,       prefix="/api")
app.include_router(remove_bg_router,  prefix="/api")
app.include_router(download_router,   prefix="/api")
app.include_router(history_router,     prefix="/api")
app.include_router(history_all_router, prefix="/api")
app.include_router(images_router,     prefix="/api")
app.include_router(enhance_router,    prefix="/api")
app.include_router(replace_bg_router, prefix="/api")
app.include_router(smart_crop_router, prefix="/api")
app.include_router(batch_router,      prefix="/api")
app.include_router(chat_router,       prefix="/api")
app.include_router(image_router,      prefix="/api")
app.include_router(recolor_router,    prefix="/api")
app.include_router(stats_router,       prefix="/api")
app.include_router(action_history_router, prefix="/api")
app.include_router(collab_router,         prefix="/api")
app.include_router(analytics_router,      prefix="/api")
app.include_router(favorites_router,      prefix="/api")
app.include_router(inpaint_router,        prefix="/api")
app.include_router(vectorize_router,      prefix="/api")
app.include_router(ml_router,             prefix="/api")


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "AI Background Remover API v2 is running."}

