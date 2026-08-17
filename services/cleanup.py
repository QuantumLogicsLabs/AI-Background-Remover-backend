"""
File Cleanup Service.

Periodically removes stale files from the uploads/ and output/ directories
so disk space doesn't grow unbounded.

Design
──────
- A single asyncio background task runs in a continuous loop, sleeping
  between sweeps. It is started at application startup and cancelled on
  shutdown via the FastAPI lifespan context manager.
- Files are deleted when their last-modified time is older than
  FILE_MAX_AGE_HOURS (default 24 hours, configurable via env var).
- Sentinel .gitkeep files (zero-byte files named ".gitkeep") are never
  deleted so the directories stay tracked in git.
- Errors on individual files are caught and logged so one bad file
  never interrupts the full sweep.

Configuration (env vars)
────────────────────────
  FILE_MAX_AGE_HOURS    — age threshold in hours before a file is deleted
                          (default: 24)
  CLEANUP_INTERVAL_MINS — how often the sweep runs, in minutes
                          (default: 60)
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────

FILE_MAX_AGE_HOURS:    float = float(os.getenv("FILE_MAX_AGE_HOURS",    "24"))
CLEANUP_INTERVAL_MINS: float = float(os.getenv("CLEANUP_INTERVAL_MINS", "60"))

_WATCHED_DIRS = ("uploads", "output")

# ── Internal ───────────────────────────────────────────────────────────────

_cleanup_task: asyncio.Task | None = None


def _sweep_once() -> tuple[int, int]:
    """
    Synchronous sweep of uploads/ and output/.

    Returns:
        (deleted_count, skipped_count) — counts for this run.
    """
    cutoff  = time.time() - FILE_MAX_AGE_HOURS * 3600
    deleted = 0
    skipped = 0

    for dir_name in _WATCHED_DIRS:
        dir_path = Path(dir_name)
        if not dir_path.is_dir():
            continue

        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue
            # Never remove git sentinel files
            if file_path.name == ".gitkeep":
                continue

            try:
                if file_path.stat().st_mtime < cutoff:
                    file_path.unlink()
                    deleted += 1
                else:
                    skipped += 1
            except FileNotFoundError:
                pass  # already gone — race with download, ignore
            except OSError as exc:
                print(f"[CLEANUP] Could not delete {file_path}: {exc}")
                skipped += 1

    return deleted, skipped


async def _cleanup_loop() -> None:
    """Continuous sweep loop — runs until cancelled."""
    interval_secs = CLEANUP_INTERVAL_MINS * 60
    print(
        f"[CLEANUP] Task started: sweep every {CLEANUP_INTERVAL_MINS:.0f} min, "
        f"delete files older than {FILE_MAX_AGE_HOURS:.0f} h."
    )
    while True:
        try:
            await asyncio.sleep(interval_secs)
            deleted, skipped = _sweep_once()
            if deleted:
                print(f"[CLEANUP] Deleted {deleted} file(s), kept {skipped}.")
        except asyncio.CancelledError:
            print("[CLEANUP] Task stopped.")
            break
        except Exception as exc:
            # Never let an unexpected error kill the background task
            print(f"[CLEANUP] Sweep error: {exc}")


# ── Public API ──────────────────────────────────────────────────────────────

def start_cleanup_task() -> None:
    """
    Start the background cleanup loop.
    Call once from the FastAPI lifespan startup block.
    """
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop())


def stop_cleanup_task() -> None:
    """
    Cancel the background cleanup loop.
    Call from the FastAPI lifespan shutdown block.
    """
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        _cleanup_task = None


def run_sweep_now() -> dict:
    """
    Trigger an immediate synchronous sweep and return a summary.
    Useful for an admin endpoint or manual testing.

    Returns:
        {"deleted": int, "skipped": int, "max_age_hours": float}
    """
    deleted, skipped = _sweep_once()
    return {
        "deleted":       deleted,
        "skipped":       skipped,
        "max_age_hours": FILE_MAX_AGE_HOURS,
    }
