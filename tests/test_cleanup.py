"""Tests for services/cleanup.py (the stale-file sweep)."""
from __future__ import annotations

import os
import time

import pytest

from services import cleanup


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run inside a temp folder that has uploads/ and output/, with a 24 h cutoff."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(cleanup, "FILE_MAX_AGE_HOURS", 24.0)
    return tmp_path


def make_file(path, age_hours: float):
    path.write_bytes(b"x")
    stamp = time.time() - age_hours * 3600
    os.utime(path, (stamp, stamp))
    return path


def test_deletes_old_files_and_keeps_fresh_ones(workdir):
    old_upload = make_file(workdir / "uploads" / "old.png", age_hours=30)
    old_output = make_file(workdir / "output" / "old_result.png", age_hours=48)
    fresh = make_file(workdir / "output" / "fresh_result.png", age_hours=1)

    deleted, skipped = cleanup._sweep_once()

    assert deleted == 2
    assert skipped == 1
    assert not old_upload.exists()
    assert not old_output.exists()
    assert fresh.exists()


def test_gitkeep_files_are_never_deleted(workdir):
    sentinel = make_file(workdir / "output" / ".gitkeep", age_hours=500)

    deleted, _ = cleanup._sweep_once()

    assert deleted == 0
    assert sentinel.exists()


def test_subfolders_are_ignored(workdir):
    folder = workdir / "output" / "nested"
    folder.mkdir()
    make_file(folder / "old.png", age_hours=100)

    deleted, skipped = cleanup._sweep_once()

    assert (deleted, skipped) == (0, 0)
    assert (folder / "old.png").exists()


def test_missing_folders_are_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no uploads/ or output/ here
    assert cleanup._sweep_once() == (0, 0)


def test_cutoff_follows_the_configured_age(workdir, monkeypatch):
    monkeypatch.setattr(cleanup, "FILE_MAX_AGE_HOURS", 1.0)
    two_hours_old = make_file(workdir / "output" / "a.png", age_hours=2)
    thirty_min_old = make_file(workdir / "output" / "b.png", age_hours=0.5)

    cleanup._sweep_once()

    assert not two_hours_old.exists()
    assert thirty_min_old.exists()


def test_run_sweep_now_returns_a_summary(workdir):
    make_file(workdir / "output" / "old.png", age_hours=30)
    make_file(workdir / "output" / "new.png", age_hours=1)

    assert cleanup.run_sweep_now() == {"deleted": 1, "skipped": 1, "max_age_hours": 24.0}