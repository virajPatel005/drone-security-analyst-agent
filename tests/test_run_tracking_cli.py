"""No-model tests for the demo CLI: --help and the duplicate-video gate."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tracking  # noqa: E402
from src.storage import init_db, store_run  # noqa: E402


def _db_with_run(tmp_path: Path, video: Path) -> tuple[Path, int]:
    """Create a temp DB with one run whose source_video matches *video*."""
    db_path = tmp_path / "target.db"
    conn = init_db(db_path)
    run_id = store_run(
        conn,
        source_video=run_tracking.normalise_video_path(video),
        start_time=datetime(2026, 11, 25, 10, 0, 0),
    )
    conn.close()
    return db_path, run_id


def test_help_exits_before_any_database_or_video_is_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse handles --help before the gate, imports, video or DB setup."""
    db_path = tmp_path / "should_not_exist.db"
    video = tmp_path / "should_not_exist.mp4"

    with pytest.raises(SystemExit) as excinfo:
        run_tracking.main(["--help", "--db", str(db_path), "--video", str(video)])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "--video" in out and "--db" in out and "--force" in out
    assert not db_path.exists()
    assert not video.exists()


def test_existing_source_video_is_refused_read_only(tmp_path: Path) -> None:
    """The check returns the blocking run id without touching the DB file."""
    video = tmp_path / "new_clip.mp4"
    db_path, run_id = _db_with_run(tmp_path, video)
    before = (db_path.stat().st_size, db_path.stat().st_mtime_ns)

    assert run_tracking.existing_run_ids(
        db_path, run_tracking.normalise_video_path(video)
    ) == [run_id]
    assert (db_path.stat().st_size, db_path.stat().st_mtime_ns) == before

    other = tmp_path / "other_clip.mp4"
    assert run_tracking.existing_run_ids(
        db_path, run_tracking.normalise_video_path(other)
    ) == []


def test_force_allows_a_duplicate_and_missing_db_is_not_created(tmp_path: Path) -> None:
    """--force skips the gate; a missing DB means there is nothing to block."""
    video = tmp_path / "new_clip.mp4"
    db_path, _run_id = _db_with_run(tmp_path, video)
    source = run_tracking.normalise_video_path(video)

    assert run_tracking.existing_run_ids(db_path, source, force=True) == []

    missing_db = tmp_path / "missing.db"
    assert run_tracking.existing_run_ids(missing_db, source) == []
    assert not missing_db.exists()
