"""Tests for the VLM description stage — fake client, no LM Studio, no GPU."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from src.description import (
    CropRecord,
    describe_run,
    parse_description,
    save_track_crop,
)
from src.storage import init_db, query_descriptions, store_description, store_run

VALID_JSON = (
    '{"object_type": "truck", "main_colour": "white", '
    '"description": "A white truck seen from above."}'
)


class FakeVlm:
    """Records calls and returns a fixed raw reply."""

    model_name = "fake-vlm"

    def __init__(self, reply: str = VALID_JSON) -> None:
        self.reply = reply
        self.calls: list[Path] = []

    def describe_crop(self, crop_path: Path) -> str:
        self.calls.append(Path(crop_path))
        return self.reply


def _add_track(conn, run_id: int, track_id: int, class_name: str = "car") -> None:
    """Insert a confirmed track row directly (tracks are the confirmation source)."""
    conn.execute(
        "INSERT INTO tracks (run_id, track_id, class_name, first_seen, last_seen, frames_seen, max_confidence) "
        "VALUES (?, ?, ?, ?, ?, 5, 0.8)",
        (run_id, track_id, class_name, "2026-11-24T14:00:00", "2026-11-24T14:00:05"),
    )
    conn.commit()


def _crop(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake-jpeg-bytes")
    return path


def _record(track_id: int, path: Path | None, *, too_small: bool = False) -> CropRecord:
    return CropRecord(
        track_id=track_id,
        class_name="car",
        crop_path=path,
        orig_width=30,
        orig_height=24,
        too_small=too_small,
    )


def test_describes_each_confirmed_track_once(tmp_path):
    """One call per confirmed track, even if the crop list repeats a track."""
    conn = init_db(":memory:")
    run_id = store_run(
        conn, source_video="test.mp4", start_time=datetime(2026, 11, 24, 14, 0), tuning_settings={}
    )
    _add_track(conn, run_id, 1)
    _add_track(conn, run_id, 2)

    fake = FakeVlm()
    crops = [
        _record(1, _crop(tmp_path, "1.jpg")),
        _record(1, _crop(tmp_path, "1_duplicate.jpg")),
        _record(2, _crop(tmp_path, "2.jpg")),
    ]
    summary = describe_run(conn, run_id=run_id, crops=crops, client=fake)

    assert len(fake.calls) == 2
    assert summary.described == 2
    assert summary.error == 0
    rows = query_descriptions(conn, run_id=run_id)
    assert [row["track_id"] for row in rows] == [1, 2]
    assert all(row["status"] == "described" for row in rows)
    assert rows[0]["attributes"]["main_colour"] == "white"
    conn.close()


def test_second_pass_does_not_call_vlm_again(tmp_path):
    """A rerun over the same run sees stored rows and makes no new calls."""
    conn = init_db(":memory:")
    run_id = store_run(
        conn, source_video="test.mp4", start_time=datetime(2026, 11, 24, 14, 0), tuning_settings={}
    )
    _add_track(conn, run_id, 1)

    fake = FakeVlm()
    crops = [_record(1, _crop(tmp_path, "1.jpg"))]
    describe_run(conn, run_id=run_id, crops=crops, client=fake)
    assert len(fake.calls) == 1

    summary = describe_run(conn, run_id=run_id, crops=crops, client=fake)
    assert len(fake.calls) == 1, "second pass must not call the VLM"
    assert summary.described == 1
    assert len(query_descriptions(conn, run_id=run_id)) == 1
    conn.close()


def test_too_small_crop_skipped_without_vlm_call():
    """A box below the short-side floor is stored as too_small, never sent."""
    conn = init_db(":memory:")
    run_id = store_run(
        conn, source_video="test.mp4", start_time=datetime(2026, 11, 24, 14, 0), tuning_settings={}
    )
    _add_track(conn, run_id, 1)

    fake = FakeVlm()
    summary = describe_run(
        conn,
        run_id=run_id,
        crops=[_record(1, None, too_small=True)],
        client=fake,
    )

    assert fake.calls == []
    assert summary.too_small == 1
    row = query_descriptions(conn, run_id=run_id)[0]
    assert row["status"] == "too_small"
    assert row["description"] is None
    assert row["crop_path"] is None
    assert row["latency_ms"] is None
    conn.close()


def test_parse_failure_stored_as_error_with_raw_text(tmp_path):
    """Unparseable model text is kept verbatim with status error."""
    conn = init_db(":memory:")
    run_id = store_run(
        conn, source_video="test.mp4", start_time=datetime(2026, 11, 24, 14, 0), tuning_settings={}
    )
    _add_track(conn, run_id, 1)

    fake = FakeVlm(reply="I cannot tell what this is.")
    summary = describe_run(
        conn, run_id=run_id, crops=[_record(1, _crop(tmp_path, "1.jpg"))], client=fake
    )

    assert summary.error == 1
    row = query_descriptions(conn, run_id=run_id)[0]
    assert row["status"] == "error"
    assert row["description"] == "I cannot tell what this is."
    assert row["attributes"] is None
    conn.close()


def test_unconfirmed_tracks_are_never_sent(tmp_path):
    """A crop whose track has no tracks-table row must not reach the VLM."""
    conn = init_db(":memory:")
    run_id = store_run(
        conn, source_video="test.mp4", start_time=datetime(2026, 11, 24, 14, 0), tuning_settings={}
    )

    fake = FakeVlm()
    summary = describe_run(
        conn, run_id=run_id, crops=[_record(99, _crop(tmp_path, "99.jpg"))], client=fake
    )

    assert fake.calls == []
    assert summary.described == 0
    assert query_descriptions(conn, run_id=run_id) == []
    conn.close()


def test_parse_description_accepts_fenced_json():
    """Models often wrap JSON in a Markdown fence; parsing must cope."""
    raw = 'Here you go:\n```json\n{"object_type": "car", "main_colour": "red", "description": "A red car."}\n```'
    parsed = parse_description(raw)
    assert parsed == {
        "object_type": "car",
        "main_colour": "red",
        "description": "A red car.",
    }
    assert parse_description("no json here") is None
    assert parse_description('{"object_type": "car"}') is None


def test_save_track_crop_pads_clamps_and_upscales(tmp_path):
    """Padding stays inside the frame and the short side reaches the target."""
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    path = tmp_path / "crop.jpg"
    too_small, width, height = save_track_crop(
        image,
        (10.0, 20.0, 30.0, 40.0),
        path,
        pad_ratio=0.5,
        min_pad_px=2,
        min_short_side_px=20,
        target_short_side_px=336,
    )

    assert too_small is False
    assert (width, height) == (20, 20)
    assert path.is_file()
    import cv2

    saved = cv2.imread(str(path))
    assert min(saved.shape[:2]) == 336
    # The original padded crop is 40x40 (square box + 10px each side).
    # 336 / 40 = 8.4 scaling factor. Width is the first dim after resize.
    assert saved.shape[1] == 336  # width after upscale


def test_save_track_crop_rejects_too_small_box(tmp_path):
    """A box below the floor is reported and no file is written."""
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    path = tmp_path / "small.jpg"

    too_small, width, height = save_track_crop(
        image,
        (5.0, 5.0, 15.0, 13.0),
        path,
        min_short_side_px=20,
    )

    assert too_small is True
    assert (width, height) == (10, 8)
    assert not path.exists()


def test_description_query_by_run_and_track():
    """Reads can be filtered by run, by track, or both."""
    conn = init_db(":memory:")
    run_a = store_run(
        conn, source_video="a.mp4", start_time=datetime(2026, 11, 24, 14, 0), tuning_settings={}
    )
    run_b = store_run(
        conn, source_video="b.mp4", start_time=datetime(2026, 11, 24, 15, 0), tuning_settings={}
    )
    for run_id, track_id in ((run_a, 1), (run_a, 2), (run_b, 1)):
        store_description(
            conn,
            run_id=run_id,
            track_id=track_id,
            status="described",
            description=f"run {run_id} track {track_id}",
            attributes={"object_type": "car"},
            crop_path="/tmp/crop.jpg",
            orig_width=30,
            orig_height=20,
            model_name="fake",
            latency_ms=12.5,
        )

    assert len(query_descriptions(conn, run_id=run_a)) == 2
    assert len(query_descriptions(conn, track_id=1)) == 2
    only = query_descriptions(conn, run_id=run_a, track_id=2)
    assert len(only) == 1
    assert only[0]["description"] == f"run {run_a} track 2"
    conn.close()
