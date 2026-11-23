"""Pure event-builder tests using fake observations, no GPU or models."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.config.settings import CLASS_CATEGORIES
from src.perception.events import EventBuilder
from src.perception.observations import Observation
from src.telemetry.records import FrameRecord

START = datetime(2026, 11, 23, 23, 59, 45)
CATEGORIES = {
    "person": "person",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "bicycle": "vehicle",
}


def _record(frame_index: int, zone: str = "gate") -> FrameRecord:
    seconds = float(frame_index - 1)
    return FrameRecord(
        frame_index=frame_index,
        video_time_s=seconds,
        timestamp=START + timedelta(seconds=seconds),
        location=zone,
        altitude_m=30.0,
        drone_x=0.0,
        drone_y=0.0,
        image=None,
    )


def _observation(
    track_id: int = 1,
    class_name: str = "car",
    confidence: float = 0.9,
) -> Observation:
    return Observation(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(0.0, 0.0, 10.0, 10.0),
    )


def test_track_is_confirmed_after_min_track_frames():
    builder = EventBuilder(
        min_track_frames=2,
        lost_frames=10,
        class_categories=CATEGORIES,
    )

    assert builder.process_frame(_record(1), [_observation()]) == []

    events = builder.process_frame(_record(2), [_observation()])

    assert [event.event_type for event in events] == ["appeared", "entered_zone"]
    assert events[0].track_id == 1
    assert events[0].category == "vehicle"
    assert events[0].timestamp == _record(2).timestamp


def test_one_frame_flicker_creates_no_events():
    builder = EventBuilder(
        min_track_frames=2,
        lost_frames=10,
        class_categories=CATEGORIES,
    )

    assert builder.process_frame(_record(1), [_observation()]) == []
    assert builder.process_frame(_record(2), []) == []
    assert builder.process_frame(_record(3), []) == []
    assert builder.finish() == []


def test_zone_change_reports_previous_zone_dwell():
    builder = EventBuilder(
        min_track_frames=2,
        lost_frames=10,
        class_categories=CATEGORIES,
    )
    builder.process_frame(_record(1, "gate"), [_observation()])
    builder.process_frame(_record(2, "gate"), [_observation()])
    builder.process_frame(_record(3, "gate"), [_observation()])

    events = builder.process_frame(_record(4, "garage"), [_observation()])

    assert [event.event_type for event in events] == ["entered_zone"]
    assert events[0].zone == "garage"
    assert events[0].dwell_s == 2.0


def test_track_is_closed_after_lost_frames():
    builder = EventBuilder(
        min_track_frames=2,
        lost_frames=2,
        class_categories=CATEGORIES,
    )
    builder.process_frame(_record(1, "gate"), [_observation()])
    builder.process_frame(_record(2, "gate"), [_observation()])
    builder.process_frame(_record(3, "gate"), [_observation()])

    events = builder.process_frame(_record(4, "gate"), [])
    events += builder.process_frame(_record(5, "gate"), [])

    assert [event.event_type for event in events] == ["left"]
    assert events[0].timestamp == _record(3).timestamp
    assert events[0].dwell_s == 2.0
    assert events[0].end_of_video is False


def test_finish_closes_open_track_at_end_of_video():
    builder = EventBuilder(
        min_track_frames=1,
        lost_frames=10,
        class_categories=CATEGORIES,
    )
    builder.process_frame(_record(1, "gate"), [_observation()])
    builder.process_frame(_record(2, "gate"), [_observation()])
    builder.process_frame(_record(3, "gate"), [_observation()])

    events = builder.finish()

    assert [event.event_type for event in events] == ["left"]
    assert events[0].end_of_video is True
    assert events[0].timestamp == _record(3).timestamp
    assert events[0].dwell_s == 2.0


def test_reappearing_id_after_lost_frames_is_a_new_appearance():
    builder = EventBuilder(
        min_track_frames=2,
        lost_frames=2,
        class_categories=CATEGORIES,
    )
    builder.process_frame(_record(1), [_observation(track_id=7)])
    appeared = builder.process_frame(_record(2), [_observation(track_id=7)])
    builder.process_frame(_record(3), [])
    lost = builder.process_frame(_record(4), [])
    builder.process_frame(_record(5), [_observation(track_id=7)])
    reappeared = builder.process_frame(_record(6), [_observation(track_id=7)])

    assert [event.event_type for event in appeared] == ["appeared", "entered_zone"]
    assert [event.event_type for event in lost] == ["left"]
    assert [event.event_type for event in reappeared] == ["appeared", "entered_zone"]
    assert [event.end_of_video for event in lost] == [False]


def test_class_filter_and_category_mapping():
    assert CLASS_CATEGORIES == {
        "person": "person",
        "car": "vehicle",
        "motorcycle": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "bicycle": "vehicle",
    }
    builder = EventBuilder(
        min_track_frames=1,
        lost_frames=10,
        class_categories=CLASS_CATEGORIES,
    )

    events = builder.process_frame(
        _record(1),
        [
            _observation(track_id=1, class_name="stop sign"),
            _observation(track_id=2, class_name="car"),
            _observation(track_id=3, class_name="person"),
        ],
    )

    appeared = [event for event in events if event.event_type == "appeared"]
    assert [(event.track_id, event.class_name, event.category) for event in appeared] == [
        (2, "car", "vehicle"),
        (3, "person", "person"),
    ]
