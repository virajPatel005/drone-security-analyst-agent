"""Pure track-to-event state machine with no model or video imports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from src.perception.observations import BBox, Observation
from src.telemetry.records import FrameRecord

EventType = Literal["appeared", "entered_zone", "left"]


@dataclass(frozen=True, slots=True)
class TrackEvent:
    event_type: EventType
    track_id: int
    category: str
    class_name: str
    timestamp: datetime
    zone: str
    bbox: BBox
    confidence: float
    dwell_s: float | None = None
    end_of_video: bool = False


@dataclass
class _TrackState:
    track_id: int
    class_name: str
    category: str
    first_seen_time: datetime
    last_seen_time: datetime
    last_seen_frame: int
    frames_seen: int
    consecutive_hits: int
    confirmed: bool
    zone: str | None
    zone_entered_at: datetime | None
    bbox: BBox
    confidence: float


class EventBuilder:
    """Turn per-frame observations into appeared/entered_zone/left events.

    Zone means the drone's current telemetry location, not an image region.
    A track must be confirmed before it appears, and a lost track is removed
    so a reused tracker ID later becomes a fresh appearance.
    """

    def __init__(
        self,
        *,
        min_track_frames: int,
        lost_frames: int,
        class_categories: Mapping[str, str],
    ) -> None:
        if min_track_frames < 1:
            raise ValueError("min_track_frames must be at least 1")
        if lost_frames < 1:
            raise ValueError("lost_frames must be at least 1")
        self._min_track_frames = min_track_frames
        self._lost_frames = lost_frames
        self._class_categories = dict(class_categories)
        self._states: dict[int, _TrackState] = {}

    def process_frame(
        self,
        record: FrameRecord,
        observations: Sequence[Observation],
    ) -> list[TrackEvent]:
        events: list[TrackEvent] = []

        for track_id, state in list(self._states.items()):
            if record.frame_index - state.last_seen_frame >= self._lost_frames:
                if state.confirmed:
                    events.append(self._left_event(state, end_of_video=False))
                del self._states[track_id]

        seen_ids: set[int] = set()
        for observation in observations:
            if observation.track_id in seen_ids:
                continue
            seen_ids.add(observation.track_id)

            category = self._class_categories.get(observation.class_name)
            if category is None:
                continue

            state = self._states.get(observation.track_id)
            if state is None:
                state = _TrackState(
                    track_id=observation.track_id,
                    class_name=observation.class_name,
                    category=category,
                    first_seen_time=record.timestamp,
                    last_seen_time=record.timestamp,
                    last_seen_frame=record.frame_index,
                    frames_seen=1,
                    consecutive_hits=1,
                    confirmed=False,
                    zone=record.location,
                    zone_entered_at=record.timestamp,
                    bbox=observation.bbox,
                    confidence=observation.confidence,
                )
                self._states[observation.track_id] = state
                events.extend(self._confirm_if_ready(state, record))
                continue

            gap = record.frame_index - state.last_seen_frame
            previous_last_seen = state.last_seen_time
            previous_zone = state.zone

            state.class_name = observation.class_name
            state.category = category
            state.last_seen_time = record.timestamp
            state.last_seen_frame = record.frame_index
            state.frames_seen += 1
            state.bbox = observation.bbox
            state.confidence = observation.confidence

            if previous_zone != record.location:
                dwell_s = self._dwell(previous_last_seen, state.zone_entered_at)
                state.zone = record.location
                state.zone_entered_at = record.timestamp
                if state.confirmed:
                    events.append(
                        self._event(
                            state,
                            "entered_zone",
                            record.timestamp,
                            record.location,
                            dwell_s,
                        )
                    )

            if not state.confirmed:
                state.consecutive_hits = state.consecutive_hits + 1 if gap == 1 else 1
                events.extend(self._confirm_if_ready(state, record))

        return events

    def finish(self) -> list[TrackEvent]:
        """Close confirmed tracks still open, marking them end_of_video."""

        events = [
            self._left_event(state, end_of_video=True)
            for state in self._states.values()
            if state.confirmed
        ]
        self._states.clear()
        return events

    def _confirm_if_ready(self, state: _TrackState, record: FrameRecord) -> list[TrackEvent]:
        if state.confirmed or state.consecutive_hits < self._min_track_frames:
            return []

        state.confirmed = True
        return [
            self._event(state, "appeared", record.timestamp, record.location),
            self._event(state, "entered_zone", record.timestamp, record.location),
        ]

    def _left_event(self, state: _TrackState, *, end_of_video: bool) -> TrackEvent:
        return self._event(
            state,
            "left",
            state.last_seen_time,
            state.zone or "",
            self._dwell(state.last_seen_time, state.zone_entered_at),
            end_of_video=end_of_video,
        )

    def _event(
        self,
        state: _TrackState,
        event_type: EventType,
        timestamp: datetime,
        zone: str,
        dwell_s: float | None = None,
        *,
        end_of_video: bool = False,
    ) -> TrackEvent:
        return TrackEvent(
            event_type=event_type,
            track_id=state.track_id,
            category=state.category,
            class_name=state.class_name,
            timestamp=timestamp,
            zone=zone,
            bbox=state.bbox,
            confidence=state.confidence,
            dwell_s=dwell_s,
            end_of_video=end_of_video,
        )

    @staticmethod
    def _dwell(end: datetime, start: datetime | None) -> float:
        if start is None:
            return 0.0
        return max(0.0, (end - start).total_seconds())
