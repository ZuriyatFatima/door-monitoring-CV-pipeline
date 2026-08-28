"""
counter.py
Line-crossing IN/OUT counting logic, ported from your Week 4 script.
Horizontal line at LINE_Y, no x-band restriction (matches your locked config).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CrossingEvent:
    frame_idx: int
    track_id: int
    direction: str          # "IN" or "OUT"
    y_position: float
    door_state: str = "unknown"   # filled in by door-state detector, if wired in


class LineCounter:
    """
    Tracks each object's previous y-center and fires a crossing event the
    first time its center crosses LINE_Y, matching Week 4 behaviour:
    MIN_HITS consecutive detections before an ID is eligible to count,
    counted_ids set prevents double-counting the same track.
    """

    def __init__(self, line_y: int, min_hits: int = 5):
        self.line_y = line_y
        self.min_hits = min_hits
        self.hit_counts = {}      # track_id -> consecutive hit count
        self.prev_y = {}          # track_id -> last y-center seen
        self.counted_ids = set()  # track_ids already counted (Week4 dedup)
        self.in_count = 0
        self.out_count = 0
        self.events: list[CrossingEvent] = []

    def update(self, frame_idx: int, track_id: int, y_center: float) -> CrossingEvent | None:
        """
        Call once per tracked box per frame. Returns a CrossingEvent if
        this update caused a new crossing, else None.
        """
        self.hit_counts[track_id] = self.hit_counts.get(track_id, 0) + 1

        event = None
        prev = self.prev_y.get(track_id)

        if (
            prev is not None
            and self.hit_counts[track_id] >= self.min_hits
            and track_id not in self.counted_ids
        ):
            crossed_down = prev < self.line_y <= y_center   # IN
            crossed_up = prev > self.line_y >= y_center     # OUT

            if crossed_down or crossed_up:
                direction = "IN" if crossed_down else "OUT"
                if direction == "IN":
                    self.in_count += 1
                else:
                    self.out_count += 1
                self.counted_ids.add(track_id)
                event = CrossingEvent(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    direction=direction,
                    y_position=y_center,
                )
                self.events.append(event)

        self.prev_y[track_id] = y_center
        return event

    def summary(self) -> dict:
        return {
            "in_count": self.in_count,
            "out_count": self.out_count,
            "total_events": len(self.events),
            "unique_tracks_seen": len(self.hit_counts),
        }
