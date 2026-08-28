"""
optimize.py
Day 4 deliverable: simple, measurable FPS optimizations.
Two independent levers — resize and frame-skip — plus a small FPS
tracker so you can quote real before/after numbers in your notes
instead of guessing.
"""

from __future__ import annotations
import time
import cv2


def resize_frame(frame, target_width: int | None = None):
    """
    Resizes frame to target_width, preserving aspect ratio.
    Cheapest FPS win: smaller frames = less work for both YOLO inference
    and any downstream drawing/encoding.
    Pass target_width=None to skip resizing (identity — for A/B comparison).
    """
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


class FrameSkipper:
    """
    Runs full detection/tracking every `skip_n`-th frame, and cheaply
    reuses the last known boxes/count state on the frames in between.
    This trades a little positional accuracy for real throughput —
    document the trade-off in your Day 4 notes rather than hiding it.
    """

    def __init__(self, skip_n: int = 1):
        """skip_n=1 means no skipping (process every frame)."""
        self.skip_n = max(1, skip_n)
        self._counter = 0

    def should_process(self) -> bool:
        should = self._counter % self.skip_n == 0
        self._counter += 1
        return should


class FPSTracker:
    """Rolling FPS measurement — use this to write real numbers in your
    Day 4 'FPS improvement notes' deliverable, not estimates."""

    def __init__(self, window: int = 30):
        self.window = window
        self._timestamps = []

    def tick(self) -> float:
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) > self.window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
