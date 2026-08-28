"""
logger.py
Handles the Day 3 deliverable: CSV/Excel export + event screenshot folder.
Keeps this logic out of the dashboard UI code so it can be reused by
CLI scripts, tests, or a future FastAPI service without rewriting it.
"""

import os
import csv
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None  # screenshot saving degrades gracefully if cv2 isn't available


class EventLogger:
    def __init__(self, export_dir: str = "exports", screenshot_dir: str = "screenshots"):
        self.export_dir = export_dir
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.export_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.csv_path = os.path.join(
            self.export_dir, f"crossing_events_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        self._rows = []

    def log_event(self, event, frame=None):
        """
        event:  a counter.CrossingEvent
        frame:  optional raw frame (numpy array) to save as an evidence
                screenshot at the moment of crossing
        """
        screenshot_path = ""
        if frame is not None and cv2 is not None:
            screenshot_path = os.path.join(
                self.screenshot_dir,
                f"frame{event.frame_idx}_id{event.track_id}_{event.direction}.jpg",
            )
            cv2.imwrite(screenshot_path, frame)

        self._rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_idx": event.frame_idx,
                "track_id": event.track_id,
                "direction": event.direction,
                "y_position": round(event.y_position, 1),
                "door_state": event.door_state,
                "screenshot": screenshot_path,
            }
        )

    def flush_csv(self):
        """Writes all logged rows to CSV. Call at the end of a run,
        or periodically for long-running live streams."""
        if not self._rows:
            return self.csv_path
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)
        return self.csv_path

    def to_excel(self):
        """Optional Excel export — requires openpyxl (see requirements.txt)."""
        import pandas as pd

        xlsx_path = self.csv_path.replace(".csv", ".xlsx")
        pd.DataFrame(self._rows).to_excel(xlsx_path, index=False)
        return xlsx_path

    @property
    def rows(self):
        return list(self._rows)
