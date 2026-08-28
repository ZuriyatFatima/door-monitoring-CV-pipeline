"""
tracker.py
Person tracking (for line-crossing counts) using your locked Week 4
ByteTrack config, PLUS a separate door-state classifier (best.pt) used
to tag each crossing event with the door's state at that moment.

These are two different jobs:
  - PersonTracker  -> tracks people (COCO class 0) across frames, IDs
                      persist so the same person isn't recounted.
  - DoorStateClassifier -> runs your door_open/door_closed model on a
                      single frame (no tracking needed — a door doesn't
                      move frame-to-frame), called once per crossing
                      event rather than every frame, to keep it cheap.

Locked tracker config (do not re-tune unless you have a specific reason):
    new_track_thresh = 0.45
    track_buffer     = 300  (raised from 120 during Week 7 QA -- see note below)
    match_thresh     = 0.7
    conf (model.track)= 0.4
    MIN_HITS          = 5   (min consecutive hits before an ID is "confirmed")

Week 7 change: track_buffer raised 120 -> 300. Root cause found during
QA: at ~28fps, track_buffer=120 only keeps a lost track alive for
~4.3 seconds. Real footage showed people pausing (e.g. to close a door
behind them) longer than that, causing ByteTrack to give up and assign
a new track ID -- one real person fragmenting into 2-3 IDs, which in
turn caused inconsistent event counts and inconsistent IN/OUT direction
labeling. 300 frames covers ~10-11 seconds at 28fps, well past the
pauses observed in testing. This must be re-verified against the
test_1.mp4 regression baseline (IN=6, OUT=2, 8 events) before trusting
it -- a buffer that's too generous can itself cause false re-identification
(merging two different people into one ID), so this is a tested change,
not a blind increase.
"""

import os
from ultralytics import YOLO

# --- Tracker config (Week 7: track_buffer raised from 120, see note above) ---
BYTETRACK_YAML_CONTENT = """\
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.45
track_buffer: 300
match_thresh: 0.7
fuse_score: True
"""

TRACK_CONF = 0.4
MIN_HITS = 5
COCO_PERSON_CLASS_ID = 0  # 'person' in the standard COCO class list


def ensure_bytetrack_config(path: str = "bytetrack_custom.yaml") -> str:
    """
    Writes the current bytetrack config to disk, always overwriting any
    existing file at this path.

    IMPORTANT (Week 7 fix): this used to only write if the file didn't
    already exist, which meant a code change to BYTETRACK_YAML_CONTENT
    silently had NO effect if a stale config file was already sitting on
    disk from an earlier run -- the tracker would keep using the old
    settings without any error or warning. Always overwriting is slightly
    more disk I/O but guarantees the code and the config on disk never
    drift apart.
    """
    with open(path, "w") as f:
        f.write(BYTETRACK_YAML_CONTENT)
    return path


class PersonTracker:
    """
    Tracks PEOPLE (not doors) for line-crossing counting — this is what
    drives your IN/OUT counter, matching the Week 4 design.
    """

    def __init__(self, model_path: str = "yolo11n.pt", device: str = "cpu", imgsz: int = 640):
        """
        model_path: a general-purpose COCO model (yolo11n.pt is fine —
                    Ultralytics auto-downloads it on first use if it's
                    not already cached locally). This is NOT your door
                    classifier — do not point this at best.pt.
        device:     'cpu', 'cuda:0', etc.
        imgsz:      inference resolution — lower this for the Day 4 FPS pass.
        """
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.tracker_yaml = ensure_bytetrack_config()

    def track_frame(self, frame, persist: bool = True):
        """
        Runs person detection+tracking on a single frame, filtered to
        the COCO 'person' class only. Returns the raw Ultralytics
        Results object — callers pull boxes/ids/conf out of it.
        """
        results = self.model.track(
            frame,
            persist=persist,
            conf=TRACK_CONF,
            tracker=self.tracker_yaml,
            device=self.device,
            imgsz=self.imgsz,
            classes=[COCO_PERSON_CLASS_ID],
            verbose=False,
        )
        return results[0]


class DoorStateClassifier:
    """
    Wraps your fine-tuned door_open/door_closed model (best.pt). Used
    for single-frame classification at the moment of a crossing event
    — NOT for tracking, since a door doesn't move across the frame.
    """

    def __init__(self, model_path: str, device: str = "cpu", imgsz: int = 640, conf: float = 0.4):
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def classify_frame(self, frame) -> str:
        """
        Runs the door detector on a single frame and returns the
        highest-confidence class name found ("door_open" / "door_closed"),
        or "unknown" if nothing was detected above threshold.
        """
        results = self.model.predict(
            frame, conf=self.conf, device=self.device, imgsz=self.imgsz, verbose=False
        )[0]
        if results.boxes is None or len(results.boxes) == 0:
            return "unknown"
        # Take the highest-confidence detection as the door's current state
        best_idx = results.boxes.conf.argmax().item()
        class_id = int(results.boxes.cls[best_idx].item())
        return results.names[class_id]
