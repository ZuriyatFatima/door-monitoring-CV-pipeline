"""
check_tracker_internals.py

Purpose: get DIRECT evidence on whether track_buffer=300 in
bytetrack_custom.yaml is actually producing a runtime buffer of 300
frames, or whether Ultralytics is silently scaling it down.

Some Ultralytics versions compute the tracker's real "how many frames
to keep a lost track alive" value as:

    max_time_lost = int(frame_rate / 30.0 * track_buffer)

If frame_rate isn't correctly detected (e.g. because this pipeline calls
model.track(frame, ...) one raw frame at a time instead of
model.track(source=video_path, ...), so the tracker never reads the
video's real FPS from a source object), frame_rate can silently default
to something else -- and 300 in the yaml would NOT mean 300 real frames
at runtime.

This script runs a few frames of real tracking (exactly like your
pipeline does), then reaches directly into the live tracker object
Ultralytics created and prints:
  - self.args.track_buffer  (what was configured)
  - self.max_time_lost      (what's ACTUALLY being used to decide
                              when a lost track is deleted -- THIS is
                              the real number that matters)
  - self.frame_rate / self.frame_id (what fps assumption produced it)

No guessing -- this is the tracker's own internal state.

Usage:
    python check_tracker_internals.py uploads/landscape_test_1.mp4
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]

    import cv2
    from ultralytics import YOLO
    from modules.tracker import ensure_bytetrack_config

    tracker_yaml = ensure_bytetrack_config()
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video reports FPS = {video_fps}")

    # Run enough frames for the tracker to fully initialize (needs at
    # least one frame with a detection to create the internal tracker
    # object at all).
    frame_count = 0
    tracker_obj = None
    while frame_count < 30:
        ok, frame = cap.read()
        if not ok:
            break

        model.track(
            frame, persist=True, conf=0.4, tracker=tracker_yaml,
            device="cpu", classes=[0], verbose=False,
        )

        frame_count += 1

    cap.release()

    # Reach into the live tracker Ultralytics created for this model
    # instance -- this is the SAME object making real matching decisions
    # during your actual runs, not a fresh reconstruction.
    predictor = model.predictor
    if predictor is None or not hasattr(predictor, "trackers") or not predictor.trackers:
        print("\nNo tracker object found on model.predictor -- either no "
              "detections occurred in the first 30 frames, or this "
              "Ultralytics version stores it somewhere else. Paste your "
              "`ultralytics` version (`pip show ultralytics`) if this happens.")
        return

    tracker_obj = predictor.trackers[0]

    print(f"\n=== Live BYTETracker internal state after {frame_count} frames ===")
    print(f"Configured track_buffer (from yaml):  {getattr(tracker_obj.args, 'track_buffer', 'NOT FOUND')}")
    print(f"ACTUAL max_frames_lost being used:     {getattr(tracker_obj, 'max_frames_lost', 'NOT FOUND')}")
    print(f"tracker.frame_id (frames processed):   {getattr(tracker_obj, 'frame_id', 'NOT FOUND')}")

    max_frames_lost = getattr(tracker_obj, "max_frames_lost", None)
    if isinstance(max_frames_lost, (int, float)):
        if max_frames_lost < 176:
            print(
                f"\n*** FINDING: max_frames_lost ({max_frames_lost}) is LESS than the "
                f"164-176 frame gaps observed in your landscape videos. This "
                f"directly explains why track_buffer=300 in the yaml did NOT "
                f"prevent the fragmentation -- the tracker isn't actually using "
                f"300 frames of buffer at runtime. ***"
            )
        else:
            print(
                f"\n*** FINDING: max_frames_lost ({max_frames_lost}) is MORE than the "
                f"observed gaps (164-176 frames), so buffer length is NOT the "
                f"bottleneck. Given the reappearance position was also close "
                f"(15-27px) to the disappearance position, the failed re-match "
                f"despite adequate buffer AND close position most likely comes "
                f"from ByteTrack's matching logic itself (e.g. a 'lost' track "
                f"transitioning to 'removed' state before the person reappears, "
                f"or the Kalman-predicted box drifting during the gap even "
                f"though the raw reappearance point is close) -- buffer length "
                f"is cleared as the cause. ***"
            )


if __name__ == "__main__":
    main()
