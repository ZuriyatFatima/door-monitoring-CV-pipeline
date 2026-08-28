"""
verify_two_person_scene.py

Week 8 - F6 follow-up: manual verification harness for real
two-people-close-together footage.

WHY THIS EXISTS
----------------
The reconciler has so far only been checked against:
  - synthetic negative controls in test_track_reconciler.py
  - incidental merges in test_1.mp4 (informational only, not annotated ground truth)

Neither proves the reconciler won't merge two DIFFERENT real people who
happen to pass a similar spot within dist_threshold_px / gap_frames_threshold
of each other. This script does not decide pass/fail for you -- it dumps
every merge decision the reconciler made, with enough detail that you can
watch the corresponding seconds of footage and judge for yourself.

INTERFACES USED (confirmed directly from app.py / run_tracking_qa.py /
test_track_reconciler.py -- not guessed):

    PersonTracker(model_path=..., device=...).track_frame(frame)
        -> Ultralytics-style result: .boxes.xywh, .boxes.id, .boxes.conf
        (track_buffer is NOT a constructor arg here -- it lives in
        bytetrack_custom.yaml on disk, already set to 300, loaded
        internally via ensure_bytetrack_config())

    LineCounter(line_y=..., min_hits=5)
        .update(frame_idx, track_id, y) -> event dict or None
        .summary() -> dict with in_count/out_count/total_events/unique_tracks_seen

    TrackReconciler(gap_frames_threshold=..., dist_threshold_px=...)
        .reconcile(frame_idx, track_id, x, y) -> canonical_id
        .merge_log -> list (exact per-entry shape not independently
        confirmed here since modules/track_reconciler.py itself wasn't
        available to read directly -- only its call sites. Printed
        generically below so nothing is silently assumed about its fields.)

REMAINING UNCERTAINTY, STATED PLAINLY
---------------------------------------
This is built from how your real files CALL these classes, which is much
stronger evidence than a guess, but I have not read modules/tracker.py,
modules/counter.py, or modules/track_reconciler.py directly. If anything
below still doesn't match (e.g. min_hits behaves differently, or
merge_log entries have different fields), that's the next thing to fix --
paste those three files directly for full certainty.

USAGE
-----
    python verify_two_person_scene.py path/to/two_people.mp4
    python verify_two_person_scene.py path/to/two_people.mp4 --line-y 200 --dist-threshold-px 100
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from modules.tracker import PersonTracker
from modules.counter import LineCounter
from modules.track_reconciler import TrackReconciler
from threshold_scaling import scaled_dist_threshold_px, REFERENCE_FRAME_WIDTH, REFERENCE_DIST_THRESHOLD_PX

PERSON_MODEL_PATH = "yolo11n.pt"


def run(video_path, line_y=None, dist_threshold_px=None, gap_frames_threshold=300, device="cpu"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if line_y is None:
        line_y = h // 2  # same fallback run_tracking_qa.py uses when no line_y is given

    # THRESHOLD SCALING FIX: dist_threshold_px=150 was tuned against
    # landscape_test_1/2 (848px wide) and became ~2.4x too permissive,
    # relative to frame size, on the 360px-wide passageway/terrace test
    # footage -- the diagnosed cause of the 77%/94% over-merging.
    # If the caller didn't explicitly override it, scale it to preserve
    # the SAME fraction of frame width (17.7%) that's already validated
    # on the landscape footage, instead of reusing an absolute pixel
    # count that gets looser on smaller footage.
    auto_scaled = dist_threshold_px is None
    if auto_scaled:
        dist_threshold_px = scaled_dist_threshold_px(w)

    person_tracker = PersonTracker(model_path=PERSON_MODEL_PATH, device=device)
    counter = LineCounter(line_y=line_y, min_hits=5)
    reconciler = TrackReconciler(gap_frames_threshold=gap_frames_threshold, dist_threshold_px=dist_threshold_px)

    track_first_seen_frame = {}
    track_last_seen_frame = {}
    track_last_real_pos = {}
    raw_ids_seen = set()
    canonical_ids_seen = set()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = person_tracker.track_frame(frame)

        if result is not None and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for (x, y, bw, bh), track_id in zip(boxes, ids):
                tid = int(track_id)
                raw_ids_seen.add(tid)

                if tid not in track_first_seen_frame:
                    track_first_seen_frame[tid] = frame_idx
                track_last_seen_frame[tid] = frame_idx
                track_last_real_pos[tid] = (float(x), float(y))

                # CRITICAL (F6 correction, confirmed in app.py / run_tracking_qa.py):
                # the counter always receives the RAW track_id, never the
                # reconciled canonical_id. Do not change this.
                counter.update(frame_idx, tid, float(y))

                canonical_id = reconciler.reconcile(frame_idx, tid, float(x), float(y))
                canonical_ids_seen.add(canonical_id)

        frame_idx += 1

    cap.release()

    print("=" * 70)
    print(f"VIDEO: {video_path}  ({w}x{h}, line_y={line_y})")
    scale_note = (
        f"auto-scaled from {REFERENCE_DIST_THRESHOLD_PX}px @ {REFERENCE_FRAME_WIDTH}px-wide reference "
        f"({100*REFERENCE_DIST_THRESHOLD_PX/REFERENCE_FRAME_WIDTH:.1f}% of frame width)"
        if auto_scaled else "MANUALLY OVERRIDDEN via --dist-threshold-px"
    )
    print(f"thresholds: dist_threshold_px={dist_threshold_px} ({scale_note})  "
          f"gap_frames_threshold={gap_frames_threshold}")
    print("=" * 70)

    print("\n--- Raw tracks seen (before any reconciliation) ---")
    for tid in sorted(track_first_seen_frame):
        start = track_first_seen_frame[tid]
        end = track_last_seen_frame[tid]
        print(f"  track_id={tid}  first_frame={start}  last_frame={end}  "
              f"duration={end - start} frames  last_real_pos={track_last_real_pos[tid]}")
    print(f"\n  unique_tracks_seen (raw) = {len(raw_ids_seen)}")

    print("\n--- Reconciler merge decisions ---")
    if not reconciler.merge_log:
        print("  No merges proposed.")
    else:
        for entry in reconciler.merge_log:
            print(f"  MERGED: {entry}")
    print(f"\n  unique_tracks_reconciled = {len(canonical_ids_seen)}")
    print(f"  tracks_merged = {len(reconciler.merge_log)}")

    summary = counter.summary()
    print("\n--- Counter output (raw IDs -- what the actual demo reports) ---")
    print(f"  IN={summary.get('in_count')}  OUT={summary.get('out_count')}  "
          f"events={summary.get('total_events')}  "
          f"(counter's own unique_tracks_seen={summary.get('unique_tracks_seen')})")

    print("\n" + "=" * 70)
    print("MANUAL VERIFICATION CHECKLIST -- fill in by watching the real footage")
    print("=" * 70)
    print(f"""
  [ ] Real people who crossed the line: _____
  [ ] Real IN crossings: _____   Real OUT crossings: _____
      -> Compare to IN/OUT printed above. Mismatch here is a counting bug,
         separate from reconciliation -- escalate immediately.

  For EACH merge printed above, watch both time ranges (around the old
  canonical track's last_frame and the new raw track's first_frame) in
  the footage and judge:

  [ ] Same physical person reappearing (pause/occlusion)?  -> correct merge.
  [ ] Two different real people who happened to be within
      dist_threshold_px ({dist_threshold_px}) and gap_frames_threshold
      ({gap_frames_threshold}) of each other?
      -> WRONG MERGE. This is exactly the failure mode under test.
         Record: actual_gap_frames=_____  actual_dist_px=_____
         Then check whether tightening the thresholds below these numbers
         would still correctly catch the single-person pause case from
         landscape_test_1.mp4 (176 frames) and landscape_test_2.mp4
         (164 frames). If tightening enough to exclude the false merge
         would also exclude those real cases, the reconciler's design
         (not just its thresholds) needs revisiting -- note that plainly.
""")

    return {
        "raw_unique_tracks": len(raw_ids_seen),
        "merge_log": reconciler.merge_log,
        "counter_summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual verification harness: reconciler behavior on real two-person footage"
    )
    parser.add_argument("video", help="Path to video with two real people crossing close together")
    parser.add_argument("--line-y", type=int, default=None,
                         help="Crossing line Y in px. Defaults to half the video's height, matching run_tracking_qa.py's fallback.")
    parser.add_argument("--dist-threshold-px", type=int, default=None,
                         help="Override the auto-scaled threshold. If omitted, it's computed "
                              "from this video's frame width to preserve the same relative "
                              "tolerance validated on landscape_test_1/2 (see threshold_scaling.py).")
    parser.add_argument("--gap-frames-threshold", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run(args.video, args.line_y, args.dist_threshold_px, args.gap_frames_threshold, args.device)
