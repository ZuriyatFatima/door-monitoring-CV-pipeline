"""
debug_f6_trace.py
Week 8 F6 diagnosis: frame-by-frame trace of one video's raw detections,
run through the SAME LineCounter + TrackReconciler pipeline as
run_tracking_qa.py, to find out exactly why landscape_test_1.mp4 gives
IN=0 / OUT=2 instead of the ground-truth IN=1 / OUT=1.

This does NOT change any pipeline logic. It just instruments it:
  - logs every raw track_id's first/last frame, hit count, and every
    y-center it produced
  - logs every CrossingEvent the real LineCounter fires (frame, track_id,
    direction, y)
  - logs the reconciler's full merge_log
  - explicitly checks, per raw track_id, whether its OWN last y and the
    NEXT track_id's OWN first y straddle line_y (an "invisible crossing"
    that no single track_id's own prev_y/y_center pair can ever catch)
  - checks whether counted OUT events cluster within a tight frame window
    (jitter near the line) vs. being spread far apart

Usage:
    python debug_f6_trace.py <video_path> [line_y]

Defaults line_y to the same "frame height // 2" convention
run_tracking_qa.py uses when no override is given.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_f6_trace.py <video_path> [line_y]")
        sys.exit(1)

    video_path = sys.argv[1]
    line_y_override = int(sys.argv[2]) if len(sys.argv) > 2 else None

    import cv2
    from ultralytics import YOLO
    from modules.tracker import ensure_bytetrack_config
    from modules.counter import LineCounter
    from modules.track_reconciler import TrackReconciler

    if not os.path.isfile(video_path):
        print(f"ERROR: '{video_path}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    line_y = line_y_override if line_y_override is not None else h // 2

    print(f"Video: {video_path}")
    print(f"Frame size: {w}x{h}, line_y = {line_y}")
    print("=" * 70)

    print("Loading yolo11n.pt ...")
    person_model = YOLO("yolo11n.pt")
    tracker_yaml = ensure_bytetrack_config()

    counter = LineCounter(line_y=line_y, min_hits=5)
    reconciler = TrackReconciler(gap_frames_threshold=300, dist_threshold_px=150)

    # raw_track_id -> dict(first_frame, last_frame, hit_count, ys=[(frame,y), ...])
    track_log = {}

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = person_model.track(
            frame, persist=True, conf=0.4, tracker=tracker_yaml,
            device="cpu", classes=[0], verbose=False,
        )[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (x, y, bw, bh), track_id in zip(boxes, ids):
                tid = int(track_id)
                yc = float(y)

                if tid not in track_log:
                    track_log[tid] = {
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "hit_count": 0,
                        "ys": [],
                    }
                track_log[tid]["last_frame"] = frame_idx
                track_log[tid]["hit_count"] += 1
                track_log[tid]["ys"].append((frame_idx, yc))

                # NOTE: this used to call counter.bridge() to propagate
                # counted_ids/prev_y/hit_counts across reconciler-confirmed
                # merges (Fix 7). Fix 7 was reverted after breaking the
                # test_1.mp4 regression baseline and landscape_test_2.mp4 --
                # see bug_fixes.md Fix 7 / failure_log.md F6. bridge() no
                # longer exists on LineCounter, so this trace now runs the
                # reconciler purely as a diagnostic (its merge_log is still
                # inspected in Section 3 below) without feeding it into the
                # actual counting logic, matching current shipped behavior.
                reconciler.reconcile(frame_idx, tid, float(x), yc)
                counter.update(frame_idx, tid, yc)

        frame_idx += 1

    cap.release()

    # ---- Section 1: raw track_id timeline ----
    print("\n--- RAW TRACK_ID TIMELINE ---")
    print(f"{'track_id':>9} {'first_frame':>12} {'last_frame':>11} {'hits':>5} "
          f"{'first_y':>9} {'last_y':>9}")
    ordered_ids = sorted(track_log.keys(), key=lambda t: track_log[t]["first_frame"])
    for tid in ordered_ids:
        info = track_log[tid]
        first_y = info["ys"][0][1]
        last_y = info["ys"][-1][1]
        print(f"{tid:>9} {info['first_frame']:>12} {info['last_frame']:>11} "
              f"{info['hit_count']:>5} {first_y:>9.1f} {last_y:>9.1f}")

    # ---- Section 2: real CrossingEvents from the real LineCounter ----
    print("\n--- COUNTER CROSSING EVENTS (what run_tracking_qa.py actually counts) ---")
    if not counter.events:
        print("  (none)")
    for ev in counter.events:
        print(f"  frame={ev.frame_idx:>5}  track_id={ev.track_id:>4}  "
              f"direction={ev.direction:<3}  y={ev.y_position:.1f}")
    summary = counter.summary()
    print(f"\n  in_count={summary['in_count']}  out_count={summary['out_count']}  "
          f"total_events={summary['total_events']}  unique_tracks_seen={summary['unique_tracks_seen']}")

    # ---- Section 3: reconciler merge log ----
    print("\n--- RECONCILER MERGE LOG (diagnostic only, NOT fed into counter) ---")
    if not reconciler.merge_log:
        print("  (no merges)")
    for m in reconciler.merge_log:
        print(f"  new_id={m['new_id']:>4} -> merged_into={m['merged_into']:>4}  "
              f"frame={m['frame']:>5}  gap_frames={m['gap_frames']:>4}  "
              f"distance_px={m['distance_px']}")

    # ---- Section 4: cross-fragment "invisible crossing" check ----
    print("\n--- CROSS-FRAGMENT BOUNDARY CHECK ---")
    print("For each consecutive pair of raw track_ids (by first_frame order),")
    print("checks whether track A's LAST y and track B's FIRST y straddle line_y.")
    print("If so, that crossing is invisible to LineCounter: neither fragment's")
    print("own prev_y/y_center pair ever contains both sides of the line.\n")
    any_invisible = False
    for a, b in zip(ordered_ids, ordered_ids[1:]):
        a_last_frame, a_last_y = track_log[a]["ys"][-1]
        b_first_frame, b_first_y = track_log[b]["ys"][0]
        gap = b_first_frame - a_last_frame
        straddles = (a_last_y < line_y <= b_first_y) or (a_last_y > line_y >= b_first_y)
        if straddles:
            any_invisible = True
            direction = "DOWN (would be IN)" if a_last_y < b_first_y else "UP (would be OUT)"
            print(f"  track {a} (last y={a_last_y:.1f} @ frame {a_last_frame})  ->  "
                  f"track {b} (first y={b_first_y:.1f} @ frame {b_first_frame})  "
                  f"gap={gap} frames  IMPLIED DIRECTION: {direction}  <-- INVISIBLE CROSSING")
    if not any_invisible:
        print("  (none found -- no cross-fragment boundary straddles line_y)")

    # ---- Section 5: OUT-event frame clustering (jitter check) ----
    print("\n--- OUT EVENT FRAME SPACING (jitter-near-line check) ---")
    out_events = [ev for ev in counter.events if ev.direction == "OUT"]
    if len(out_events) < 2:
        print("  (fewer than 2 OUT events, nothing to compare)")
    else:
        for e1, e2 in zip(out_events, out_events[1:]):
            print(f"  frame {e1.frame_idx} (track {e1.track_id}, y={e1.y_position:.1f})  ->  "
                  f"frame {e2.frame_idx} (track {e2.track_id}, y={e2.y_position:.1f})  "
                  f"gap={e2.frame_idx - e1.frame_idx} frames")

    print("\n--- GROUND TRUTH REMINDER ---")
    print("  Expected: IN=1, OUT=1 (person enters, pauses, exits closing the door)")
    print("  This trace runs the CURRENT shipped pipeline: raw per-track counting,")
    print("  reconciler diagnostic-only (not fed into counter.update()). The")
    print("  Section 3 merge log above shows what the reconciler WOULD merge if")
    print("  it were wired in -- it does not affect the counts in Section 2.")
    print("  IN=0 is a SEPARATE unresolved bug (entry crossing never detected")
    print("  before the first raw track_id even appears), unrelated to F6/F11.")


if __name__ == "__main__":
    main()
