"""
check_track_coexistence.py (fixed)

Fix vs. the previous version: that version called model.track() with no
classes=[0] filter, no conf=0.4, and loaded bytetrack_custom.yaml directly
instead of via ensure_bytetrack_config(). Result: it tracked ALL COCO
classes (not just people) and found 5-8 track IDs per video, disagreeing
with run_tracking_qa.py / diagnose_fragmentation.py's verified 2 IDs per
video. This version copies diagnose_fragmentation.py's tracking call
and loop structure exactly, so its track IDs should match that script's
[1, 2] / [1, 3] output before you trust any coexistence conclusion below.

VERIFY FIRST: run this and confirm "Unique track IDs" below matches what
diagnose_fragmentation.py printed for the same video. If it doesn't match,
stop -- something else differs (e.g. modules/tracker.py's PersonTracker
uses a different model checkpoint or extra args not shown in the scripts
I've seen) and the coexistence numbers still can't be trusted.

Usage:
    python check_track_coexistence.py uploads/landscape_test_1.mp4
    python check_track_coexistence.py uploads/landscape_test_2.mp4
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GAP_FRAMES_THRESHOLD = 60  # same window diagnose_fragmentation.py uses, for direct comparison


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"ERROR: video not found at '{video_path}'")
        sys.exit(1)

    import cv2
    from ultralytics import YOLO
    from modules.tracker import ensure_bytetrack_config

    tracker_yaml = ensure_bytetrack_config()
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        sys.exit(1)

    # track_id -> list of (frame_idx, cx, cy)
    track_frames = defaultdict(list)

    frame_idx = 0
    print(f"Running tracker on {video_path} ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Same call as diagnose_fragmentation.py / run_tracking_qa.py
        r = model.track(
            frame, persist=True, conf=0.4, tracker=tracker_yaml,
            device="cpu", classes=[0], verbose=False,
        )[0]

        if r.boxes is not None and r.boxes.id is not None:
            xywh = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (cx, cy, w, h), track_id in zip(xywh, ids):
                track_frames[int(track_id)].append((frame_idx, float(cx), float(cy)))

        frame_idx += 1

    cap.release()
    total_frames = frame_idx
    track_ids = sorted(track_frames.keys())

    print(f"\n=== {video_path} ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Unique track IDs: {track_ids}")
    print("^^ COMPARE THIS LINE to diagnose_fragmentation.py's output for the same "
          "video before trusting anything below.\n")

    for tid in track_ids:
        frames = [f for f, _, _ in track_frames[tid]]
        print(f"  Track {tid}: first_seen={min(frames)}, last_seen={max(frames)}, "
              f"n_appearances={len(frames)}")

    if len(track_ids) < 2:
        print("Fewer than 2 track IDs -- nothing to compare.")
        return

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            a, b = track_ids[i], track_ids[j]
            frames_a = {f: (cx, cy) for f, cx, cy in track_frames[a]}
            frames_b = {f: (cx, cy) for f, cx, cy in track_frames[b]}
            overlap_frames = sorted(set(frames_a) & set(frames_b))

            print(f"\n  --- Track {a} vs Track {b} ---")
            if overlap_frames:
                dists = []
                for f in overlap_frames:
                    ax, ay = frames_a[f]
                    bx, by = frames_b[f]
                    dists.append(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
                avg_dist = sum(dists) / len(dists)
                print(f"  COEXIST: {len(overlap_frames)} frame(s) of overlap "
                      f"(frames {overlap_frames[0]}-{overlap_frames[-1]}).")
                print(f"  Avg bbox-center distance during overlap: {avg_dist:.1f}px")
                print("  -> Large distance: two genuinely separate detections "
                      "(consistent with 2 real people/objects).")
                print("  -> Small distance: overlapping/duplicate boxes on the "
                      "same person (a different bug, not the fragmentation theory).")
            else:
                last_a = max(frames_a)
                first_b = min(frames_b)
                last_b = max(frames_b)
                first_a = min(frames_a)
                gap = (first_b - last_a) if first_b > last_a else (first_a - last_b)
                # Position distance between the disappearance point and the
                # reappearance point -- direct evidence for whether this is
                # "same spot, brief miss" (buffer-fixable) vs "left frame,
                # returned elsewhere" (not buffer-fixable, ByteTrack has no
                # appearance re-ID to bridge this).
                if first_b > last_a:
                    px, py = frames_a[last_a]
                    qx, qy = frames_b[first_b]
                else:
                    px, py = frames_b[last_b]
                    qx, qy = frames_a[first_a]
                reentry_dist = ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
                print(f"  NO OVERLAP. Gap: {gap} frames.")
                print(f"  Position distance (disappear point -> reappear point): {reentry_dist:.1f}px")
                if reentry_dist > 150:
                    print("  -> Reappeared FAR from where it vanished -- consistent with "
                          "genuinely leaving frame and re-entering elsewhere. track_buffer "
                          "size would NOT fix this (ByteTrack has no appearance re-ID; "
                          "position-based matching can't bridge a location change).")
                else:
                    print("  -> Reappeared CLOSE to where it vanished -- if this is really "
                          "the same person, this is more consistent with a brief occlusion "
                          "at roughly the same spot, which track_buffer/match_thresh tuning "
                          "could plausibly still help with.")
                if gap <= GAP_FRAMES_THRESHOLD:
                    print(f"  -> Gap is WITHIN diagnose_fragmentation.py's {GAP_FRAMES_THRESHOLD}-frame "
                          "window. If these track IDs also matched [1,2]/[1,3] exactly, "
                          "diagnose_fragmentation.py should have flagged this pair as a "
                          "candidate event -- if it didn't, that script's event-matching "
                          "logic needs a direct re-check against this result.")
                else:
                    print(f"  -> Gap exceeds the {GAP_FRAMES_THRESHOLD}-frame window, so "
                          "diagnose_fragmentation.py correctly would not flag it as a "
                          "candidate fragmentation event.")


if __name__ == "__main__":
    main()
