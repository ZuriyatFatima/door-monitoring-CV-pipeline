"""
diagnose_fragmentation.py
Purpose: get REAL evidence for why track_buffer=300 did not reduce
unique_tracks_seen on landscape_test_1/2, instead of guessing.

What it does, frame by frame:
  - Runs the same model.track(...) call as run_tracking_qa.py (same
    tracker_yaml, same conf, same classes) so results are directly
    comparable to that run.
  - Logs every track ID's first-seen frame, last-seen frame, and the
    (x, y, w, h) box at each of those moments.
  - Whenever a NEW track ID appears within `gap_frames` frames of an
    OLD track ID disappearing, treats that as a candidate "fragmentation
    event" and saves three frames as evidence:
        <video>_evt<N>_before.jpg   -- last frame the old ID was seen
        <video>_evt<N>_after.jpg    -- first frame the new ID was seen
        <video>_evt<N>_gap.jpg      -- a frame from the middle of the gap
    Each saved frame has the relevant box(es) drawn on it with the
    track ID and confidence, so you can SEE whether it's the same
    person, how far the box moved, and how much the pose/occlusion
    changed -- not just infer it from numbers.
  - Writes a CSV with one row per candidate fragmentation event:
    old_id, new_id, old_last_frame, new_first_frame, gap_frames,
    center_distance_px (how far the predicted position would need to
    jump to match) -- this number is the direct evidence for whether
    match_thresh/motion-prediction is the real bottleneck (large jump
    = plausible bottleneck) or something else (small jump = ByteTrack
    SHOULD have matched these; something else is wrong).

Usage:
    python diagnose_fragmentation.py uploads/landscape_test_1.mp4
    python diagnose_fragmentation.py uploads/landscape_test_2.mp4

Requires ultralytics + opencv, run locally (not in this sandbox).
"""

import sys
import os
import csv
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GAP_FRAMES_THRESHOLD = 60  # candidate fragmentation if a new ID appears
                            # within this many frames of an old ID vanishing
                            # (~2s at 28fps -- generous, since we want to
                            # SEE all plausible candidates, not pre-filter)


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_fragmentation.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    import cv2
    from ultralytics import YOLO
    from modules.tracker import ensure_bytetrack_config

    tracker_yaml = ensure_bytetrack_config()
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        sys.exit(1)

    # track_id -> list of (frame_idx, cx, cy, w, h, conf)
    track_history = defaultdict(list)
    # keep every frame in memory is too costly for long videos, so instead
    # we do a first pass to find event windows, then a second pass to grab
    # just the frames we need. First pass:
    frame_idx = 0
    all_frames_boxes = {}  # frame_idx -> list of (track_id, cx, cy, w, h, conf)

    print(f"Pass 1/2: running tracker on {video_path} ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = model.track(
            frame, persist=True, conf=0.4, tracker=tracker_yaml,
            device="cpu", classes=[0], verbose=False,
        )[0]

        boxes_this_frame = []
        if r.boxes is not None and r.boxes.id is not None:
            xywh = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (cx, cy, bw, bh), track_id, conf in zip(xywh, ids, confs):
                track_history[int(track_id)].append(
                    (frame_idx, float(cx), float(cy), float(bw), float(bh), float(conf))
                )
                boxes_this_frame.append((int(track_id), float(cx), float(cy), float(bw), float(bh), float(conf)))

        all_frames_boxes[frame_idx] = boxes_this_frame
        frame_idx += 1

    cap.release()
    total_frames = frame_idx
    print(f"Done. {total_frames} frames, {len(track_history)} unique track IDs seen: {sorted(track_history.keys())}")

    # Build last-seen / first-seen per track
    track_last = {tid: hist[-1] for tid, hist in track_history.items()}   # (frame,cx,cy,w,h,conf)
    track_first = {tid: hist[0] for tid, hist in track_history.items()}

    # Find candidate fragmentation events: old track disappears, new track
    # appears within GAP_FRAMES_THRESHOLD frames afterward.
    events = []
    old_ids_by_last_frame = sorted(track_last.items(), key=lambda kv: kv[1][0])
    new_ids_by_first_frame = sorted(track_first.items(), key=lambda kv: kv[1][0])

    for old_id, (old_frame, ocx, ocy, ow, oh, oconf) in old_ids_by_last_frame:
        for new_id, (new_frame, ncx, ncy, nw, nh, nconf) in new_ids_by_first_frame:
            if new_id == old_id:
                continue
            gap = new_frame - old_frame
            if 0 < gap <= GAP_FRAMES_THRESHOLD:
                dist = math.hypot(ncx - ocx, ncy - ocy)
                events.append({
                    "old_id": old_id, "new_id": new_id,
                    "old_last_frame": old_frame, "new_first_frame": new_frame,
                    "gap_frames": gap, "center_distance_px": round(dist, 1),
                    "old_box": (ocx, ocy, ow, oh), "new_box": (ncx, ncy, nw, nh),
                })

    if not events:
        print("No candidate fragmentation events found (no old-ID-vanish -> "
              "new-ID-appear pair within the gap threshold). This would mean "
              "the tracks=2 result is NOT from one person splitting into two "
              "IDs -- re-check whether it's genuinely two different people, "
              "or a detection appearing/disappearing for another reason.")
        return

    print(f"\n{len(events)} candidate fragmentation event(s) found:")
    for e in events:
        print(f"  old_id={e['old_id']} (last seen frame {e['old_last_frame']}) -> "
              f"new_id={e['new_id']} (first seen frame {e['new_first_frame']}), "
              f"gap={e['gap_frames']} frames, center jumped {e['center_distance_px']}px")

    # Pass 2: re-open video and grab evidence frames for each event
    print("\nPass 2/2: extracting evidence frames ...")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    wanted_frames = set()
    for e in events:
        wanted_frames.add(e["old_last_frame"])
        wanted_frames.add(e["new_first_frame"])
        mid = (e["old_last_frame"] + e["new_first_frame"]) // 2
        wanted_frames.add(mid)

    grabbed = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted_frames:
            grabbed[frame_idx] = frame.copy()
        frame_idx += 1
        if len(grabbed) == len(wanted_frames):
            break
    cap.release()

    out_dir = "fragmentation_evidence"
    os.makedirs(out_dir, exist_ok=True)

    def draw_box(frame, cx, cy, w, h, label):
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    csv_rows = []
    for i, e in enumerate(events):
        before = grabbed.get(e["old_last_frame"])
        after = grabbed.get(e["new_first_frame"])
        mid_frame = (e["old_last_frame"] + e["new_first_frame"]) // 2
        gap_img = grabbed.get(mid_frame)

        if before is not None:
            b = draw_box(before.copy(), *e["old_box"], f"id={e['old_id']} frame={e['old_last_frame']}")
            cv2.imwrite(os.path.join(out_dir, f"{video_name}_evt{i}_before.jpg"), b)
        if after is not None:
            a = draw_box(after.copy(), *e["new_box"], f"id={e['new_id']} frame={e['new_first_frame']}")
            cv2.imwrite(os.path.join(out_dir, f"{video_name}_evt{i}_after.jpg"), a)
        if gap_img is not None:
            cv2.imwrite(os.path.join(out_dir, f"{video_name}_evt{i}_gap.jpg"), gap_img)

        csv_rows.append({k: v for k, v in e.items() if k not in ("old_box", "new_box")})

    csv_path = os.path.join(out_dir, f"{video_name}_events.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nEvidence frames saved to: {out_dir}/")
    print(f"Event summary CSV: {csv_path}")
    print("\nWhat to look at next:")
    print(" - Open the _before / _after / _gap image trio for each event.")
    print(" - Is it visibly the same person? If yes, fragmentation confirmed.")
    print(" - Look at center_distance_px: if it's large (person moved far / "
          "changed pose a lot during the gap), that supports match_thresh "
          "or motion-prediction being the real bottleneck, not buffer length.")
    print(" - If center_distance_px is SMALL and it's still the same person "
          "failing to match, that points to a different bug entirely "
          "(e.g. new_track_thresh too low, or conf dropping below 0.4 "
          "mid-occlusion so the person isn't even detected during the gap).")


if __name__ == "__main__":
    main()
