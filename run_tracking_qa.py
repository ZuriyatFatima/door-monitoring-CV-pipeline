"""
run_tracking_qa.py
Week 7 Day 2 deliverable: runs person tracking/counting across every
video in a folder and writes one structured QA log -- instead of
manually running the dashboard once per video and eyeballing results.

Flags anomalies automatically:
  - zero events (either genuinely no crossings, or a possible bug)
  - a crash on a specific video (caught and logged, doesn't kill the
    whole run for the remaining videos)

Usage:
    python run_tracking_qa.py [uploads_folder]

Defaults to the uploads/ folder next to this script. Writes
qa_results/tracking_qa_<timestamp>.csv with one row per video.
"""

import sys
import os
import glob
import csv
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO_EXTS = (".mp4", ".avi", ".mov")


def run_one_video(video_path, person_model, tracker_yaml, line_y):
    """
    Runs the full person-tracking/counting pipeline on one video.
    Returns a result dict. Never raises -- catches its own errors so
    one bad video doesn't stop the QA run for the rest.
    """
    import cv2
    from modules.counter import LineCounter
    from modules.track_reconciler import TrackReconciler

    result = {
        "video": os.path.basename(video_path),
        "status": "ok",
        "error": "",
        "width": None, "height": None, "fps_video": None, "frames": None,
        "in_count": None, "out_count": None, "total_events": None,
        "unique_tracks_seen": None, "processing_fps": None,
        "unique_tracks_reconciled": None, "tracks_merged": None,
        "flag": "",
    }

    try:
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result.update({"width": w, "height": h, "fps_video": round(fps_video, 1), "frames": total_frames})

        counter = LineCounter(line_y=line_y, min_hits=5)
        # Week 8 fix for F6 (Kalman motion-prediction drift) -- see
        # modules/track_reconciler.py and failure_log.md F6.
        #
        # IMPORTANT: the reconciler's canonical_id is used ONLY to compute a
        # corrected track count for reporting. It is deliberately NOT passed
        # to counter.update() -- an earlier version did that and it silently
        # merged genuinely separate crossings (confirmed on test_1.mp4: OUT
        # dropped 2->1 and total_events dropped 8->7, breaking the regression
        # baseline). counter.update() always receives the RAW track_id, so
        # IN/OUT/total_events are guaranteed identical to the pre-fix numbers.
        reconciler = TrackReconciler(gap_frames_threshold=300, dist_threshold_px=150)
        raw_ids_seen = set()
        canonical_ids_seen = set()
        frame_idx = 0
        start = time.time()

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
                    raw_ids_seen.add(int(track_id))
                    canonical_id = reconciler.reconcile(frame_idx, int(track_id), float(x), float(y))
                    canonical_ids_seen.add(canonical_id)
                    counter.update(frame_idx, int(track_id), float(y))

            frame_idx += 1

        elapsed = time.time() - start
        cap.release()

        summary = counter.summary()
        result.update({
            "in_count": summary["in_count"],
            "out_count": summary["out_count"],
            "total_events": summary["total_events"],
            "unique_tracks_seen": summary["unique_tracks_seen"],
            "unique_tracks_reconciled": len(canonical_ids_seen),
            "tracks_merged": len(reconciler.merge_log),
            "processing_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
        })

        if summary["total_events"] == 0:
            result["flag"] = "ZERO EVENTS -- check if this is expected (no lateral crossing in this footage) or a bug"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["flag"] = "CRASHED -- see error column, and full traceback was printed to console"
        print(f"\n--- Full traceback for {os.path.basename(video_path)} ---")
        traceback.print_exc()
        print("--- end traceback ---\n")

    return result


def main():
    uploads_dir = sys.argv[1] if len(sys.argv) > 1 else "uploads"
    line_y = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.isdir(uploads_dir):
        print(f"ERROR: '{uploads_dir}' is not a folder.")
        sys.exit(1)

    videos = sorted(
        p for ext in VIDEO_EXTS for p in glob.glob(os.path.join(uploads_dir, f"*{ext}"))
    )
    if not videos:
        print(f"No videos found in '{uploads_dir}'.")
        sys.exit(1)

    from ultralytics import YOLO
    from modules.tracker import ensure_bytetrack_config

    print("Loading yolo11n.pt ...")
    person_model = YOLO("yolo11n.pt")
    tracker_yaml = ensure_bytetrack_config()

    os.makedirs("qa_results", exist_ok=True)
    out_path = os.path.join("qa_results", f"tracking_qa_{datetime.now():%Y%m%d_%H%M%S}.csv")

    results = []
    for video_path in videos:
        print(f"\nRunning: {os.path.basename(video_path)} ...")

        # Per-video line_y: use override if given, else a sensible guess
        # based on this video's own height (roughly middle of frame).
        import cv2
        cap = cv2.VideoCapture(video_path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        this_line_y = line_y if line_y is not None else h // 2

        r = run_one_video(video_path, person_model, tracker_yaml, this_line_y)
        r["line_y_used"] = this_line_y
        results.append(r)

        status_str = "OK" if r["status"] == "ok" else "ERROR"
        merged_note = f" (unique_tracks_seen={r['unique_tracks_seen']}, reconciled={r['unique_tracks_reconciled']}, merged={r['tracks_merged']})" if r["tracks_merged"] else ""
        print(f"  [{status_str}] {r['width']}x{r['height']}  IN={r['in_count']} OUT={r['out_count']} "
              f"events={r['total_events']} tracks={r['unique_tracks_seen']}{merged_note}  fps={r['processing_fps']}"
              + (f"  FLAG: {r['flag']}" if r['flag'] else ""))

    fieldnames = ["video", "status", "width", "height", "fps_video", "frames", "line_y_used",
                  "in_count", "out_count", "total_events", "unique_tracks_seen",
                  "unique_tracks_reconciled", "tracks_merged",
                  "processing_fps", "flag", "error"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"QA log written to: {out_path}")
    flagged = [r for r in results if r["flag"]]
    if flagged:
        print(f"\n{len(flagged)} video(s) flagged for review:")
        for r in flagged:
            print(f"  {r['video']}: {r['flag']}")
    else:
        print("\nNo anomalies flagged.")


if __name__ == "__main__":
    main()
