"""
gap_threshold_sweep.py

Why this exists: the threshold_scaling.py fix tightened dist_threshold_px
(150px -> 63.7px on 360-wide footage) but produced almost no change in
merge rate (passageway: 8/27 reconciled/merged, byte-identical to before;
terrace: 16/163, barely different from 10/169). Looking at the actual
merge_log distance_px values, nearly all real merges were already well
under 63.7px -- meaning distance was never the loose constraint letting
false merges through.

gap_frames_threshold=300 (~12 seconds) was NOT touched by that fix, and
many observed merges have short gaps (2-60 frames) -- consistent with
ordinary foot traffic at a chokepoint, not a buggy pause. This script
gets direct evidence on whether narrowing the gap window actually helps,
instead of guessing at a new value.

APPROACH (efficient -- one YOLO pass, many cheap replays):
  1. Run PersonTracker ONCE across the whole video, caching every
     (frame_idx, track_id, x, y) detection to a list. This is the
     expensive step (real inference) -- done only once.
  2. For each candidate gap_frames_threshold in a sweep list, construct
     a FRESH TrackReconciler (dist_threshold_px fixed at the already-
     scaled value) and replay the cached detection stream through
     .reconcile() in order. This is cheap (no inference), so many
     thresholds can be tested in seconds.
  3. Print a comparison table: for each gap_frames_threshold, how many
     unique canonical IDs and how many merges resulted.

This directly shows whether tightening the gap window reduces merges
meaningfully, and -- critically -- whether a tight-enough window to fix
the passageway/terrace over-merging would ALSO be tight enough to still
exclude the real single-person-pause gaps from landscape_test_1.mp4 (176
frames) and landscape_test_2.mp4 (164 frames). If the sweep shows no
value below ~200 frames meaningfully helps AND values above 176 are
needed to keep the real single-pause case working, that's direct
evidence the reconciler's whole gap+distance heuristic can't cleanly
separate "same person paused" from "different person, same spot" in a
busy scene -- a design-level finding, not a tuning one.

Usage:
    python gap_threshold_sweep.py passageway1-c0.avi
    python gap_threshold_sweep.py terrace1-c0.avi
    python gap_threshold_sweep.py passageway1-c0.avi --gaps 30,60,90,120,150,176,200,250,300
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from modules.tracker import PersonTracker
from modules.track_reconciler import TrackReconciler
from threshold_scaling import scaled_dist_threshold_px

PERSON_MODEL_PATH = "yolo11n.pt"
DEFAULT_GAP_SWEEP = [30, 60, 90, 120, 150, 164, 176, 200, 250, 300]


def collect_detections(video_path, device="cpu"):
    """Run the tracker ONCE, return (width, height, list of (frame_idx, track_id, x, y))."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    person_tracker = PersonTracker(model_path=PERSON_MODEL_PATH, device=device)

    detections = []  # (frame_idx, track_id, x, y) in order
    frame_idx = 0
    print(f"Running tracker ONCE on {video_path} to cache the detection stream ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = person_tracker.track_frame(frame)
        if result is not None and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for (x, y, bw, bh), track_id in zip(boxes, ids):
                detections.append((frame_idx, int(track_id), float(x), float(y)))

        frame_idx += 1

    cap.release()
    print(f"Done. {frame_idx} frames processed, {len(detections)} detections cached.\n")
    return w, h, detections


def sweep(detections, dist_threshold_px, gap_values):
    """Replay the cached detection stream through a fresh TrackReconciler
    for each gap_frames_threshold value. Cheap -- no inference, just the
    reconciler's own bookkeeping."""
    results = []
    for gap in gap_values:
        reconciler = TrackReconciler(gap_frames_threshold=gap, dist_threshold_px=dist_threshold_px)
        canonical_ids_seen = set()
        for frame_idx, track_id, x, y in detections:
            canonical_id = reconciler.reconcile(frame_idx, track_id, x, y)
            canonical_ids_seen.add(canonical_id)
        results.append({
            "gap_frames_threshold": gap,
            "unique_tracks_reconciled": len(canonical_ids_seen),
            "tracks_merged": len(reconciler.merge_log),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--gaps", default=None,
                         help="Comma-separated list of gap_frames_threshold values to test. "
                              f"Default: {DEFAULT_GAP_SWEEP}")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    gap_values = (
        [int(x) for x in args.gaps.split(",")] if args.gaps else DEFAULT_GAP_SWEEP
    )

    w, h, detections = collect_detections(args.video, args.device)
    raw_unique = len(set(tid for _, tid, _, _ in detections))
    dist_threshold_px = scaled_dist_threshold_px(w)

    print(f"=== Gap threshold sweep: {args.video} ({w}x{h}) ===")
    print(f"dist_threshold_px held constant at {dist_threshold_px} (auto-scaled)")
    print(f"raw unique_tracks_seen (no reconciliation) = {raw_unique}\n")

    results = sweep(detections, dist_threshold_px, gap_values)

    print(f"{'gap_frames_threshold':<22} {'unique_tracks_reconciled':<26} {'tracks_merged'}")
    print("-" * 65)
    for r in results:
        note = ""
        if r["gap_frames_threshold"] in (164, 176):
            note = "  <- real single-person-pause gap (landscape_test_1/2)"
        print(f"{r['gap_frames_threshold']:<22} {r['unique_tracks_reconciled']:<26} "
              f"{r['tracks_merged']}{note}")

    print(
        "\nWhat to look for:\n"
        "  - Does unique_tracks_reconciled stay roughly flat as gap_frames_threshold\n"
        "    drops, even down near 30-60 frames? If YES, gap length isn't the driver\n"
        "    either, and dist_threshold_px (or the whole heuristic) needs a harder look.\n"
        "  - Does it drop sharply only once gap_frames_threshold goes below ~164-176?\n"
        "    If YES, that's the real evidence the reconciler CANNOT be tuned to solve\n"
        "    both problems at once with a single global threshold -- fixing this busy-\n"
        "    scene over-merging would break the real single-person-pause case it was\n"
        "    built for, and the reconciler's design (not just its thresholds) needs\n"
        "    revisiting, e.g. requiring BOTH a short gap AND close distance instead of\n"
        "    treating them as independently-sufficient conditions, or dropping position-\n"
        "    only reconciliation in busy scenes entirely."
    )


if __name__ == "__main__":
    main()
