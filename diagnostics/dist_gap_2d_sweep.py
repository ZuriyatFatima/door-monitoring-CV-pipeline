"""
dist_gap_2d_sweep.py

F6 investigation — 2D parameter sweep for TrackReconciler.

WHY THIS SCRIPT EXISTS
-----------------------
Prior evidence (this same investigation):
  - Scaling dist_threshold_px for resolution did NOT reduce over-merging
    (byte-identical / near-identical results on passageway & terrace).
  - Sweeping gap_frames_threshold ALONE, from 300 down to 30, reduced merges
    gradually but NEVER approached zero false merging, even at gap=30
    (~1 second) — terrace still merged 130/179 raw tracks.
  - Conclusion so far: neither axis alone can be tuned to fix busy-scene
    over-merging. This script tests the last remaining hypothesis: does
    some *combination* of a much smaller distance threshold WITH the real
    needed gap (164-176 frames, the actual single-person-pause gap on the
    landscape footage) avoid busy-scene merges while still being loose
    enough to catch that real pause?

KEY DESIGN CHOICE
------------------
We do NOT sweep gap freely again — the earlier gap-only sweep already
proved gap must stay >=164-176 frames or it stops catching the real
single-person-pause case on landscape_test_1/2. So this script fixes gap
to the two real candidate values (164 and 176) and instead sweeps distance
finely underneath them. This directly answers the open question from the
"what we do next" list: can distance alone, once gap is pinned to what the
real use case needs, kill the busy-scene merges?

If NO combination in this grid gets busy-scene merge fraction close to zero
while gap stays >=164, that settles it: this is a design-level limitation
(position+time-only matching can't distinguish different people sharing a
walking path), not a tuning problem, and F6 should be scoped as validated
for low-traffic single/two-person scenes only.

USAGE
-----
    python dist_gap_2d_sweep.py --video passageway1-c0.avi --model yolo11n.pt
    python dist_gap_2d_sweep.py --video terrace1-c0.avi --model yolo11n.pt

Run once per stand-in video (same as gap_threshold_sweep.py did). Paste
back the printed grid + verdict for both videos.
"""

import argparse
import time
from pathlib import Path

import cv2

# --- these come from the real, confirmed call-site interfaces ---
# NOTE: internals of TrackReconciler have still never been read directly.
# Everything below only calls its public interface as confirmed from
# real call sites: reconcile(frame_idx, track_id, x, y) -> canonical_id,
# .merge_log, .summary().
from modules.tracker import PersonTracker
from modules.track_reconciler import TrackReconciler


def cache_raw_detection_stream(video_path: str, model_path: str, device: str = "cpu"):
    """
    Run YOLO tracking ONCE over the whole video and cache every
    (frame_idx, track_id, x, y) detection. This is the expensive step
    (inference) — we do it exactly once per video, then replay the cached
    stream through many cheap fresh TrackReconciler instances below.

    Returns: list of (frame_idx, track_id, x, y) tuples, in frame order.
    """
    tracker = PersonTracker(model_path=model_path, device=device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    stream = []
    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = tracker.track_frame(frame)

        # Guard: some frames may have zero detections, or ids may be None
        # (tracker hasn't confirmed a track yet) — skip those safely rather
        # than crashing the whole cache pass.
        if result.boxes is not None and result.boxes.id is not None:
            xywh = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for (x, y, w, h), track_id in zip(xywh, ids):
                stream.append((frame_idx, int(track_id), float(x), float(y)))

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"[cache] {video_path}: {frame_idx} frames, "
          f"{len(stream)} detections cached in {elapsed:.1f}s "
          f"(inference run ONCE)")
    return stream


def replay_with_params(stream, dist_threshold_px: float, gap_frames_threshold: int):
    """
    Cheap replay: feed the cached (frame_idx, track_id, x, y) stream through
    a FRESH TrackReconciler at the given thresholds. No re-inference.

    Returns the reconciler's .summary() dict, which is expected to include
    at least unique_tracks_reconciled-style counts — we read whatever keys
    it actually returns rather than assuming a fixed shape, and also count
    merges directly off .merge_log for a threshold-agnostic cross-check.
    """
    reconciler = TrackReconciler(
        gap_frames_threshold=gap_frames_threshold,
        dist_threshold_px=dist_threshold_px,
    )
    canonical_ids_seen = set()
    for frame_idx, track_id, x, y in stream:
        canonical_id = reconciler.reconcile(frame_idx, track_id, x, y)
        canonical_ids_seen.add(canonical_id)

    n_merges = len(reconciler.merge_log)
    n_reconciled = len(canonical_ids_seen)
    return {
        "dist_threshold_px": dist_threshold_px,
        "gap_frames_threshold": gap_frames_threshold,
        "unique_tracks_reconciled": n_reconciled,
        "merges": n_merges,
        "raw_track_ids": len({tid for _, tid, _, _ in stream}),
    }


def main():
    parser = argparse.ArgumentParser(description="F6: 2D dist x gap sweep")
    parser.add_argument("--video", required=True, help="Path to stand-in test video")
    parser.add_argument("--model", required=True, help="Path to YOLO model weights")
    parser.add_argument("--device", default="cpu")
    # Fine distance grid, well below the 63.7px scaled threshold that
    # already failed to fix things, down to very tight values. If even
    # the tightest distance here (still generous enough to be physically
    # plausible for one walking person) fails, distance is not the lever.
    parser.add_argument(
        "--dist-values",
        type=float,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 40, 50, 63.7],
        help="Distance thresholds (px, already resolution-scaled) to test",
    )
    # Only the two real candidate gap values from the landscape footage
    # (164 and 176 frames) — see docstring for why we don't re-sweep gap
    # freely here.
    parser.add_argument(
        "--gap-values",
        type=int,
        nargs="+",
        default=[164, 176],
        help="Gap thresholds (frames) to test -- pinned to real single-pause range",
    )
    args = parser.parse_args()

    video_name = Path(args.video).stem
    stream = cache_raw_detection_stream(args.video, args.model, args.device)

    print(f"\n=== 2D sweep results for {video_name} ===")
    header = f"{'dist_px':>8} {'gap':>6} {'reconciled':>11} {'merges':>7} {'merge_%':>8}"
    print(header)
    print("-" * len(header))

    results = []
    raw_count = len({tid for _, tid, _, _ in stream})
    for gap in args.gap_values:
        for dist in args.dist_values:
            r = replay_with_params(stream, dist, gap)
            merge_pct = 100.0 * r["merges"] / max(raw_count, 1)
            print(f"{dist:>8.1f} {gap:>6d} {r['unique_tracks_reconciled']:>11d} "
                  f"{r['merges']:>7d} {merge_pct:>7.1f}%")
            results.append((dist, gap, r["merges"], merge_pct))

    # Verdict: flag any combo that gets merge_% below a "plausibly safe"
    # bar (5%) while keeping gap at a real candidate value. This is a
    # signal to go inspect by eye, NOT an auto pass/fail -- same
    # non-judgmental philosophy as verify_two_person_scene.py.
    print(f"\n=== Candidates with <5% merge rate (raw_track_ids={raw_count}) ===")
    candidates = [r for r in results if r[3] < 5.0]
    if candidates:
        for dist, gap, merges, pct in candidates:
            print(f"  dist={dist} gap={gap} -> {merges} merges ({pct:.1f}%)")
        print("\n  ^ Inspect these merge_logs by hand before trusting them -- "
              "low merge count on a busy scene could also mean the "
              "reconciler is now too strict to catch the real single-pause "
              "case. Re-run against landscape_test_1/2 with these exact "
              "params to confirm the real pause still merges correctly "
              "before adopting any of these.")
    else:
        print("  NONE. No (distance, gap) combination in this grid got "
              "busy-scene merging below 5%, even at the tightest distance "
              "tested and with gap pinned to the real single-pause range.\n"
              "  This settles the open question from the F6 investigation: "
              "distance and gap together (not just individually) cannot be "
              "tuned to fix this. It is a design-level limitation of "
              "position+time-only reconciliation on busy multi-person "
              "scenes, not a threshold problem. Next step per the "
              "investigation plan: scope F6 as validated-safe only for "
              "low-traffic single/two-person scenes matching the real "
              "project footage, and document the busy-scene failure mode "
              "as a known limitation rather than leaving it unmentioned.")


if __name__ == "__main__":
    main()
