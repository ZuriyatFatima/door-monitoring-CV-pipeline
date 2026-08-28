"""
verify_chokepoint_hypothesis.py

Purpose: confirm/refute the NEW explanation for the F6 merge floor, now that
track_reconciler.py has been read directly and confirmed to use AND logic
(gap check AND distance check) -- not the OR logic previously inferred
from behavior alone.

New theory: the residual merges at dist_threshold_px=5 (from
dist_gap_2d_sweep.py's grid) aren't caused by loose matching -- they're
caused by different real people legitimately passing within a few pixels
of each other's last position, because busy footage funnels foot traffic
through a narrow spatial band (a chokepoint) regardless of identity.

This does NOT rely on LINE_Y -- dist_gap_2d_sweep.py never set one for the
stand-in videos, and the chokepoint theory doesn't need it: if merges
cluster tightly relative to the FRAME size (not relative to a specific
line), that alone supports "different people funnel through the same
small area", independent of where any line is drawn.

Non-invasive: subclasses TrackReconciler unchanged (no edits to the locked
file) purely to also capture (cx, cy) at merge time, since the shipped
merge_log only stores distance_px.

Reuses cache_raw_detection_stream() from dist_gap_2d_sweep.py directly --
same caching, so results are directly comparable to that script's grid.

--- UPDATE (this session) ---
Chokepoint theory (above) was REFUTED: merge points spanned 52.6% of frame
width (passageway) and 96.7% width / 27.2% height (terrace) -- scattered,
not clustered.

New theory #3, raised by a follow-on observation (results were byte-
identical between gap=164 and gap=176 on both videos -- a wider gap window
picked up zero additional merges): the real gap_frames on these false
merges is likely well under 164, meaning the failure mode is short-gap
coincidental proximity in a busy scene, independent of the long real-pause
gap window entirely. check_chokepoint() below now also prints the actual
gap_frames value per merge (already present on every merge_log entry from
TrackReconciler.reconcile() itself -- no reconciler change needed) so this
can be confirmed or refuted directly from real output, no re-run of the
tracker/inference required beyond what this script already does.

USAGE
-----
    python verify_chokepoint_hypothesis.py --video passageway1-c0.avi --model yolo11n.pt
    python verify_chokepoint_hypothesis.py --video terrace1-c0.avi --model yolo11n.pt

Run once per stand-in video, same pattern as dist_gap_2d_sweep.py. Paste
back both printed results.
"""

import argparse
import statistics
from pathlib import Path

import cv2

from modules.track_reconciler import TrackReconciler
from dist_gap_2d_sweep import cache_raw_detection_stream


class InstrumentedReconciler(TrackReconciler):
    """Identical behavior to TrackReconciler -- only adds new_cx/new_cy to
    each merge_log entry so we can check spatial clustering after the fact.
    gap_frames and distance_px are already present on every merge_log entry
    from the base class -- nothing extra needed for those."""

    def reconcile(self, frame_idx, raw_track_id, cx, cy):
        n_before = len(self.merge_log)
        canonical_id = super().reconcile(frame_idx, raw_track_id, cx, cy)
        if len(self.merge_log) > n_before:
            self.merge_log[-1]["new_cx"] = cx
            self.merge_log[-1]["new_cy"] = cy
        return canonical_id


def get_frame_dimensions(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def check_chokepoint(stream, gap_frames_threshold: int, dist_threshold_px: float,
                      frame_w: int, frame_h: int, video_name: str = ""):
    r = InstrumentedReconciler(
        gap_frames_threshold=gap_frames_threshold,
        dist_threshold_px=dist_threshold_px,
    )
    for frame_idx, track_id, x, y in stream:
        r.reconcile(frame_idx, track_id, x, y)

    merges = r.merge_log
    label = f"{video_name} " if video_name else ""
    print(f"\n--- {label}dist={dist_threshold_px}px gap={gap_frames_threshold}f "
          f"(frame {frame_w}x{frame_h}) ---")

    # --- Theory #3 check: is the failure short-gap coincidental proximity,
    # independent of the long real-pause gap window (164/176), rather than
    # anything related to that window at all? gap_frames is already stored
    # per merge_log entry by TrackReconciler.reconcile() itself
    # ("gap_frames": frame_idx - self._last_seen[best_match][0]), so this
    # is just reading a field that already exists -- no reconciler change,
    # no re-inference.
    gaps = [m["gap_frames"] for m in merges]
    print(f"  actual gap_frames per merge: {gaps}")
    if gaps:
        print(f"  min={min(gaps)}, max={max(gaps)}, mean={sum(gaps) / len(gaps):.1f}")

    if not merges:
        print("  no merges at this config")
        return merges

    xs = [m["new_cx"] for m in merges]
    ys = [m["new_cy"] for m in merges]

    print(f"  {len(merges)} merges")
    print(f"  x: mean={statistics.mean(xs):.1f} "
          f"stdev={(statistics.pstdev(xs) if len(xs) > 1 else 0):.1f} "
          f"range=[{min(xs):.1f}, {max(xs):.1f}]  (frame width={frame_w})")
    print(f"  y: mean={statistics.mean(ys):.1f} "
          f"stdev={(statistics.pstdev(ys) if len(ys) > 1 else 0):.1f} "
          f"range=[{min(ys):.1f}, {max(ys):.1f}]  (frame height={frame_h})")

    x_spread_pct = 100.0 * (max(xs) - min(xs)) / frame_w
    y_spread_pct = 100.0 * (max(ys) - min(ys)) / frame_h
    print(f"  merge points span {x_spread_pct:.1f}% of frame width, "
          f"{y_spread_pct:.1f}% of frame height")

    if x_spread_pct < 15 and y_spread_pct < 15:
        print("  -> TIGHT cluster: supports the chokepoint theory -- merges "
              "are concentrated in a small region, consistent with different "
              "people funneling through the same narrow spot.")
    else:
        print("  -> SPREAD OUT: does NOT support the chokepoint theory as-is "
              "-- merges are happening across a wide area of the frame, not "
              "one narrow spot. Worth looking at individual merge_log "
              "entries by hand before concluding anything.")

    # Theory #3 verdict, printed alongside the spatial verdict above so
    # both are visible together in one run's output.
    if gaps:
        if max(gaps) < 164:
            print(f"  -> THEORY #3 SUPPORTED: all observed gap_frames ({min(gaps)}-{max(gaps)}) "
                  f"are well under the real single-pause gap window (164-176 frames). "
                  f"These false merges are happening on short-gap coincidental proximity "
                  f"in a busy scene -- the long gap window preserved for the real "
                  f"single-pause case is not implicated in these particular merges at all.")
        else:
            print(f"  -> THEORY #3 NOT CLEANLY SUPPORTED: at least one merge has "
                  f"gap_frames={max(gaps)}, close to or within the real single-pause "
                  f"range (164-176). Inspect that specific merge_log entry by hand -- "
                  f"it may be competing with the real single-pause signal rather than "
                  f"being an independent short-gap busy-scene case.")

    return merges


def main():
    parser = argparse.ArgumentParser(description="F6: chokepoint clustering check")
    parser.add_argument("--video", required=True, help="Path to stand-in test video")
    parser.add_argument("--model", required=True, help="Path to YOLO model weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dist", type=float, default=5.0,
                         help="Distance threshold to test (default: tightest value from the 2D sweep grid)")
    parser.add_argument("--gap-values", type=int, nargs="+", default=[164, 176],
                         help="Gap thresholds to test -- same pinned real values as dist_gap_2d_sweep.py")
    args = parser.parse_args()

    video_name = Path(args.video).stem
    frame_w, frame_h = get_frame_dimensions(args.video)
    stream = cache_raw_detection_stream(args.video, args.model, args.device)

    for gap in args.gap_values:
        check_chokepoint(stream, gap_frames_threshold=gap, dist_threshold_px=args.dist,
                          frame_w=frame_w, frame_h=frame_h, video_name=video_name)


if __name__ == "__main__":
    main()
