"""
benchmark_fps.py
Headless FPS benchmark for the Day 4 "FPS improvement notes" deliverable.
Runs the actual PersonTracker (same code path as the dashboard) across
a few resize/frame-skip settings on a real video and reports real,
measured FPS for each — no dashboard UI needed, no guessing.

Usage:
    python benchmark_fps.py uploads/test_1.mp4

Runs 4 configurations by default (baseline, resize only, skip only,
both combined) and prints a comparison table you can paste directly
into your Day 4 notes.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PERSON_MODEL_PATH = "yolo11n.pt"

# Each tuple: (label, resize_width_or_None, skip_n)
CONFIGS = [
    ("Baseline (no resize, no skip)", None, 1),
    ("Resize to 480px", 480, 1),
    ("Skip every 2nd frame", None, 2),
    ("Resize 480px + skip every 2nd frame", 480, 2),
]

MAX_FRAMES_TO_TEST = 150  # cap so the benchmark finishes in reasonable time


def run_config(video_path, model, tracker_yaml, resize_width, skip_n):
    import cv2
    from modules.optimize import resize_frame, FrameSkipper

    cap = cv2.VideoCapture(video_path)
    skipper = FrameSkipper(skip_n=skip_n)

    frame_count = 0
    start = time.time()

    while frame_count < MAX_FRAMES_TO_TEST:
        ok, frame = cap.read()
        if not ok:
            break

        if resize_width is not None:
            frame = resize_frame(frame, resize_width)

        if skipper.should_process():
            model.track(
                frame, persist=True, conf=0.4, tracker=tracker_yaml,
                device="cpu", classes=[0], verbose=False,
            )

        frame_count += 1

    elapsed = time.time() - start
    cap.release()
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    return fps, frame_count, elapsed


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"ERROR: video not found at '{video_path}'")
        sys.exit(1)

    from ultralytics import YOLO
    from modules.tracker import ensure_bytetrack_config

    print(f"Loading {PERSON_MODEL_PATH} ...")
    model = YOLO(PERSON_MODEL_PATH)
    tracker_yaml = ensure_bytetrack_config()

    print(f"Benchmarking on '{video_path}' (up to {MAX_FRAMES_TO_TEST} frames per config)\n")
    print(f"{'Configuration':<40} {'FPS':<10} {'Frames':<10} {'Time (s)'}")
    print("-" * 75)

    results = []
    for label, resize_width, skip_n in CONFIGS:
        fps, frames, elapsed = run_config(video_path, model, tracker_yaml, resize_width, skip_n)
        results.append((label, fps))
        print(f"{label:<40} {fps:<10.2f} {frames:<10} {elapsed:.1f}")

    baseline_fps = results[0][1]
    print("\nSpeedup vs baseline:")
    for label, fps in results[1:]:
        speedup = fps / baseline_fps if baseline_fps > 0 else 0
        print(f"  {label}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
