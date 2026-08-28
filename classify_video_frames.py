"""
classify_video_frames.py
Samples frames evenly across a video and runs the door classifier on
each one -- independent of the person-counter/line-crossing logic.

Use this when your video doesn't have someone crossing a counting
line (e.g. someone just standing near the door, or the door opening/
closing without anyone walking through) -- door_state normally only
gets logged at a crossing event, so this bypasses that and gives you
a direct read of "what does the classifier think the door state is"
at evenly-spaced points through the whole clip.

Usage:
    python classify_video_frames.py path/to/video.mp4 [num_samples]

num_samples defaults to 15.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "models/best.pt"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    if not os.path.exists(video_path):
        print(f"ERROR: video not found at '{video_path}'")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at '{MODEL_PATH}'")
        sys.exit(1)

    import cv2
    from modules.tracker import DoorStateClassifier

    print(f"Loading model from {MODEL_PATH} ...")
    classifier = DoorStateClassifier(model_path=MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1

    if total_frames <= 0:
        print("ERROR: could not read frame count from video.")
        sys.exit(1)

    # Evenly spaced sample points across the whole video
    sample_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]

    print(f"\nVideo: {total_frames} frames at {fps:.1f}fps (~{total_frames/fps:.1f}s)")
    print(f"Sampling {num_samples} frames:\n")
    print(f"{'Frame':<10} {'Time (s)':<12} {'Predicted door state'}")
    print("-" * 45)

    # Also save each sampled frame as a jpg so you can visually cross-check
    out_dir = "video_frame_samples"
    os.makedirs(out_dir, exist_ok=True)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"{idx:<10} {'':<12} FAILED TO READ FRAME")
            continue

        state = classifier.classify_frame(frame)
        timestamp = idx / fps
        print(f"{idx:<10} {timestamp:<12.1f} {state}")

        out_path = os.path.join(out_dir, f"frame{idx}_{state}.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()
    print(f"\nSampled frames saved to {out_dir}/ (filename includes the prediction) "
          f"-- open a few and compare by eye against what the label says.")


if __name__ == "__main__":
    main()
