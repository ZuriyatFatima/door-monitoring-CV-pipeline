"""
test_door_classifier.py
Standalone sanity check for the door classifier (best.pt) — no video,
no dashboard, no line-crossing logic. Just: does this model correctly
label a handful of real images as door_open / door_closed?

Use this on a few images you already have and trust the label for —
ideally a handful from your own training/validation set, since those
are guaranteed to match the domain the model was actually trained on.
(Random stock photos of doors won't tell you much — see the note in
your Week 6 dashboard chat about why generalization to new domains is
already a known weak spot for this model.)

Usage:
    python test_door_classifier.py path/to/image1.jpg path/to/image2.jpg ...

Or point it at a folder:
    python test_door_classifier.py path/to/folder_of_images/
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "models/best.pt"  # edit if yours lives elsewhere

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_image_paths(args):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            for ext in IMAGE_EXTS:
                paths.extend(glob.glob(os.path.join(arg, f"*{ext}")))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"WARNING: '{arg}' not found, skipping")
    return paths


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at '{MODEL_PATH}'. Edit MODEL_PATH at "
              f"the top of this script.")
        sys.exit(1)

    image_paths = collect_image_paths(sys.argv[1:])
    if not image_paths:
        print("No images found in the given path(s).")
        sys.exit(1)

    # Imported here (not at module top) so --help / bad-args exit cleanly
    # even in an environment where ultralytics isn't installed yet.
    from modules.tracker import DoorStateClassifier
    import cv2

    print(f"Loading model from {MODEL_PATH} ...")
    classifier = DoorStateClassifier(model_path=MODEL_PATH)

    print(f"\nTesting {len(image_paths)} image(s):\n")
    print(f"{'Image':<40} {'Predicted state':<20}")
    print("-" * 60)

    results = {"door_open": 0, "door_closed": 0, "unknown": 0}
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{os.path.basename(path):<40} FAILED TO LOAD IMAGE")
            continue
        state = classifier.classify_frame(frame)
        results[state] = results.get(state, 0) + 1
        print(f"{os.path.basename(path):<40} {state:<20}")

    print("\nSummary:")
    for state, count in results.items():
        print(f"  {state}: {count}")

    if results.get("unknown", 0) == len(image_paths):
        print(
            "\nAll images came back 'unknown' — the model isn't detecting "
            "anything above its confidence threshold on these images. If "
            "you expected real detections, double-check these images match "
            "the framing/domain your training data used."
        )


if __name__ == "__main__":
    main()
