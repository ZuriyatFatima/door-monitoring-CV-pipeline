"""
verify_classifier_accuracy.py
Computes REAL accuracy for the door classifier against ground-truth
YOLO label files — no more eyeballing images one at a time.

Your door_bbox_dataset follows the standard YOLO layout:
    images/val/some_image.jpg
    labels/val/some_image.txt   <- one line per box: "<class_id> x y w h"

class_id 0/1 maps to door_open/door_closed per your data.yaml (check
CLASS_MAP below matches your actual data.yaml — this is the one thing
you must confirm, everything else is automatic).

Usage:
    python verify_classifier_accuracy.py "G:\\dataset for internship\\door_bbox_dataset"

Expects that path to contain images/val/ and labels/val/ as siblings.
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "models/best.pt"

# CONFIRM THIS matches your data.yaml's 'names' list order before trusting results.
# data.yaml usually looks like: names: ['door_open', 'door_closed']  -> 0=open, 1=closed
CLASS_MAP = {0: "door_open", 1: "door_closed"}


def read_ground_truth_label(label_path: str) -> str | None:
    """
    Reads a YOLO label file and returns the class name of the box with
    the LARGEST area (in case of multiple boxes, the dominant door).
    Returns None if the label file is missing or empty.
    """
    if not os.path.exists(label_path):
        return None
    best_area = -1
    best_class = None
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            area = w * h
            if area > best_area:
                best_area = area
                best_class = CLASS_MAP.get(class_id, f"unknown_class_{class_id}")
    return best_class


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_root = sys.argv[1]
    images_dir = os.path.join(dataset_root, "images", "val")
    labels_dir = os.path.join(dataset_root, "labels", "val")

    if not os.path.isdir(images_dir):
        print(f"ERROR: '{images_dir}' not found. Pass the dataset root "
              f"(the folder containing images/ and labels/), not the images folder itself.")
        sys.exit(1)
    if not os.path.isdir(labels_dir):
        print(f"ERROR: '{labels_dir}' not found — can't get ground truth without labels.")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at '{MODEL_PATH}'.")
        sys.exit(1)

    from modules.tracker import DoorStateClassifier
    import cv2

    print(f"Loading model from {MODEL_PATH} ...")
    classifier = DoorStateClassifier(model_path=MODEL_PATH)

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"\nFound {len(image_paths)} validation images.\n")

    # confusion matrix: confusion[true][predicted] = count
    confusion = {}
    correct = 0
    total_scored = 0
    mismatches = []

    print(f"{'Image':<30} {'Ground truth':<15} {'Predicted':<15} {'Match'}")
    print("-" * 75)

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")

        true_label = read_ground_truth_label(label_path)
        if true_label is None:
            print(f"{os.path.basename(img_path):<30} NO LABEL FILE — skipped")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"{os.path.basename(img_path):<30} FAILED TO LOAD IMAGE — skipped")
            continue

        predicted = classifier.classify_frame(frame)

        match = "YES" if predicted == true_label else "NO"
        if predicted == true_label:
            correct += 1
        else:
            mismatches.append((os.path.basename(img_path), true_label, predicted))
        total_scored += 1

        confusion.setdefault(true_label, {}).setdefault(predicted, 0)
        confusion[true_label][predicted] += 1

        print(f"{os.path.basename(img_path):<30} {true_label:<15} {predicted:<15} {match}")

    print("\n" + "=" * 75)
    if total_scored > 0:
        acc = 100 * correct / total_scored
        print(f"ACCURACY: {correct}/{total_scored} = {acc:.1f}%")
    else:
        print("No images could be scored (missing labels or unreadable images).")

    print("\nConfusion matrix (rows=ground truth, cols=predicted):")
    all_classes = sorted(set(list(confusion.keys()) + [c for d in confusion.values() for c in d.keys()]))
    header = " " * 16 + "".join(f"{c:<15}" for c in all_classes)
    print(header)
    for true_c in all_classes:
        row = f"{true_c:<16}"
        for pred_c in all_classes:
            row += f"{confusion.get(true_c, {}).get(pred_c, 0):<15}"
        print(row)

    if mismatches:
        print(f"\n{len(mismatches)} misclassified image(s) — worth a manual look:")
        for name, true_l, pred_l in mismatches:
            print(f"  {name}: ground truth={true_l}, predicted={pred_l}")


if __name__ == "__main__":
    main()
