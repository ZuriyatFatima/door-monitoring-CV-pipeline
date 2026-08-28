"""
check_kalman_drift.py

Purpose: pin down the EXACT mechanism inside ByteTrack that causes the
re-match to fail, now that buffer length (max_frames_lost=300) has been
directly ruled out as the cause (confirmed via check_tracker_internals.py).

Two remaining live hypotheses:
  A) Kalman-prediction drift: the tracker keeps predicting Track 1's box
     forward every frame using its last known velocity, even while lost.
     If the person was moving when they left frame, that prediction keeps
     "walking" in that direction for the whole gap -- so by the time the
     real detection reappears (even if physically close to WHERE the
     person vanished), the *predicted* box has drifted far from it, and
     the IoU-based match fails.
  B) Premature removal: Track 1 gets silently moved from 'lost' to
     'removed' state before frame 610, despite max_frames_lost=300 -- a
     known Ultralytics edge case (github.com/ultralytics/ultralytics
     issue #20719) where a track can be removed one frame earlier than
     expected due to a bookkeeping bug in how removed_stracks is applied.

This script logs, EVERY frame (not sampled), for the track we care about:
  - which list it's in: tracked / lost / removed / not present
  - its current predicted (Kalman) bbox center, if it has one
  - the frame it disappeared from 'tracked' and the frame (if any) it
    moved to 'removed'

Then at the exact frame the new track (e.g. ID 2) first appears, it
prints:
  - the distance between the OLD track's real last detection and its
    CURRENT Kalman-predicted position (this is hypothesis A's evidence:
    large = drift is the cause)
  - whether the OLD track was still in lost_stracks at that frame, or
    already removed (this is hypothesis B's evidence)

Usage:
    python check_kalman_drift.py uploads/landscape_test_1.mp4 --old-id 1 --new-id 2
    python check_kalman_drift.py uploads/landscape_test_2.mp4 --old-id 1 --new-id 3

(old-id / new-id come straight from your earlier check_track_coexistence.py
output -- Track 1 -> Track 2 for video 1, Track 1 -> Track 3 for video 2.)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_strack(tracker_obj, track_id):
    """Search tracked/lost/removed lists for a given track_id. Returns (strack, list_name) or (None, None)."""
    for list_name in ("tracked_stracks", "lost_stracks", "removed_stracks"):
        stracks = getattr(tracker_obj, list_name, [])
        for s in stracks:
            if getattr(s, "track_id", None) == track_id:
                return s, list_name
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--old-id", type=int, required=True, help="Track ID that vanishes (e.g. 1)")
    parser.add_argument("--new-id", type=int, required=True, help="Track ID that appears later (e.g. 2 or 3)")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO
    from modules.tracker import ensure_bytetrack_config

    tracker_yaml = ensure_bytetrack_config()
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.video}")
        sys.exit(1)

    old_id = args.old_id
    new_id = args.new_id

    last_real_pos_old = None       # (frame, cx, cy) -- last REAL detection of old_id
    old_removed_at_frame = None    # frame old_id first shows up in removed_stracks
    old_last_seen_in_lost = None   # last frame old_id was confirmed still in lost_stracks
    new_id_first_frame = None
    predicted_pos_at_new_appearance = None

    frame_idx = 0
    print(f"Running frame-by-frame trace on {args.video} (tracking old_id={old_id}, new_id={new_id}) ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = model.track(
            frame, persist=True, conf=0.4, tracker=tracker_yaml,
            device="cpu", classes=[0], verbose=False,
        )[0]

        # Record real detections this frame
        if r.boxes is not None and r.boxes.id is not None:
            xywh = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (cx, cy, w, h), tid in zip(xywh, ids):
                if tid == old_id:
                    last_real_pos_old = (frame_idx, float(cx), float(cy))
                if tid == new_id and new_id_first_frame is None:
                    new_id_first_frame = frame_idx
                    # At the exact moment new_id appears, check old_id's
                    # internal state BEFORE this frame's update potentially
                    # moves/removes it.
                    tracker_obj = model.predictor.trackers[0]
                    strack, list_name = find_strack(tracker_obj, old_id)
                    print(f"\n=== new_id={new_id} first appears at frame {frame_idx} ===")
                    print(f"old_id={old_id} internal state at this moment: {list_name or 'NOT FOUND (fully gone)'}")
                    if strack is not None:
                        tlwh = strack.tlwh  # top-left width height, from Kalman mean
                        pred_cx = tlwh[0] + tlwh[2] / 2
                        pred_cy = tlwh[1] + tlwh[3] / 2
                        predicted_pos_at_new_appearance = (pred_cx, pred_cy)
                        print(f"old_id Kalman-predicted center at this frame: ({pred_cx:.1f}, {pred_cy:.1f})")

        # Track when old_id transitions to removed (check every frame after
        # its last real detection)
        if last_real_pos_old is not None and old_removed_at_frame is None:
            tracker_obj = model.predictor.trackers[0]
            strack, list_name = find_strack(tracker_obj, old_id)
            if list_name == "lost_stracks":
                old_last_seen_in_lost = frame_idx
            elif list_name == "removed_stracks" and old_removed_at_frame is None:
                old_removed_at_frame = frame_idx

        frame_idx += 1

    cap.release()

    print(f"\n=== Summary for {args.video} ===")
    print(f"old_id={old_id} last REAL detection: {last_real_pos_old}")
    print(f"old_id={old_id} last confirmed still in lost_stracks at frame: {old_last_seen_in_lost}")
    print(f"old_id={old_id} first confirmed in removed_stracks at frame: {old_removed_at_frame}")
    print(f"new_id={new_id} first appeared at frame: {new_id_first_frame}")

    if last_real_pos_old and new_id_first_frame:
        gap = new_id_first_frame - last_real_pos_old[0]
        print(f"Gap (last real detection -> new_id appears): {gap} frames")

    if old_removed_at_frame is not None and new_id_first_frame is not None:
        if old_removed_at_frame < new_id_first_frame:
            frames_early = new_id_first_frame - old_removed_at_frame
            print(
                f"\n*** HYPOTHESIS B SUPPORTED: old_id was already REMOVED "
                f"{frames_early} frame(s) BEFORE new_id appeared, even though "
                f"max_frames_lost=300 and the real gap was smaller than that. "
                f"This points to a state-management issue removing the track "
                f"too early, not a buffer-length problem. ***"
            )
        else:
            print(
                f"\nold_id was NOT yet removed when new_id appeared (removal "
                f"happened at frame {old_removed_at_frame}, after new_id's "
                f"frame {new_id_first_frame}) -- premature removal is ruled out."
            )

    if last_real_pos_old and predicted_pos_at_new_appearance:
        real_cx, real_cy = last_real_pos_old[1], last_real_pos_old[2]
        pred_cx, pred_cy = predicted_pos_at_new_appearance
        drift = ((real_cx - pred_cx) ** 2 + (real_cy - pred_cy) ** 2) ** 0.5
        print(
            f"\nKalman drift: predicted position moved {drift:.1f}px from the "
            f"last REAL detected position, over the gap while lost."
        )
        if drift > 100:
            print(
                "*** HYPOTHESIS A SUPPORTED: the Kalman filter's predicted box "
                "drifted substantially (>100px) from the real last-seen position "
                "during the gap -- meaning even though the person reappeared "
                "physically close to where they vanished, the tracker's motion "
                "extrapolation had 'walked' the predicted box somewhere else "
                "entirely by then, so the IoU-based match against the real "
                "reappearing detection failed. This is what's causing the ID "
                "switch. ***"
            )
        else:
            print(
                "Kalman drift was small -- predicted position stayed close to "
                "the real last-seen position, so drift alone likely isn't "
                "the explanation. The IoU/match_thresh value itself, or the "
                "specific match_thresh=0.7 in your yaml, is worth checking next."
            )


if __name__ == "__main__":
    main()
