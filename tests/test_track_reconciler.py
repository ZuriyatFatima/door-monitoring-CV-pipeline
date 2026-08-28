"""
test_track_reconciler.py

Direct evidence that TrackReconciler actually fixes the F6 pattern found
in Week 8, using the real numbers from check_kalman_drift.py's output
(not synthetic guesses):

  landscape_test_1.mp4: old_id=1 last real detection at frame 434,
    pos=(759.7, 241.1). new_id=2 first appears at frame 610 (gap=176).
  landscape_test_2.mp4: old_id=1 last real detection at frame 430,
    pos=(743.4, 242.7). new_id=3 first appears at frame 594 (gap=164).

check_kalman_drift.py didn't print the new id's exact reappearance (x, y)
(it printed the drifted *predicted* position, not the real one) -- so
this test uses the description already confirmed in this project's own
evidence chain ("the person reappeared physically close to where they
vanished" -- Hypothesis A's premise, and consistent with
check_track_coexistence.py's <150px "reappeared close" branch). Both
scenarios below are run once with a close reappearance point (should
merge) and once with a far one (should NOT merge, since that's the
genuinely-two-people case check_track_coexistence.py is built to catch).

Run:
    python test_track_reconciler.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.track_reconciler import TrackReconciler


def run_case(label, old_last_frame, old_pos, new_first_frame, new_pos, expect_merge):
    r = TrackReconciler(gap_frames_threshold=300, dist_threshold_px=150)

    # Old track seen once (its last real detection)
    canon_old = r.reconcile(old_last_frame, 1, *old_pos)
    assert canon_old == 1

    # New track's first real detection
    canon_new = r.reconcile(new_first_frame, 99, *new_pos)

    merged = (canon_new == 1)
    status = "PASS" if merged == expect_merge else "FAIL"
    print(f"[{status}] {label}: gap={new_first_frame - old_last_frame} frames, "
          f"dist={((old_pos[0]-new_pos[0])**2 + (old_pos[1]-new_pos[1])**2) ** 0.5:.1f}px "
          f"-> canonical_id={canon_new} (expected merge={expect_merge}, got merge={merged})")
    if r.merge_log:
        print(f"       merge_log: {r.merge_log}")
    return status == "PASS"


def main():
    results = []

    # --- Real F6 case, landscape_test_1.mp4 -----------------------------
    # old_id last real pos from check_kalman_drift.py output: (759.7, 241.1)
    # Reappearance assumed close (<150px) -- this is the actual pause/
    # re-enter-near-same-spot scenario the whole investigation confirmed.
    results.append(run_case(
        "landscape_test_1.mp4 pattern (close reappearance -> should merge)",
        old_last_frame=434, old_pos=(759.7, 241.1),
        new_first_frame=610, new_pos=(770.0, 250.0),  # ~13px away
        expect_merge=True,
    ))

    # --- Real F6 case, landscape_test_2.mp4 -----------------------------
    results.append(run_case(
        "landscape_test_2.mp4 pattern (close reappearance -> should merge)",
        old_last_frame=430, old_pos=(743.4, 242.7),
        new_first_frame=594, new_pos=(735.0, 255.0),  # ~14.7px away
        expect_merge=True,
    ))

    # --- Negative control: genuinely two different people ---------------
    # Same gap window, but reappearance is far away -- must NOT merge, or
    # this "fix" would silently undercount real second people.
    results.append(run_case(
        "two-different-people control (far reappearance -> must NOT merge)",
        old_last_frame=434, old_pos=(759.7, 241.1),
        new_first_frame=610, new_pos=(120.0, 400.0),  # ~730px away
        expect_merge=False,
    ))

    # --- Negative control: gap too large (beyond track_buffer window) ---
    results.append(run_case(
        "gap-too-large control (350 frames > 300 threshold -> must NOT merge)",
        old_last_frame=100, old_pos=(500.0, 200.0),
        new_first_frame=450, new_pos=(505.0, 205.0),  # close in space, far in time
        expect_merge=False,
    ))

    print()
    if all(results):
        print(f"ALL {len(results)} TESTS PASSED.")
        sys.exit(0)
    else:
        failed = len(results) - sum(results)
        print(f"{failed}/{len(results)} TEST(S) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
