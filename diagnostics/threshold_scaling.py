"""
threshold_scaling.py

Fix for the over-merging bug found in verify_two_person_scene.py:
dist_threshold_px=150 was tuned against landscape_test_1/2 footage
(848x478), where it worked correctly for the single-person-pause case
(real measured reappearance drift was only 15-27px, so 150px gave
comfortable margin without being dangerously loose).

That same ABSOLUTE pixel value is resolution-dependent, not a fixed
"how far can a person's box realistically move" quantity. On the
360x288 stand-in test footage:
    150px / 848px frame width = 17.7% of frame width  (landscape_test_1/2)
    150px / 360px frame width = 41.7% of frame width  (passageway/terrace)
The same threshold became 2.4x more permissive, relative to the frame,
on the smaller footage -- which is consistent with the observed
77%/94% merge rates: at 42% of frame width, almost any two people
passing through the same doorway/chokepoint within the gap window
fall inside the distance threshold, merge-eligible, regardless of
whether they're the same person.

Fix: scale dist_threshold_px by frame width, anchored to the ratio
that's already validated against real single-person-pause behavior
on landscape_test_1/2. This keeps the SAME relative tolerance instead
of the same absolute pixel count, so it doesn't loosen just because
the footage is smaller.

Usage (drop into wherever TrackReconciler gets constructed):

    from threshold_scaling import scaled_dist_threshold_px

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    reconciler = TrackReconciler(
        gap_frames_threshold=300,
        dist_threshold_px=scaled_dist_threshold_px(frame_width),
    )
"""

# Anchor values: the resolution and threshold that were actually
# validated against real footage and real measured drift
# (landscape_test_1/2, 848x478, single-person-pause case, real
# reappearance drift was 15-27px -- 150px gives ~5.5x-10x margin over
# that, which is the safety margin this scaling preserves at any
# resolution).
REFERENCE_FRAME_WIDTH = 848
REFERENCE_DIST_THRESHOLD_PX = 150


def scaled_dist_threshold_px(frame_width: int) -> float:
    """
    Returns a dist_threshold_px scaled to preserve the SAME fraction
    of frame width as the validated reference (150px / 848px = 17.7%),
    rather than reusing an absolute pixel count that becomes too loose
    (or too strict) on different-resolution footage.

    Example results:
        848 width (landscape_test_1/2) -> 150.0px  (unchanged, as expected)
        360 width (passageway/terrace)  -> 63.7px   (was 150px -- the bug)
        1920 width (a hypothetical 1080p camera) -> 339.4px
    """
    ratio = REFERENCE_DIST_THRESHOLD_PX / REFERENCE_FRAME_WIDTH
    return round(frame_width * ratio, 1)


if __name__ == "__main__":
    # Quick sanity check you can run standalone before wiring it in:
    #   python threshold_scaling.py
    test_widths = [848, 360, 288, 1920, 1280]
    print(f"{'Frame width':<15} {'Scaled dist_threshold_px':<28} {'% of frame width'}")
    print("-" * 60)
    for w in test_widths:
        t = scaled_dist_threshold_px(w)
        print(f"{w:<15} {t:<28} {100*t/w:.1f}%")
