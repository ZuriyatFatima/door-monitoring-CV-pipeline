"""
modules/track_reconciler.py — Week 8 fix for F6 (Kalman motion-prediction drift)

Root cause (confirmed, Week 8, see failure_log.md F6 / Week7_TrackingQA_Report
Section 5): ByteTrack's Kalman filter keeps extrapolating a lost track's
last-known velocity forward every frame it's missing. When a person pauses
164-176 frames (as measured on landscape_test_1.mp4 / landscape_test_2.mp4),
the predicted box drifts 1500+ px in a frame that's only 848px wide -- so
even though the person physically reappears close to where they vanished,
IoU matching against the *predicted* box fails and ByteTrack hands out a
new track ID. track_buffer=300 does NOT fix this (verified byte-identical
before/after in Week 8) because the track isn't being removed early --
it's being mismatched.

Fix strategy: don't try to patch ByteTrack's Kalman/IoU matching itself
(fragile, version-specific, matches the exact class of problem flagged in
Ultralytics issue #20719). Instead, add a thin reconciliation pass AFTER
the tracker returns its (possibly-fragmented) IDs, matching on REAL
detected positions only:

  - When a track goes quiet, remember its last REAL (not predicted) position.
  - When a brand-new track ID first appears, check whether it started close
    (in space, not the drifted prediction) to a recently-quiet track, within
    a bounded number of frames.
  - If so, treat the new ID as a continuation of the old one for counting
    and logging purposes.

This module does not import or modify anything in modules/tracker.py and
does not touch Ultralytics/ByteTrack internals -- it only operates on the
(track_id, x, y) stream that PersonTracker.track_frame() already exposes
via result.boxes, exactly as consumed in app.py and run_tracking_qa.py.
It can be disabled by simply not calling reconcile().
"""

import math


class TrackReconciler:
    """
    Create ONE instance per video / per run, and feed every detection
    through reconcile() before handing the id to LineCounter.update():

        reconciler = TrackReconciler(gap_frames_threshold=300, dist_threshold_px=150)
        ...
        for (x, y, w, h), raw_track_id in zip(boxes, ids):
            canonical_id = reconciler.reconcile(frame_idx, int(raw_track_id), float(x), float(y))
            event = counter.update(frame_idx, canonical_id, float(y))

    Parameters
    ----------
    gap_frames_threshold : int
        Max frames between an old track going quiet and a new track
        appearing, to be considered a candidate merge. Default 300
        matches track_buffer in bytetrack_custom.yaml -- comfortably
        above the 164-176 frame gaps actually observed.
    dist_threshold_px : float
        Max distance between the old track's last REAL position and the
        new track's first REAL position for them to be treated as the
        same person. Default 150 matches the threshold already used in
        check_track_coexistence.py's diagnostic output (>150px there was
        the "reappeared far away, not the same continuity case" cutoff),
        kept consistent with the evidence that established this fix.
    """

    def __init__(self, gap_frames_threshold=300, dist_threshold_px=150):
        self.gap_frames_threshold = gap_frames_threshold
        self.dist_threshold_px = dist_threshold_px

        # raw_track_id -> canonical_id, built up as merges happen
        self._canonical = {}

        # canonical_id -> (last_frame_idx, cx, cy) of its last REAL detection
        self._last_seen = {}

        # audit trail -- one entry per merge applied, for the QA report/CSV
        self.merge_log = []

    def _resolve(self, track_id):
        """Follow the canonical chain (handles a track merging more than once)."""
        seen = set()
        while track_id in self._canonical and track_id not in seen:
            seen.add(track_id)
            track_id = self._canonical[track_id]
        return track_id

    def reconcile(self, frame_idx, raw_track_id, cx, cy):
        """
        Call once per detection per frame.

        Returns the canonical track_id to use downstream: raw_track_id
        itself if this is a genuinely new/already-known identity, or the
        id of an earlier track it has just been merged into.
        """
        track_id = self._resolve(raw_track_id)

        if track_id not in self._last_seen:
            # First appearance of this identity -- check whether it should
            # instead be merged into a recently-quiet track based on real
            # position, not merely "it's a new id".
            best_match = None
            best_dist = None
            for cand_id, (cand_frame, cand_cx, cand_cy) in self._last_seen.items():
                gap = frame_idx - cand_frame
                if gap <= 0 or gap > self.gap_frames_threshold:
                    continue
                dist = math.hypot(cx - cand_cx, cy - cand_cy)
                if dist <= self.dist_threshold_px and (best_dist is None or dist < best_dist):
                    best_match = cand_id
                    best_dist = dist

            if best_match is not None:
                self._canonical[raw_track_id] = best_match
                self.merge_log.append({
                    "new_id": raw_track_id,
                    "merged_into": best_match,
                    "frame": frame_idx,
                    "gap_frames": frame_idx - self._last_seen[best_match][0],
                    "distance_px": round(best_dist, 1),
                })
                track_id = best_match

        self._last_seen[track_id] = (frame_idx, cx, cy)
        return track_id

    def summary(self):
        return {
            "merges_applied": len(self.merge_log),
            "merge_log": self.merge_log,
        }
