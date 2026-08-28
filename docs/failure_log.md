# Failure Log — Week 7 (Team C)

Each entry: what was tested, what happened, evidence, and current status.

## F1 — Zero crossing events on dwell-only footage

**Test:** Week 6 door_test.mp4 (person stands near door, doesn't cross the line).
**Result:** IN=0, OUT=0, 0 events, but 4 unique tracks detected.
**Status:** BY DESIGN — door_state only logs at a crossing event. No crossing occurred, so no reading. Documented as a known behavior, not a defect.

## F2 — Screenshot gallery crash on corrupted file

**Test:** Dashboard screenshot gallery after an interrupted earlier run left a partial/corrupt jpg in `screenshots/`.
**Result:** `PIL.UnidentifiedImageError` took down the entire results page.
**Status:** RESOLVED. Wrapped each `st.image()` call in try/except; unreadable files now show a warning caption instead of crashing. Verified against a real corrupted file (same exception type reproduced and caught).

## F3 — Wrong model file used

**Test:** A second `best.pt` (18.7MB) was substituted for the verified model (72.5MB).
**Result:** Model misclassified an obviously-open door as closed on every sampled frame.
**Status:** RESOLVED. Extracted and compared embedded training metadata from both files: the 18.7MB file was an earlier `yolo11s_v2-2` run built on stock YOLO11s (not on top of v3 at all), not the verified `v4_finetune`. Correct model identified and restored by base-model path, run name, and training date — not by guessing from file size or folder location.

## F4 — Door classifier fails on portrait-orientation video

**Test:** Week 6 door_test.mp4 (portrait, 478x850) vs. two new landscape videos (848x478, matching training data).
**Result:** Portrait video: wrong on 15/15 sampled frames. Landscape videos: correctly tracked real door-state transitions over time (one single-frame blip in one video).
**Status:** OPEN / KNOWN LIMITATION, root cause narrowed. Orientation mismatch with training data is now the confirmed leading explanation, not lighting or camera quality (both landscape videos also have heavy backlight and still worked). Recommendation: add portrait-oriented training examples if portrait input is expected in deployment.

## F5 — Inconsistent event count depending on Resize setting

**Test:** Same landscape video run twice — dashboard with Resize=720 vs. Resize off.
**Result:** Resize=720 gave 1 event; Resize off gave 2 events, on identical footage.
**Status:** RESOLVED / EXPLAINED. Confirmed the resize setting was causing a real crossing to be missed at lower resolution (a detection dropped a frame, breaking the track at the critical moment) — not a random inconsistency. Recommend running at full resolution when accuracy matters more than speed.

## F6 — Inconsistent IN/OUT counts and directions between similar videos

**Test:** Two videos, same room, same "person walks through doorway" scenario.
**Result:** Video 1: IN=0, OUT=2. Video 2: IN=1, OUT=1. Screenshots showed the same physical person appearing under 2-3 different track IDs within a single clip.
**Status:** OPEN — NOT resolved. This entry was previously marked "fix applied, verification pending"; that was wrong, corrected in Week 8 with the same honesty as the original write-up.

Timeline, corrected:
- **Week 7 theory (superseded):** track pauses exceeded `track_buffer` (120 frames, ~4.3s), so the track expired and a new ID was issued on re-detection. Fix proposed: raise `track_buffer` to 300.
- **Week 8, root cause actually confirmed:** the real mechanism is not track expiry — it's Kalman-filter drift. ByteTrack keeps extrapolating a lost track's last-known velocity forward every frame it's missing; during a 164-176 frame pause the predicted box drifts 1500+ px in an 848px-wide frame, so IoU matching fails on re-detection even though the person is still right there. **`track_buffer=300` was re-tested and confirmed byte-identical before/after** — it does not fix this, because the track isn't being removed too early, it's being mismatched. The Week 7 "fix applied" status for this bug was therefore incorrect; see `bug_fixes.md` Fix 5 for the corrected verification.
- **Week 8, mitigation added:** `modules/track_reconciler.py` — a post-hoc reconciliation pass matching fragmented IDs on real observed positions. Kept diagnostic-only (feeds a parallel `unique_tracks_reconciled`/`tracks_merged` count and a dashboard warning), deliberately **not** wired into the actual counter — an earlier version that did wire it in broke the `test_1.mp4` regression baseline (OUT 2→1, total_events 8→7) by falsely merging two genuinely separate people's crossings.
- **Week 8, exact double-count mechanism diagnosed:** frame-by-frame trace (`debug_f6_trace.py`) on `landscape_test_1.mp4` showed the OUT=2 pattern is two ID fragments each independently registering the same brief near-line dip during one continuous pause (reconciler confirms both fragments are the same person: 176-frame gap, ~27-29px). Separately, IN=0 was found to be a **different, unrelated bug**: the first raw track ID's first detected frame is already past the crossing line — the true entry was never detected at all, not a counting-logic failure.
- **Week 8, fix attempted and reverted:** bridged `LineCounter`'s counted/prev_y/hit_count state across reconciler-confirmed merges only (narrower than full ID substitution). Fixed `landscape_test_1.mp4` (OUT 2→1) but **failed regression testing**: broke the `test_1.mp4` baseline (IN/OUT split moved 6/2→7/1) and broke `landscape_test_2.mp4`, which was previously correct (IN 1→0). Root cause of the regression: `counted_ids` has no concept of direction, so bridging suppresses a genuinely separate, correctly-directed crossing whenever the reconciler merges two fragments — the same over-merge failure class as the original canonical_id revert, now shown to affect targeted state-bridging too. **Reverted immediately on regression failure.** Current shipped behavior is unchanged: raw per-track counting, reconciler diagnostic-only.

**Current state:** dashboard IN/OUT numbers can still be wrong on footage with a pause near the crossing line. No fix has survived regression testing yet. See F11 below — same underlying limitation.

## F11 — Reconciler over-merge risk in busy/multi-person scenes

**Test:** `test_1.mp4` (multi-person regression baseline) used as a stand-in for busy-scene stress testing; earlier investigated via EPFL CVLab footage.
**Result:** Any approach that lets the reconciler's fragment-matching influence the actual IN/OUT counts risks merging two genuinely different people's crossings into one, or (Week 8 finding) suppressing one person's real crossing because a *different* person's fragment triggered a merge match. Confirmed twice: once when canonical_id was fed directly into `counter.update()` (Week 7, reverted), and again when only counting *state* was bridged across confirmed merges (Week 8, also reverted).
**Status:** OPEN, same underlying structural limitation as F6 — not two separate issues. Any real fix needs `counted_ids` (or equivalent) to be direction-aware, not just identity-aware, so a reconciler-confirmed merge can dedupe a repeated same-direction event without blocking a legitimately different one. Real two-person footage from the actual project camera is still needed to test this properly — current evidence relies on `test_1.mp4` and EPFL stand-in footage, not project-specific multi-person footage.

## F7 — Single-frame door-state misclassification blip

**Test:** landscape_test_1.mp4 door-state tracking over time.
**Result:** One sampled frame (15.1s) read as `door_closed` in the middle of an otherwise-correct open sequence (8.6s-26.0s).
**Status:** OPEN, minor. Likely a transition-moment frame or motion blur. Not investigated further given low impact (1 of 15 samples, self-correcting on the next sample).

## F8 — yolo11n.pt download failures / file lock errors

**Test:** Local environment attempting to auto-download `yolo11n.pt`.
**Result:** `ConnectionError` (network) and separately `PermissionError` (file locked by a stray running process).
**Status:** RESOLVED — environment/process issue, not an application defect. Fixed by closing duplicate Streamlit sessions and retrying on a working connection.

## F9 — Missing ground-truth labels handled safely

**Test:** `verify_classifier_accuracy.py` run against a folder where some images might lack label files.
**Result:** Missing labels are skipped with a clear message rather than crashing.
**Status:** NOT A BUG — defensive design, verified in the script's own edge-case testing.
