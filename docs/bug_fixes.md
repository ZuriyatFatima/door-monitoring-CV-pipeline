# Bug Fix List — Week 7 (Team C)

## Fix 1 — Screenshot gallery crash (F2)
**File:** `app.py`
**Before:** `st.image(os.path.join(shot_dir, shot), ...)` called directly in a loop — one bad file crashed the whole page.
**After:** Each call wrapped in try/except; unreadable files show a caption warning and the rest of the gallery still renders.
**Verified:** Yes — reproduced the exact exception type (`UnidentifiedImageError`) against a real corrupted file and confirmed it's now caught.

## Fix 2 — Wrong model file in use (F3)
**Action:** Not a code fix — a verification/process fix. Extracted embedded training metadata (base model, run name, training date) from candidate `.pt` files to identify the correct one, rather than trusting filename or folder location.
**Verified:** Yes — confirmed via direct metadata inspection (base=`best_v3.pt`, run=`v4_finetune`, matching byte size to the originally-uploaded correct file).

## Fix 3 — Architecture separation (carried from Week 6, still in effect)
**File:** `modules/tracker.py`
**Before:** The door classifier itself was used as the line-crossing tracker.
**After:** `PersonTracker` (drives counting) and `DoorStateClassifier` (tags state per event) kept as two separate classes with distinct responsibilities.
**Verified:** Yes — confirmed still in place this week, no regressions.

## Fix 4 — Stale tracker config silently ignored
**File:** `modules/tracker.py`, `ensure_bytetrack_config()`
**Before:** Only wrote the ByteTrack YAML config if it didn't already exist — a code change to tracker settings would silently have no effect if a stale config file was already on disk.
**After:** Always overwrites the config file on every call, guaranteeing code and on-disk config never drift apart.
**Verified:** Yes — tested against a simulated stale file, confirmed the fix correctly overwrites with the new settings.

## Fix 5 — track_buffer too short for real dwell time (F6)
**File:** `modules/tracker.py`
**Before:** `track_buffer: 120` — at ~28fps, only keeps a lost track alive for ~4.3 seconds.
**After:** `track_buffer: 300` (~10-11 seconds at 28fps) — covers the pause length observed when a person stops to close the door behind them.
**Verified:** NO — this was previously marked "PARTIALLY verified, pending clean re-run." The re-run happened in Week 8 and **disproved the fix**: `track_buffer=300` was confirmed byte-identical before/after on the affected footage. Corrected root-cause understanding: the track isn't expiring too early (which `track_buffer` controls) — it's being actively mismatched due to Kalman-filter prediction drift during the pause. Raising `track_buffer` doesn't address that. The setting is left at 300 (it's not harmful and is closer to the real dwell time), but it should not be described as a fix for F6. See `failure_log.md` F6 for the full corrected timeline.

## Fix 6 — Track reconciliation pass added (F6/F11, diagnostic only)
**File:** `modules/track_reconciler.py` (new, Week 8)
**What:** Added `TrackReconciler`, matching fragmented track IDs on real observed positions (not Kalman predictions) after a gap. Produces a parallel diagnostic count (`unique_tracks_reconciled`, `tracks_merged`) and a dashboard warning.
**Deliberately NOT done:** feeding the reconciler's `canonical_id` into `counter.update()`. An earlier version did this and broke the `test_1.mp4` regression baseline (OUT 2→1, total_events 8→7) by falsely merging two genuinely separate people's crossings. `LineCounter` always receives the raw `track_id`.
**Verified:** Yes, as a diagnostic — reconciler correctly identifies known fragment pairs (e.g. 176-frame gap, ~27-29px match on `landscape_test_1.mp4`). Does NOT correct the actual IN/OUT numbers, by design.

## Fix 7 — Attempted: bridge counting state across confirmed merges (F6, REVERTED)
**File:** `modules/counter.py` (`LineCounter.bridge()`), `run_tracking_qa.py`, `app.py`
**What was tried:** When the reconciler confirms a merge, propagate `counted_ids`/`prev_y`/`hit_counts` from the old track ID to the new one, so a reconciler-confirmed continuation can't double-fire a crossing. Narrower than Fix 6's rejected approach — bridges state only, not identity.
**Result:** Fixed `landscape_test_1.mp4` (OUT 2→1, matching ground truth). **Broke regression testing**: `test_1.mp4` baseline moved (IN/OUT split 6/2→7/1) and `landscape_test_2.mp4` regressed from correct (IN=1 OUT=1) to broken (IN=0 OUT=1).
**Root cause of failure:** `counted_ids` has no concept of direction — bridging marks a new ID "already counted" based on whether the old ID fired *any* event, incorrectly suppressing a genuinely separate, correctly-directed crossing whenever the reconciler merges two fragments.
**Verified:** Reverted immediately on regression failure. Not shipped. `modules/counter.py`, `run_tracking_qa.py`, and `app.py` are back to their pre-Fix-7 state. Documented here per team practice — a rejected fix with real evidence of why it doesn't work is useful even though it isn't shipped.

