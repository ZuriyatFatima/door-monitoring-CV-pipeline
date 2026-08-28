# Door-Monitoring CV System

Person tracking and IN/OUT counting for a door/passageway camera feed, with a
separate door-open/closed classifier and a Streamlit dashboard.

**Team C — VC Tech Internship** (Zuriyat Fatima, Kiran Farwa)

## What it does

- **Tracking**: YOLO11n + ByteTrack, per-frame person detection and ID assignment.
- **Counting**: line-crossing IN/OUT counts (`modules/counter.py`), driven by raw
  ByteTrack IDs.
- **Reconciliation**: `modules/track_reconciler.py` matches fragmented tracks
  (same person, different raw IDs) on observed position, for **diagnostics only**
  — it does not feed into the shipped count (see Known Issues).
- **Door state**: separate fine-tuned open/closed classifier.
- **Dashboard**: Streamlit app (`app.py`) with CSV/Excel export and event
  screenshots.

## Repo layout

| Folder | Contents |
|---|---|
| `modules/` | Production pipeline code (tracker, counter, reconciler, optimizer, logger) |
| `docs/` | Bug tracking, failure log, test plan, weekly reports |
| `diagnostics/` | One-off investigation scripts kept for audit trail (frame-level tracing, drift measurement, threshold sweeps) — not part of the shipped pipeline |
| `tests/` | Regression / unit tests |
| `qa_results/` | Headless QA runner CSV output history (gitignored — regenerable) |
| `qa_reruns/` | Rerun logs kept as evidence for specific verification claims (e.g. `track_buffer` 120→300 byte-identical confirmation) — tracked in git, not gitignored |
| `models/` | Model weights (gitignored via `.gitkeep` pattern — see Setup) |
| `uploads/` | Test/demo video files, incl. the regression baseline `test_1.mp4` (gitignored via `.gitkeep` pattern — see Setup) |

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Model weights and video files are not committed to git (too large / regenerable).
Obtain them from [team drive / describe source] and place them in:
- `models/` — detector and classifier weights (`yolo11n.pt`, `best.pt`)
- `uploads/` — test videos (`test_1.mp4`, `landscape_test_1.mp4`, `landscape_test_2.mp4`)

**Note**: `models/best.pt` is ~72MB — too large for a normal git commit. It's
hosted at [team drive link] rather than tracked in git; if you need it version
controlled, use Git LFS instead.

## Running

```powershell
# Dashboard
streamlit run app.py

# Headless QA runner (no UI)
python run_tracking_qa.py
```

## Known issues — read before touching counting logic

Full detail in `docs/failure_log.md` and `docs/bug_fixes.md`. Summary:

- **F6 / F11 (same underlying issue)**: a person pausing near the crossing line
  causes ByteTrack's Kalman filter to drift 1500+ px, fragmenting the track.
  Each fragment can independently register a line-crossing dip, causing a
  double-count. Root cause fully diagnosed (`diagnostics/debug_f6_trace.py`),
  not yet fixed without regressing other footage.
- **IN=0**: separate bug. The true entry crossing sometimes happens before the
  track is ever created (missed detection), not a counting-logic failure.
  Root cause not yet pinned down (candidates: pre-frame-228 footage gap, or
  `conf=0.4` threshold too high).
- **`track_reconciler.canonical_id` is intentionally not fed into
  `counter.update()`.** Doing so over-merges genuinely separate crossings.
  Two fix attempts (canonical_id direct feed, `LineCounter.bridge()`) were
  tried and reverted after breaking the regression baseline — see
  `docs/bug_fixes.md` Fix 5 and Fix 7 for why.
- **Next direction (not yet attempted)**: make `counted_ids` direction-aware
  (track IN vs OUT per identity separately), so a confirmed merge can dedupe a
  repeat crossing without blocking a genuinely different one.

## Regression baseline

Any change to counting logic must be checked against `test_1.mp4` **first**:

```
IN = 6, OUT = 2, total_events = 8
```

If this baseline breaks, the change is reverted regardless of how well it
performs on other footage.

## Current shipped behavior

Raw per-track counting only. The reconciler runs in parallel and surfaces a
dashboard warning (`unique_tracks_reconciled`, `tracks_merged`) when it
detects likely fragmentation, but does not correct the displayed count.
Numbers can still be wrong on footage with a pause near the crossing line.
