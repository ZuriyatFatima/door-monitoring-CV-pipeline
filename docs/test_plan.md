# Tracking QA Test Plan — Week 7 (Team C)

Scope: validate counting/tracking accuracy across videos, covering the
required conditions (images, video, lighting, distance, occlusion, speed),
and record errors + fixes for the Friday Tracking QA report.

## Test cases

| # | Video | Resolution / orientation | Condition under test | Purpose |
|---|---|---|---|---|
| 1 | `test_1.mp4` | 402x300, landscape | Baseline: multiple pedestrians, moderate distance, daylight, lateral crossing motion | Regression baseline — must keep matching the documented IN=6/OUT=2/8-events result. Any drift here means something broke, not the new video. |
| 2 | `landscape_test_1.mp4` | 848x478, landscape (matches training resolution) | Lighting: severe backlight/glare when door opens; single subject; door state change mid-clip | Direct test of the orientation hypothesis from Week 6 (this video removes orientation as a variable) while introducing a real lighting extreme |
| 3 | `landscape_test_2.mp4` | 848x478, landscape (matches training resolution) | Occlusion + lighting: subject silhouetted directly in the doorway, door largely hidden behind them; different camera distance than test 2 | Worst-case combined test — occlusion and extreme backlight together |
| 4 | `door_test.mp4` (kept local, not shared) | 478x850, portrait | Orientation: same subject/lighting style as training but 90° rotated | Already run in Week 6 — kept as the portrait comparison point against tests 2 and 3 |

## What each result will tell us

- **If tests 2 and 3 (landscape, training-matched resolution) classify door
  state correctly** while test 4 (portrait) did not → confirms orientation
  was the dominant cause of the Week 6 failure.
- **If tests 2 and 3 also misclassify** → orientation is ruled out as the
  sole cause; lighting extremity (backlight/silhouette) becomes the
  leading suspect instead, and that becomes the new documented limitation.
- **Test 1 must reproduce IN=6/OUT=2/8 events exactly** — any deviation
  means a regression was introduced since Week 6, not a property of new
  footage, and gets fixed before anything else.

## Week 8 follow-up (added, not a rewrite of the original plan)

Test 1 (`test_1.mp4`) was re-run twice more in Week 8 as the regression baseline while investigating F6 further — both times matched IN=6/OUT=2/8 events exactly, confirming no drift. A new diagnostic script, `debug_f6_trace.py`, was added this week specifically to trace test 2's footage (`landscape_test_1.mp4`) frame-by-frame; see `failure_log.md` F6 for what it found.

The multi-person occlusion gap noted below is still open — `test_1.mp4` has multiple pedestrians but they don't cross close together, so it doesn't actually stress-test simultaneous crossings. It has been used as a regression check, not as evidence F11 is resolved. Real two-person footage from the project camera, filmed to deliberately test simultaneous or near-simultaneous crossings, is still needed and still not filmed.

## Untested conditions (documented gap, not silently skipped)

- **Speed**: no fast-motion/running footage available yet — flagged as a
  known gap in the Friday report rather than assumed fine.
- **Distance extremes**: all available footage is close-to-moderate range;
  no far-field test where subjects are small in frame.
- **Multi-person occlusion**: `test_1.mp4` has multiple pedestrians but no
  close/overlapping crossing; no test case yet for two people crossing
  the line at the same moment (this is the already-documented Week 4
  under-count limitation, still not independently re-verified this week).

## How each test is run

Person tracking / counting: through the dashboard (`app.py`) or headlessly
via `run_tracking_qa.py` (below).
Door classification over time: `classify_video_frames.py <video>`.
