"""
app.py — Week 6 Dashboard (Team C)
Run with:  streamlit run app.py

What this covers off the Week 6 checklist:
  Day 1: dashboard skeleton                -> this file, the sidebar/layout
  Day 2: live result display               -> the video frame loop below
  Day 3: CSV/Excel export + screenshots    -> modules/logger.py, sidebar download buttons
  Day 4: FPS optimization                  -> modules/optimize.py, sidebar toggles
  Day 5: integrated demo                   -> this app IS the integrated demo

EDIT THESE before running:
  PERSON_MODEL_PATH -> a general COCO model that tracks/counts PEOPLE
                        crossing the line (default 'yolo11n.pt' auto-downloads)
  DOOR_MODEL_PATH   -> your fine-tuned door_open/door_closed classifier,
                        used to tag each crossing event with door state
  LINE_Y            -> your locked crossing line (Week 4 default: 1167,
                        tuned for the 1080x1920 portrait clip — adjust if
                        your video's resolution differs)
"""

import os
import sys
import time
import tempfile

import cv2
import streamlit as st
import pandas as pd

# Make modules/ importable regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.tracker import PersonTracker, DoorStateClassifier
from modules.counter import LineCounter
from modules.logger import EventLogger
from modules.optimize import resize_frame, FrameSkipper, FPSTracker
from modules.track_reconciler import TrackReconciler

# --------------------------------------------------------------------
# CONFIG — edit these for your setup
# --------------------------------------------------------------------
PERSON_MODEL_PATH = "yolo11n.pt"    # generic COCO model — tracks/counts PEOPLE
DOOR_MODEL_PATH = "models/best.pt"  # <-- your fine-tuned door_open/door_closed model
LINE_Y = 1167                        # <-- your locked Week 4 crossing line
# --------------------------------------------------------------------

st.set_page_config(page_title="Door Tracking Dashboard", layout="wide")
st.title("Door Detection + Tracking Dashboard")
st.caption("Team C — Week 6: Dashboard, Integration and Optimization")

# ---- Sidebar: controls -----------------------------------------------
with st.sidebar:
    st.header("Run settings")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    device = st.selectbox("Inference device", ["cpu", "cuda:0"], index=0)

    st.subheader("Day 4 — FPS optimization")
    resize_enabled = st.checkbox("Resize input", value=True)
    target_width = st.slider("Resize width (px)", 320, 1280, 640, step=80,
                              disabled=not resize_enabled)
    skip_n = st.slider("Process every Nth frame", 1, 5, 1,
                        help="1 = no skipping. Higher = faster but less precise counting.")

    line_y_input = st.number_input("Crossing line Y (px)", value=LINE_Y, step=10)

    run_button = st.button("Run detection + tracking", type="primary")

# ---- Session state for persisting results between reruns --------------
if "summary" not in st.session_state:
    st.session_state.summary = None
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "events_df" not in st.session_state:
    st.session_state.events_df = None

# ---- Main run loop ------------------------------------------------------
video_placeholder = st.empty()
metrics_placeholder = st.empty()

if run_button:
    if uploaded_video is None:
        st.error("Upload a video first.")
    elif not os.path.exists(DOOR_MODEL_PATH):
        st.error(
            f"Door model not found at '{DOOR_MODEL_PATH}'. Edit DOOR_MODEL_PATH "
            f"at the top of app.py to point at your best.pt weights."
        )
    else:
        # Save upload to a temp file so cv2.VideoCapture can read it
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_file.write(uploaded_video.read())
        tmp_file.close()

        # Two separate models doing two separate jobs — see modules/tracker.py
        person_tracker = PersonTracker(model_path=PERSON_MODEL_PATH, device=device)
        door_classifier = DoorStateClassifier(model_path=DOOR_MODEL_PATH, device=device)
        counter = LineCounter(line_y=int(line_y_input), min_hits=5)
        logger = EventLogger()
        skipper = FrameSkipper(skip_n=skip_n)
        fps_tracker = FPSTracker()
        # Week 8 fix for F6 (Kalman motion-prediction drift, confirmed root
        # cause -- see failure_log.md F6). Sits between the raw tracker
        # output and the counter; merges a track that reappears close to
        # where an earlier track went quiet, instead of trusting ByteTrack's
        # drifted Kalman prediction. See modules/track_reconciler.py.
        reconciler = TrackReconciler(gap_frames_threshold=300, dist_threshold_px=150)

        cap = cv2.VideoCapture(tmp_file.name)
        frame_idx = 0
        last_result = None  # reused on skipped frames

        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if resize_enabled:
                frame = resize_frame(frame, target_width)

            if skipper.should_process():
                result = person_tracker.track_frame(frame)
                last_result = result
            else:
                result = last_result  # reuse last detections on skipped frames

            annotated = frame
            if result is not None and result.boxes is not None and result.boxes.id is not None:
                annotated = result.plot()  # Ultralytics built-in box/ID drawing

                boxes = result.boxes.xywh.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for (x, y, w, h), track_id in zip(boxes, ids):
                    # Week 8 fix for F6: reconciler runs in parallel for
                    # reporting only. It is NOT passed to counter.update() --
                    # an earlier version did that and it silently merged
                    # genuinely separate crossings (confirmed regression on
                    # the test_1.mp4 baseline: OUT 2->1, events 8->7). The
                    # counter always sees the RAW track_id, so IN/OUT counts
                    # are guaranteed identical to pre-fix behavior.
                    reconciler.reconcile(frame_idx, int(track_id), float(x), float(y))
                    event = counter.update(frame_idx, int(track_id), float(y))
                    if event is not None:
                        # Only run the door classifier at the moment of a
                        # crossing — cheap because it's not every frame.
                        event.door_state = door_classifier.classify_frame(frame)
                        logger.log_event(event, frame=annotated)

            # Draw the crossing line for visual reference
            cv2.line(annotated, (0, int(line_y_input)),
                      (annotated.shape[1], int(line_y_input)), (0, 0, 255), 2)

            video_placeholder.image(annotated, channels="BGR", use_container_width=True)

            fps = fps_tracker.tick()
            summary = counter.summary()
            metrics_placeholder.markdown(
                f"**FPS:** {fps:.1f} &nbsp;|&nbsp; "
                f"**IN:** {summary['in_count']} &nbsp;|&nbsp; "
                f"**OUT:** {summary['out_count']} &nbsp;|&nbsp; "
                f"**Frame:** {frame_idx}/{total_frames}"
            )

            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        os.unlink(tmp_file.name)

        csv_path = logger.flush_csv()
        summary = counter.summary()
        summary["track_reconciliation"] = reconciler.summary()
        st.session_state.summary = summary
        st.session_state.csv_path = csv_path
        st.session_state.events_df = pd.DataFrame(logger.rows)

        st.success(f"Done. {len(logger.rows)} crossing events logged.")
        if reconciler.merge_log:
            st.info(
                f"Track reconciliation merged {len(reconciler.merge_log)} fragmented "
                f"track ID(s) back into their original identity (F6 fix — see "
                f"modules/track_reconciler.py). Details in the run summary below."
            )

# ---- Results / export section (Day 3) ----------------------------------
if st.session_state.summary is not None:
    st.subheader("Run summary")
    st.json(st.session_state.summary)

    st.subheader("Event log")
    st.dataframe(st.session_state.events_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
            with open(st.session_state.csv_path, "rb") as f:
                st.download_button("Download CSV", f, file_name=os.path.basename(st.session_state.csv_path))
    with col2:
        if st.session_state.events_df is not None and not st.session_state.events_df.empty:
            excel_buf = st.session_state.csv_path.replace(".csv", ".xlsx")
            st.session_state.events_df.to_excel(excel_buf, index=False)
            with open(excel_buf, "rb") as f:
                st.download_button("Download Excel", f, file_name=os.path.basename(excel_buf))

    st.subheader("Event screenshots")
    shot_dir = "screenshots"
    if os.path.isdir(shot_dir):
        shots = sorted(os.listdir(shot_dir))[-12:]  # most recent 12
        if shots:
            cols = st.columns(4)
            shown_any = False
            for i, shot in enumerate(shots):
                shot_path = os.path.join(shot_dir, shot)
                try:
                    with cols[i % 4]:
                        st.image(shot_path, caption=shot, use_container_width=True)
                        shown_any = True
                except Exception:
                    # A corrupted/partial file from an earlier interrupted run
                    # shouldn't take down the whole page -- skip it and note it.
                    with cols[i % 4]:
                        st.caption(f"⚠️ {shot} (unreadable, skipped)")
            if not shown_any:
                st.caption("No readable screenshots in this batch.")
        else:
            st.caption("No screenshots yet.")
