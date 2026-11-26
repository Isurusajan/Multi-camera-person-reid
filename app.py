"""
Multi-Camera Video Surveillance & Person Re-Identification Dashboard
Optimized for 20-30+ FPS Real-Time Inference on CPU
"""
import os
import sys
import time
import base64
import tempfile
import cv2
import numpy as np
import importlib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import src.multi_cam_manager
importlib.reload(src.multi_cam_manager)
from src.multi_cam_manager import MultiCameraPipeline

st.set_page_config(
    page_title="Multi-Camera Video Analytics & Re-ID",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title at the very top
st.title("Multi-Camera Surveillance & Person Re-Identification")
st.caption("Real-time video analytics pipeline for cross-camera person tracking, appearance feature extraction, and global identity management.")

# Video Player & Controls CSS
st.markdown("""
<style>
/* Clean page container with proper top navbar clearance */
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Clean KPI metric cards */
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
}
div[data-testid="stMetric"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}

/* Video Feed Card */
div[data-testid="stImage"] img {
    border-radius: 10px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    width: 100% !important;
}

/* Modern Video Player Controls Bar */
.st-key-in_video_player_bar,
.st-key-in_video_player_bar_multi,
.st-key-in_video_controls {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    padding: 6px 16px 8px 16px !important;
    max-width: 640px !important;
    margin: 8px auto 20px auto !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
}

/* Single Video Player Scrubber Timeline (Only ONE bar on screen) */
.st-key-in_video_player_bar div[data-testid="stSlider"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"],
.st-key-in_video_controls div[data-testid="stSlider"] {
    margin-top: 0px !important;
    margin-bottom: -10px !important;
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    position: relative !important;
    z-index: 2 !important;
}

/* Scrubber Track: 6px sleek track */
.st-key-in_video_player_bar div[data-testid="stSlider"] div[class*="e13xtgje5"],
.st-key-in_video_player_bar div[data-testid="stSlider"] [data-orientation="horizontal"] > div > div:first-child,
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] div[class*="e13xtgje5"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] [data-orientation="horizontal"] > div > div:first-child,
.st-key-in_video_controls div[data-testid="stSlider"] div[class*="e13xtgje5"],
.st-key-in_video_controls div[data-testid="stSlider"] [data-orientation="horizontal"] > div > div:first-child {
    background: rgba(255, 255, 255, 0.22) !important;
    height: 6px !important;
    border-radius: 4px !important;
}

/* Hide all slider labels, ticks, and text values for clean video player look */
.st-key-in_video_player_bar div[data-testid="stSlider"] label,
.st-key-in_video_player_bar div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
.st-key-in_video_player_bar div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
.st-key-in_video_player_bar div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
.st-key-in_video_player_bar div[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] label,
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
.st-key-in_video_controls div[data-testid="stSlider"] label,
.st-key-in_video_controls div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
.st-key-in_video_controls div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
.st-key-in_video_controls div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
.st-key-in_video_controls div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    display: none !important;
}

/* Force pointer cursor on hover anywhere over the timeline bar */
.st-key-in_video_player_bar div[data-testid="stSlider"],
.st-key-in_video_player_bar div[data-testid="stSlider"] *,
.st-key-in_video_player_bar div[data-testid="stSlider"] div,
.st-key-in_video_player_bar_multi div[data-testid="stSlider"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] *,
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] div,
.st-key-in_video_controls div[data-testid="stSlider"],
.st-key-in_video_controls div[data-testid="stSlider"] *,
.st-key-in_video_controls div[data-testid="stSlider"] div {
    cursor: pointer !important;
}

/* Scrubber Thumb (Glowing Red Circle) */
.st-key-in_video_player_bar div[data-testid="stSlider"] div[data-rac][style*="left"],
.st-key-in_video_player_bar div[data-testid="stSlider"] [role="slider"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] div[data-rac][style*="left"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"] [role="slider"],
.st-key-in_video_controls div[data-testid="stSlider"] div[data-rac][style*="left"],
.st-key-in_video_controls div[data-testid="stSlider"] [role="slider"] {
    background-color: #ff2222 !important;
    border: 2px solid #ffffff !important;
    border-radius: 50% !important;
    box-shadow: 0 0 10px rgba(255, 34, 34, 1.0) !important;
    cursor: pointer !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.st-key-in_video_player_bar div[data-testid="stSlider"]:hover div[data-rac][style*="left"],
.st-key-in_video_player_bar div[data-testid="stSlider"]:hover [role="slider"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"]:hover div[data-rac][style*="left"],
.st-key-in_video_player_bar_multi div[data-testid="stSlider"]:hover [role="slider"],
.st-key-in_video_controls div[data-testid="stSlider"]:hover div[data-rac][style*="left"],
.st-key-in_video_controls div[data-testid="stSlider"]:hover [role="slider"] {
    transform: translate(-50%, -50%) scale(1.35) !important;
    box-shadow: 0 0 16px rgba(255, 34, 34, 1.0) !important;
}

/* In-Video Buttons */
.st-key-in_video_player_bar button,
.st-key-in_video_player_bar_multi button,
.st-key-in_video_controls button {
    background: rgba(255, 255, 255, 0.14) !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 3px 12px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    cursor: pointer !important;
    height: 32px !important;
    min-height: 32px !important;
    line-height: 1 !important;
    transition: all 0.2s ease !important;
}

.st-key-in_video_player_bar button:hover,
.st-key-in_video_player_bar_multi button:hover,
.st-key-in_video_controls button:hover {
    background: #ff3b3b !important;
    border-color: #ff3b3b !important;
    transform: scale(1.05) !important;
    box-shadow: 0 0 10px rgba(255, 59, 59, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# Global frame prepper and zero-HTTP Streamlit Cloud video renderer
def prep_frame(img, max_w=640):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def render_video_frame(placeholder, frame_bgr, max_w=640):
    """
    Renders video frames directly into Streamlit DOM via lightweight Base64 Data URI.
    Constrained to max-width: 640px centered, with optimized JPEG encoding (74)
    and INTER_AREA downsampling for razor-sharp rendering with ultra-fast network transfer.
    """
    if placeholder is None or frame_bgr is None:
        return
    h, w = frame_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        frame_bgr = cv2.resize(frame_bgr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ret, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
    if ret:
        b64 = base64.b64encode(buf).decode('ascii')
        html_code = (
            f'<div style="max-width: 640px; width: 100%; margin: 0 auto; border-radius: 10px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1); line-height: 0;">'
            f'<img src="data:image/jpeg;base64,{b64}" style="width: 100%; height: auto; display: block;" />'
            f'</div>'
        )
        if hasattr(placeholder, "html"):
            placeholder.html(html_code)
        else:
            placeholder.markdown(html_code, unsafe_allow_html=True)

def render_timeline_bar(placeholder, current_frame, total_frames, video_fps, is_running=False, avg_fps=0.0):
    """
    Renders an edge-to-edge live surveillance timeline bar with moving playhead knob,
    timecode, frame counter, and AI tracking status.
    Updates synchronously with video playback in real time without DOM thrashing.
    """
    if placeholder is None or total_frames <= 0:
        return
    pct = min(100.0, max(0.0, (current_frame / float(total_frames)) * 100.0))
    cur_sec = current_frame / max(1.0, video_fps)
    tot_sec = total_frames / max(1.0, video_fps)
    cur_m, cur_s = int(cur_sec // 60), int(cur_sec % 60)
    tot_m, tot_s = int(tot_sec // 60), int(tot_sec % 60)

    if is_running:
        status_badge = f'<span style="color: #4ade80; font-weight: 600;">🟢 {avg_fps:.1f} FPS | Tracking</span>'
    elif current_frame >= total_frames - 5:
        status_badge = '<span style="color: #38bdf8; font-weight: 600;">✅ Playback Completed</span>'
    else:
        status_badge = '<span style="color: #fbbf24; font-weight: 500;">⏸️ Paused</span>'

    html_code = (
        f'<div style="width: 100%; background: transparent; padding: 2px 0; margin-top: 1px; margin-bottom: 2px;">'
        f'<div style="position: relative; width: 100%; height: 8px; background: rgba(255,255,255,0.18); border-radius: 4px; margin-bottom: 6px; overflow: visible;">'
        f'<div style="position: absolute; left: 0; top: 0; width: {pct:.2f}%; height: 100%; background: linear-gradient(90deg, #ef4444, #f43f5e); border-radius: 4px; box-shadow: 0 0 10px rgba(239,68,68,0.6);"></div>'
        f'<div style="position: absolute; left: calc({pct:.2f}% - 6px); top: -3px; width: 14px; height: 14px; background: #ffffff; border: 3px solid #ef4444; border-radius: 50%; box-shadow: 0 0 8px rgba(239,68,68,0.9); pointer-events: none;"></div>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; font-family: monospace; font-size: 12px; line-height: 1.2;">'
        f'<div style="color: #ffffff; font-weight: 600;">'
        f'⏱️ {cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d} '
        f'<span style="color: #94a3b8; font-size: 11px; margin-left: 4px;">(Frame {current_frame}/{total_frames} · {pct:.1f}%)</span>'
        f'</div>'
        f'<div>{status_badge}</div>'
        f'</div>'
        f'</div>'
    )
    if hasattr(placeholder, "html"):
        placeholder.html(html_code)
    else:
        placeholder.markdown(html_code, unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.header("Stream Configuration")

input_mode = st.sidebar.selectbox(
    "Input Source",
    [
        "Single Camera Feed (CCTV Video)",
        "Multi-Camera Network (Dual Cameras)",
        "Live Camera Stream (Webcam / IP Camera)"
    ]
)

# Reset pipeline if input mode changes
if st.session_state.get("last_input_mode") != input_mode:
    st.session_state.pipeline = None
    st.session_state.current_frame_pos = 0
    st.session_state.scrubber_pos = 0.0
    st.session_state.last_scrub_input = 0.0
    st.session_state.last_input_mode = input_mode

is_cctv_file = (input_mode == "Single Camera Feed (CCTV Video)")
is_multi_cam = (input_mode == "Multi-Camera Network (Dual Cameras)")
is_live_stream = (input_mode == "Live Camera Stream (Webcam / IP Camera)")
is_single_cam = is_cctv_file

cam1_source = None
cam2_source = None
dual_mode = ""

if is_cctv_file:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Surveillance Footage")
    video_source_option = st.sidebar.selectbox(
        "Choose Video Footage:",
        [
            "-- Choose a Video --",
            "🏪 Demo 1: Store Interior Aisle (Yellow Jacket Person)",
            "🚶 Demo 2: Street Sidewalk Outside (Yellow Jacket Person)",
            "🛒 Demo 3: Store Checkout Counter",
            "🍿 Demo 4: Retail Mart Snack Aisle",
            "📁 Upload CCTV Video File (.mp4, .avi)",
        ],
        index=0,
        help="Select '-- Choose a Video --' to start empty, choose a demo footage, or upload your own video"
    )

    if video_source_option == "📁 Upload CCTV Video File (.mp4, .avi)":
        file_single = st.sidebar.file_uploader(
            "Upload CCTV Video (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload standard surveillance video footage to analyze"
        )
        if file_single is not None:
            if "uploaded_video_path" not in st.session_state or st.session_state.get("uploaded_filename") != file_single.name:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                file_single.seek(0)
                tfile.write(file_single.read())
                tfile.flush()
                tfile.close()
                st.session_state.uploaded_video_path = tfile.name
                st.session_state.uploaded_filename = file_single.name
                st.session_state.pipeline = None
                st.session_state.current_frame_pos = 0
                st.session_state.scrubber_pos = 0.0
                st.session_state.last_scrub_input = 0.0
            cam1_source = st.session_state.uploaded_video_path
            st.sidebar.success(f"📹 Loaded: {file_single.name}")
        else:
            cam1_source = None
            st.sidebar.info("👈 Please select or drag a video file to upload.")

    elif video_source_option == "🏪 Demo 1: Store Interior Aisle (Yellow Jacket Person)":
        inside_path = os.path.join("videos", "street_store_cam1_inside.mp4")
        if os.path.exists(inside_path):
            cam1_source = inside_path
            st.sidebar.success("📹 Loaded: Store Interior Aisle Surveillance (Person in Yellow Jacket)")

    elif video_source_option == "🚶 Demo 2: Street Sidewalk Outside (Yellow Jacket Person)":
        outside_path = os.path.join("videos", "street_store_cam2_outside.mp4")
        if os.path.exists(outside_path):
            cam1_source = outside_path
            st.sidebar.success("📹 Loaded: Street Sidewalk Outside Storefront (Person in Yellow Jacket)")

    elif video_source_option == "🛒 Demo 3: Store Checkout Counter":
        cctv_path = os.path.join("videos", "cctv_single_feed.mp4")
        if os.path.exists(cctv_path):
            cam1_source = cctv_path
            st.sidebar.success("📹 Loaded: Store Checkout Counter Surveillance")

    elif video_source_option == "🍿 Demo 4: Retail Mart Snack Aisle":
        mart_path = os.path.join("videos", "mart_cam1_aisle.mp4")
        if os.path.exists(mart_path):
            cam1_source = mart_path
            st.sidebar.success("📹 Loaded: Retail Mart Snack Aisle Surveillance")

    else:  # "-- Choose a Video --"
        cam1_source = None
        st.sidebar.info("👈 Choose a video or upload CCTV footage above.")

elif is_multi_cam:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Multi-Camera Feeds")

    dual_mode = st.sidebar.selectbox(
        "Choose Video Feed Pair:",
        [
            "-- Choose a Dual Video Feed --",
            "🏪 Preset 1: Store & Street Network (Identical Person - Yellow Jacket)",
            "🛒 Preset 2: Retail Mart Surveillance (Aisle & Cashier Counter)",
            "🔬 Preset 3: EPFL Research Benchmark (Laboratory Walkway)",
            "📁 Upload Custom Dual Video Files"
        ],
        index=0,
        help="Select synchronized dual-camera feeds featuring the same person inside and outside!"
    )

    if st.session_state.get("last_dual_mode") != dual_mode:
        st.session_state.pipeline = None
        st.session_state.transition_log = []
        st.session_state.gallery_data = {}
        st.session_state.last_dual_mode = dual_mode

    if dual_mode == "-- Choose a Dual Video Feed --":
        cam1_source = None
        cam2_source = None
        st.sidebar.info("👈 Choose a dual-camera preset or upload videos above.")

    elif dual_mode == "🏪 Preset 1: Store & Street Network (Identical Person - Yellow Jacket)":
        cam1_source = os.path.join("videos", "street_store_cam1_inside.mp4")
        cam2_source = os.path.join("videos", "street_store_cam2_outside.mp4")
        st.sidebar.success("✅ **Store & Street CCTV Active**: Inside Store + Outside Sidewalk")
        st.sidebar.caption("🏪 Shows the exact same person (Yellow Jacket) walking inside the store (Cam 1) and walking outside on the street (Cam 2) with clear face Re-ID!")

    elif dual_mode == "🛒 Preset 2: Retail Mart Surveillance (Aisle & Cashier Counter)":
        cam1_source = os.path.join("videos", "mart_cam1_aisle.mp4")
        cam2_source = os.path.join("videos", "mart_cam2_counter.mp4")
        st.sidebar.success("✅ **Retail Mart CCTV Active**: Aisle + Cashier Counter")
        st.sidebar.caption("🛒 Real-world store CCTV: Identical customers shopping in snack aisle (Cam 1) and checking out at cashier (Cam 2).")

    elif dual_mode == "🔬 Preset 3: EPFL Research Benchmark (Laboratory Walkway)":
        cam1_source = os.path.join("videos", "reid_cam1.mp4")
        cam2_source = os.path.join("videos", "reid_cam2.mp4")
        st.sidebar.success("✅ **Benchmark Active**: EPFL Multi-Cam Lab")
        st.sidebar.caption("👥 Shows identical people walking across Camera 01 and Camera 02, triggering Cross-Camera Re-ID transitions.")

    else:
        file_cam1 = st.sidebar.file_uploader("Camera 01 (Angle A / Corridor)", type=["mp4", "avi", "mov", "mkv"])
        file_cam2 = st.sidebar.file_uploader("Camera 02 (Angle B / Lobby)", type=["mp4", "avi", "mov", "mkv"])

        if file_cam1 and file_cam2:
            if "uploaded_dual_path1" not in st.session_state or st.session_state.get("uploaded_dual_name1") != file_cam1.name:
                tf1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                file_cam1.seek(0)
                tf1.write(file_cam1.read())
                tf1.flush()
                tf1.close()
                st.session_state.uploaded_dual_path1 = tf1.name
                st.session_state.uploaded_dual_name1 = file_cam1.name
            cam1_source = st.session_state.uploaded_dual_path1

            if "uploaded_dual_path2" not in st.session_state or st.session_state.get("uploaded_dual_name2") != file_cam2.name:
                tf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                file_cam2.seek(0)
                tf2.write(file_cam2.read())
                tf2.flush()
                tf2.close()
                st.session_state.uploaded_dual_path2 = tf2.name
                st.session_state.uploaded_dual_name2 = file_cam2.name
            cam2_source = st.session_state.uploaded_dual_path2
            st.sidebar.success("✅ Uploaded dual camera feeds loaded!")
        else:
            st.sidebar.info("Upload two synchronized video files (.mp4) to test custom multi-camera Re-ID.")

else:  # Live Camera Stream (Webcam / IP Camera)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Camera Setup")
    live_source_type = st.sidebar.radio(
        "Camera Type:",
        ["Integrated / USB Webcam", "IP Camera (RTSP / HTTP)"],
        index=0
    )
    if live_source_type == "Integrated / USB Webcam":
        cam_idx = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, help="0 is standard default built-in laptop webcam or first USB camera.")
        cam1_source = int(cam_idx)
        st.sidebar.caption("📷 Ready to capture from local webcam.")
    else:
        rtsp_url = st.sidebar.text_input("RTSP / Stream URL", value="rtsp://admin:password@192.168.1.100:554/stream1", help="Enter RTSP or HTTP camera stream address.")
        cam1_source = rtsp_url
        st.sidebar.caption("🌐 Ready to connect to network IP camera.")
    cam2_source = None

# Persistent Session State
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "transition_log" not in st.session_state:
    st.session_state.transition_log = []
if "gallery_data" not in st.session_state:
    st.session_state.gallery_data = {}
if "current_frame_pos" not in st.session_state:
    st.session_state.current_frame_pos = 0
if "scrubber_pos" not in st.session_state:
    st.session_state.scrubber_pos = 0.0
if "timeline_scrubber_slider" not in st.session_state:
    st.session_state.timeline_scrubber_slider = 0.0

# Extract Video Metadata immediately so total_frames & video_fps are ready
total_frames = 1
video_fps = 30.0
total_sec = 1.0

if cam1_source and isinstance(cam1_source, str) and os.path.exists(cam1_source):
    _cap_meta = cv2.VideoCapture(cam1_source)
    if _cap_meta.isOpened():
        total_frames = int(_cap_meta.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_fps = _cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
        total_sec = max(1.0, total_frames / video_fps)
        _cap_meta.release()

# Primary Action Controls (Placed immediately below video selection: ZERO SCROLLING REQUIRED!)
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Feed Controls")
col_start, col_stop = st.sidebar.columns(2)

if col_start.button("▶️ Start Feed", use_container_width=True, type="primary"):
    if is_cctv_file and cam1_source is None:
        st.sidebar.warning("⚠️ Please choose or upload a video in the sidebar first.")
    elif is_multi_cam and (cam1_source is None or cam2_source is None):
        st.sidebar.warning("⚠️ Please choose a dual-camera video pair in the sidebar first.")
    else:
        # If video reached the end earlier, reset to start
        if st.session_state.current_frame_pos >= total_frames - 5:
            st.session_state.current_frame_pos = 0
            st.session_state.scrubber_pos = 0.0
            st.session_state.last_scrub_input = 0.0
            st.session_state.timeline_scrubber_slider = 0.0
        st.session_state.is_running = True
        st.rerun()

if col_stop.button("⏸️ Stop Feed", use_container_width=True):
    st.session_state.is_running = False
    st.session_state.timeline_scrubber_slider = round(st.session_state.current_frame_pos / video_fps, 1)
    st.rerun()

if st.sidebar.button("🗑️ Clear / Reset Results", use_container_width=True):
    st.session_state.is_running = False
    if "pipeline" in st.session_state and st.session_state.pipeline is not None:
        st.session_state.pipeline.reset()
    st.session_state.transition_log = []
    st.session_state.gallery_data = {}
    st.session_state.current_frame_pos = 0
    st.session_state.scrubber_pos = 0.0
    st.session_state.last_scrub_input = 0.0
    st.session_state.timeline_scrubber_slider = 0.0
    st.rerun()

# Model Parameters & Speed Controls (Cleanly grouped in compact expander below action buttons)
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ AI & Playback Settings", expanded=False):
    playback_speed = st.selectbox(
        "⚡ Video Playback Speed",
        [
            "⚡ 2.0x Fast Forward (Skip 1 frame - Fast & Fluid)",
            "⚡ 1.5x Smooth Fast (Adaptive Frame Stride)",
            "🎯 1.0x Real-Time (Process Every Frame)",
            "🚀 3.0x Ultra-Fast (Skip 2 frames - High Speed)",
            "🏎️ 4.0x Max Turbo (Skip 3 frames - Fastest)"
        ],
        index=0,
        help="Higher speeds skip intermediate frames using hardware seeking, advancing through footage 2x - 4x faster!"
    )
    conf_threshold = st.slider("Detection Confidence", 0.15, 0.85, 0.25, 0.05)
    reid_threshold = st.slider("Re-ID Match Threshold (Cosine)", 0.30, 0.85, 0.67, 0.02)
    st.caption("💡 Recommended: **0.65 - 0.70** (keeps customer & barista distinct while consolidating profiles).")

speed_mult = 2.0
frame_step = 2
if "1.5x" in playback_speed:
    speed_mult = 1.5
    frame_step = 1.5
elif "1.0x" in playback_speed:
    speed_mult = 1.0
    frame_step = 1
elif "3.0x" in playback_speed:
    speed_mult = 3.0
    frame_step = 3
elif "4.0x" in playback_speed or "Max" in playback_speed:
    speed_mult = 4.0
    frame_step = 4

# Real-Time KPI Metrics Bar
if not is_multi_cam:
    col_m1, col_m2, col_m3 = st.columns(3)
    m_unique = col_m1.empty()
    m_active = col_m2.empty()
    m_fps = col_m3.empty()
    m_unique.metric("Total Unique Persons Identified", len(st.session_state.gallery_data))
    m_active.metric("Active in Camera View", 0)
    m_fps.metric("Processing Rate", "0.0 FPS")
else:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    m_unique = col_m1.empty()
    m_cam1 = col_m2.empty()
    m_cam2 = col_m3.empty()
    m_fps = col_m4.empty()
    m_unique.metric("Total Unique Persons Identified", len(st.session_state.gallery_data))
    m_cam1.metric("Active in Camera 01", 0)
    m_cam2.metric("Active in Camera 02", 0)
    m_fps.metric("Processing Rate", "0.0 FPS")

# Video Player View
if is_cctv_file:
    if cam1_source is None:
        video_placeholder1 = st.empty()
        video_placeholder2 = None
        timeline_placeholder = None
        timecode_placeholder = None
        status_placeholder = None
        video_placeholder1.markdown(
            """
            <div style="
                border: 2px dashed rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                padding: 65px 20px;
                text-align: center;
                background: rgba(15, 23, 42, 0.45);
                color: #94a3b8;
                max-width: 640px;
                margin: 15px auto 25px auto;
            ">
                <div style="font-size: 52px; margin-bottom: 14px;">📹</div>
                <h3 style="color: #f8fafc; margin-bottom: 8px; font-weight: 600;">No Video Selected</h3>
                <p style="font-size: 14px; color: #94a3b8; max-width: 520px; margin: 0 auto; line-height: 1.6;">
                    👈 Please select <strong>"Choose Video Footage"</strong> in the left sidebar to upload your video file or load a demo feed.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Video Display Container
        video_placeholder1 = st.empty()
        video_placeholder2 = None
        timecode_placeholder = None
        status_placeholder = None

        # Unified Single Player Bar: Dynamic Pause/Resume on Left + Single Seekable Timeline on Right
        with st.container(key="in_video_player_bar"):
            c_ctrl, c_timeline = st.columns([2.0, 8.0])

            with c_ctrl:
                is_active = st.session_state.is_running
                btn_label = "⏸️ Pause" if is_active else ("▶️ Resume" if st.session_state.current_frame_pos > 0 else "▶️ Start")
                btn_type = "secondary" if is_active else "primary"
                if st.button(btn_label, key="btn_in_video_toggle", use_container_width=True, type=btn_type):
                    if st.session_state.is_running:
                        st.session_state.is_running = False
                        st.session_state.timeline_scrubber_slider = round(st.session_state.current_frame_pos / video_fps, 1)
                    else:
                        if st.session_state.current_frame_pos >= total_frames - 5:
                            st.session_state.current_frame_pos = 0
                            st.session_state.timeline_scrubber_slider = 0.0
                        st.session_state.is_running = True
                    st.rerun()

            with c_timeline:
                timeline_placeholder = st.empty()
                if not st.session_state.is_running:
                    # When paused, render the interactive draggable timeline INSIDE timeline_placeholder
                    with timeline_placeholder.container():
                        current_time_val = float(min(total_sec, max(0.0, st.session_state.current_frame_pos / video_fps)))

                        def on_timeline_scrub():
                            scrub_val = st.session_state.get("timeline_scrubber_slider", 0.0)
                            st.session_state.current_frame_pos = int(scrub_val * video_fps)
                            st.session_state.scrubber_pos = scrub_val
                            if "pipeline" in st.session_state and st.session_state.pipeline is not None:
                                st.session_state.pipeline.gallery.active_in_cam["CAM_01"].clear()

                        st.slider(
                            "Timeline",
                            min_value=0.0,
                            max_value=float(round(total_sec, 1)),
                            value=current_time_val,
                            step=0.5,
                            format="%.1fs",
                            key="timeline_scrubber_slider",
                            label_visibility="collapsed",
                            on_change=on_timeline_scrub,
                            help="Drag anywhere on this timeline to seek forward or backward."
                        )
                        cur_m, cur_s = int(current_time_val // 60), int(current_time_val % 60)
                        tot_m, tot_s = int(total_sec // 60), int(total_sec % 60)
                        pct = (st.session_state.current_frame_pos / float(total_frames)) * 100.0 if total_frames > 0 else 0.0
                        st.caption(
                            f"⏱️ **{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}** "
                            f"(Frame {st.session_state.current_frame_pos}/{total_frames} · {pct:.1f}%) · "
                            f"⏸️ *Paused (Drag slider to seek)*"
                        )
                else:
                    # When playing, render initial live moving timeline state replacing timeline_placeholder
                    timeline_placeholder.empty()
                    render_timeline_bar(
                        timeline_placeholder,
                        st.session_state.current_frame_pos,
                        total_frames,
                        video_fps,
                        is_running=True
                    )

elif is_live_stream:
    st.subheader("🔴 Live Camera Feed: Real-Time Person Detection & Re-ID")
    video_placeholder1 = st.empty()
    video_placeholder2 = None
    timeline_placeholder = None
    timecode_placeholder = None
    status_placeholder = None

else:  # is_multi_cam
    if dual_mode and "Store & Street" in dual_mode:
        cam1_title = "Camera 01: Store Interior (Shop Aisle)"
        cam2_title = "Camera 02: Street Exterior (Sidewalk)"
    elif dual_mode and "Retail Mart" in dual_mode:
        cam1_title = "Camera 01: Retail Mart - Snack Aisle"
        cam2_title = "Camera 02: Retail Mart - Cashier Counter"
    elif dual_mode and "EPFL" in dual_mode:
        cam1_title = "Camera 01: Laboratory Walkway A"
        cam2_title = "Camera 02: Laboratory Walkway B"
    else:
        cam1_title = "Camera 01: Surveillance View A"
        cam2_title = "Camera 02: Surveillance View B"

    if cam1_source is None or cam2_source is None:
        video_placeholder1 = st.empty()
        video_placeholder2 = None
        timeline_placeholder = None
        timecode_placeholder = None
        status_placeholder = None
        video_placeholder1.markdown(
            """
            <div style="
                border: 2px dashed rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                padding: 65px 20px;
                text-align: center;
                background: rgba(15, 23, 42, 0.45);
                color: #94a3b8;
                margin-top: 15px;
                margin-bottom: 25px;
            ">
                <div style="font-size: 52px; margin-bottom: 14px;">📹</div>
                <h3 style="color: #f8fafc; margin-bottom: 8px; font-weight: 600;">No Dual-Camera Feed Selected</h3>
                <p style="font-size: 14px; color: #94a3b8; max-width: 520px; margin: 0 auto; line-height: 1.6;">
                    👈 Please select <strong>"Choose Video Feed Pair"</strong> in the left sidebar to load synchronized store feeds or upload custom dual videos.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        feed_col1, feed_col2 = st.columns(2)
        with feed_col1:
            st.subheader(cam1_title)
            video_placeholder1 = st.empty()
        with feed_col2:
            st.subheader(cam2_title)
            video_placeholder2 = st.empty()
        timecode_placeholder = None
        status_placeholder = None

        with st.container(key="in_video_player_bar_multi"):
            c_ctrl_m, c_tl_m = st.columns([2.0, 8.0])
            with c_ctrl_m:
                is_active = st.session_state.is_running
                btn_label = "⏸️ Pause" if is_active else ("▶️ Resume" if st.session_state.current_frame_pos > 0 else "▶️ Start")
                btn_type = "secondary" if is_active else "primary"
                if st.button(btn_label, key="btn_in_video_toggle_multi", use_container_width=True, type=btn_type):
                    st.session_state.is_running = not st.session_state.is_running
                    st.rerun()
            with c_tl_m:
                timeline_placeholder = st.empty()
                render_timeline_bar(
                    timeline_placeholder,
                    st.session_state.current_frame_pos,
                    total_frames,
                    video_fps,
                    is_running=st.session_state.is_running
                )

# Surveillance Data Tabs
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "Activity & Re-ID Events",
    "Global Identity Registry",
    "Person Dossier (Forensic Inspector)",
    "Dataset Sources"
])
transition_placeholder = tab1.empty()
gallery_placeholder = tab2.empty()
dossier_placeholder = tab3.empty()

with tab4:
    st.markdown("""
    ### 🌐 Multi-Camera Benchmark Datasets & Video Sources

    To test Multi-Camera Person Re-ID, both cameras must capture the **same individual** moving between different camera view angles.

    ---

    #### ⚡ 1. Included Local Datasets (Already downloaded in your project!):
    - **Preset 1 (`videos/mart_cam1_aisle.mp4` & `videos/mart_cam2_counter.mp4`)**:
      - Real commercial retail mart CCTV system.
      - **Camera 01**: Snack aisle surveillance.
      - **Camera 02**: Cashier checkout counter POS surveillance.
      - Demonstrates customers moving from aisles to the cash register with high-confidence cross-camera Re-ID!
    - **Preset 2 (`videos/reid_cam1.mp4` & `videos/reid_cam2.mp4`)**:
      - Recorded from the **EPFL Multi-Camera Pedestrian Benchmark**.
      - Multiple individuals walk between Room/Walkway (Cam 1) and Corridor (Cam 2).

    ---

    #### 📥 2. Public Multi-Camera Re-ID Research Benchmarks (Free Downloads):
    If you want to download additional research datasets:
    1. **EPFL Multi-Camera Pedestrian Dataset (CVLab)**:
       - 4 to 6 synchronized cameras recording outdoor campus & indoor laboratory pedestrians.
       - Link: [EPFL CVLab Pedestrian Dataset](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
    2. **Wildtrack Multi-Camera Dataset**:
       - 7 synchronized 1080p cameras capturing pedestrians in a public square.
       - Link: [EPFL Wildtrack Dataset](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)
    3. **DukeMTMC (Multi-Target Multi-Camera Tracking)**:
       - 8 synchronized 1080p 60fps cameras across Duke University campus.
       - Available on MOTChallenge: [MOTChallenge.net](https://motchallenge.net/)
    4. **PETS 2009 Benchmark Dataset**:
       - Multi-sensor surveillance sequences with multiple camera viewpoints.
       - Link: [PETS 2009 Benchmark](https://www.cvg.reading.ac.uk/PETS2009/a.html)

    ---

    #### 📱 3. How to Record Your Own Custom Test Footage (Best for Live Interviews!):
    You can easily create custom test videos using just **two smartphones**:
    1. Place **Phone 1** facing Doorway A or Room 1.
    2. Place **Phone 2** facing Doorway B, Hallway, or Corridor.
    3. Start recording on both phones at the same time.
    4. Have a friend (or yourself) walk past Phone 1, then walk past Phone 2.
    5. Transfer the `.mp4` videos to your PC and select **"Upload Custom Dual Video Files"** in the sidebar.
    """)

def render_tabs(trans_log, gallery_identities, include_dossier=True):
    # Tab 1: Activity & Re-ID Events
    with transition_placeholder.container():
        if trans_log:
            if is_single_cam:
                formatted_events = []
                for entry in trans_log[::-1]:
                    formatted_events.append({
                        "Timestamp": entry.get("time", ""),
                        "Person Tag": f"Person #{entry.get('global_id', '')}",
                        "Camera / Area": entry.get("to_cam", "CAM_01"),
                        "Surveillance Event": entry.get("event_type", "Re-identified"),
                        "Match Confidence": f"{int(entry.get('similarity', 0.0) * 100)}%"
                    })
                st.dataframe(pd.DataFrame(formatted_events), width="stretch")
            else:
                formatted_events = []
                for entry in trans_log[::-1]:
                    formatted_events.append({
                        "Timestamp": entry.get("time", ""),
                        "Person Tag": f"Person #{entry.get('global_id', '')}",
                        "Origin Camera": entry.get("from_cam", ""),
                        "Destination Camera": entry.get("to_cam", ""),
                        "Transition Type": entry.get("event_type", ""),
                        "Match Confidence": f"{int(entry.get('similarity', 0.0) * 100)}%"
                    })
                st.dataframe(pd.DataFrame(formatted_events), width="stretch")
        else:
            if is_single_cam:
                st.info("Monitoring single surveillance feed. Re-identification and detection events will appear here.")
            else:
                st.info("Monitoring multi-camera feeds. Cross-camera transition events will appear when a person travels between zones.")

    # Tab 2: Clean Master Identity Registry Table
    with gallery_placeholder.container():
        if gallery_identities:
            reg_data = []
            for gid, meta in gallery_identities.items():
                reg_data.append({
                    "Global ID": f"Person #{gid}",
                    "Current Zone": meta["last_cam"],
                    "Total Detections": meta["count"],
                    "Snapshots Captured": len(meta.get("snapshots", [])),
                    "Last Seen Timestamp": time.strftime("%H:%M:%S", time.localtime(meta["last_time"]))
                })
            st.dataframe(pd.DataFrame(reg_data), width="stretch")
            st.caption("🔍 Switch to the **'Person Dossier (Forensic Inspector)'** tab above to view multi-angle mugshots and full movement history.")
        else:
            st.info("Monitoring stream. Detected persons will be registered into the global identity catalog.")

    # Tab 3: Detailed Person Dossier (Forensic Multi-Angle Snapshot Inspector)
    if not include_dossier:
        return

    with dossier_placeholder.container():
        if gallery_identities:
            st.markdown("#### Person Forensic Dossier & Visual Multi-Shot Inspector")
            st.caption("Review multi-angle CCTV snapshots, clothing/appearance features, and camera trajectory for each tracked individual.")

            for gid, meta in gallery_identities.items():
                snaps = meta.get("snapshots", [])
                exp_label = f"👤 Person #{gid} | Currently in {meta['last_cam']} ({len(snaps)} Multi-Angle Snapshots Captured)"
                with st.expander(exp_label, expanded=(gid == 1)):
                    kp1, kp2, kp3 = st.columns(3)
                    kp1.metric("Identity Tag", f"Global #{gid}")
                    kp2.metric("Active Camera", meta["last_cam"])
                    kp3.metric("Frames Tracked", f"{meta['count']} frames")

                    st.markdown("##### Multi-Angle Appearance Snapshots")
                    st.caption("💡 *Click the ⛶ expand icon at the top-right of any image to view in full size.*")
                    if snaps:
                        s_cols = st.columns(min(max(len(snaps), 1), 6))
                        for s_idx, snap in enumerate(snaps):
                            with s_cols[s_idx % 6]:
                                snap_rgb = cv2.cvtColor(snap["image"], cv2.COLOR_BGR2RGB)
                                st.image(snap_rgb, caption=f"📸 {snap['cam']} | {snap['time']}", width=110)
                    else:
                        st.info("Collecting visual snapshots...")

                    st.markdown("##### Camera Movement Timeline")
                    history = meta.get("history", [])
                    if history:
                        for h in history[::-1]:
                            if is_single_cam:
                                st.markdown(f"- ⏱️ **{h['time']}** — {h['event']}")
                            else:
                                st.markdown(f"- ⏱️ **{h['time']}** — `{h['cam']}` : {h['event']}")
        else:
            st.info("No persons tracked yet. Start the stream to build individual forensic profiles.")

# Initial render of tabs from session state
render_tabs(st.session_state.transition_log, st.session_state.gallery_data)

if st.session_state.is_running:
    if is_single_cam:
        if cam1_source is None:
            st.warning("Please upload a valid shop surveillance video in the sidebar.")
            st.session_state.is_running = False
        else:
            if "pipeline" not in st.session_state or st.session_state.pipeline is None:
                pipeline = MultiCameraPipeline(model_path="yolov8n.pt", reid_threshold=reid_threshold, conf=conf_threshold)
                st.session_state.pipeline = pipeline
            else:
                pipeline = st.session_state.pipeline
                pipeline.gallery.similarity_threshold = reid_threshold
                pipeline.conf = conf_threshold

            cap1 = cv2.VideoCapture(cam1_source)
            fps_history = []
            frame_idx = 0

            try:
                # Seek to current scrubbed/saved frame position
                start_frame = st.session_state.get("current_frame_pos", 0)
                if start_frame > 0 and start_frame < total_frames:
                    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                while st.session_state.is_running and cap1.isOpened():
                    ret1, frame1 = cap1.read()
                    if not ret1:
                        # Video completed cleanly! Stop playback and preserve all records
                        st.session_state.is_running = False
                        st.session_state.current_frame_pos = 0
                        st.session_state.transition_log = list(pipeline.gallery.transition_log)
                        st.session_state.gallery_data = dict(pipeline.gallery.identities)
                        render_tabs(st.session_state.transition_log, st.session_state.gallery_data, include_dossier=True)
                        if timeline_placeholder is not None:
                            render_timeline_bar(
                                timeline_placeholder,
                                total_frames,
                                total_frames,
                                video_fps,
                                is_running=False
                            )
                        break

                    t_start = time.perf_counter()
                    ann1, objs1 = pipeline.process_camera_frame(frame1, cam_id="CAM_01")
                    t_end = time.perf_counter()

                    fps = 1.0 / max(t_end - t_start, 1e-4)
                    fps_history.append(fps)
                    if len(fps_history) > 15:
                        fps_history.pop(0)
                    avg_fps = sum(fps_history) / len(fps_history)

                    render_video_frame(video_placeholder1, ann1, max_w=640)

                    # Update in-video control bar timecode and live AI tracking status
                    cur_f = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
                    st.session_state.current_frame_pos = cur_f
                    cur_s = cur_f / video_fps

                    # Smooth live timeline update in exact real-time sync with video playback
                    if frame_idx % 3 == 0 and timeline_placeholder is not None:
                        render_timeline_bar(
                            timeline_placeholder,
                            cur_f,
                            total_frames,
                            video_fps,
                            is_running=True,
                            avg_fps=avg_fps
                        )

                    # Update all KPI metrics together every 5 frames to maximize WebSocket throughput & eliminate stutter
                    if frame_idx % 5 == 0:
                        m_active.metric("Active in Camera View", len(objs1))
                        total_unique = len(pipeline.gallery.identities)
                        m_unique.metric("Total Unique Persons Identified", total_unique)
                        m_fps.metric("Processing Rate", f"{avg_fps:.1f} FPS")

                    # Calm tab update during active streaming (skips expensive forensic mugshot expanders)
                    if frame_idx % 40 == 0:
                        st.session_state.transition_log = list(pipeline.gallery.transition_log)
                        st.session_state.gallery_data = dict(pipeline.gallery.identities)
                        render_tabs(st.session_state.transition_log, st.session_state.gallery_data, include_dossier=False)

                    # Hardware frame skip to advance video at true fast-forward speed
                    if frame_step == 2:
                        cap1.grab()
                    elif frame_step == 3:
                        cap1.grab()
                        cap1.grab()
                    elif frame_step == 4:
                        cap1.grab()
                        cap1.grab()
                        cap1.grab()
                    elif frame_step == 1.5:
                        if frame_idx % 2 == 0:
                            cap1.grab()

                    frame_idx += 1

                    # Only throttle if running at 1.0x Real-time speed and completed faster than camera FPS
                    if speed_mult <= 1.0:
                        target_delay = 1.0 / max(10.0, float(video_fps))
                        elapsed = time.perf_counter() - t_start
                        sleep_time = max(0.0, target_delay - elapsed)
                        if sleep_time > 0.003:
                            time.sleep(sleep_time)
            finally:
                cap1.release()

    elif is_live_stream:
        if cam1_source is None:
            st.warning("Please configure a valid webcam index or RTSP URL in the sidebar.")
            st.session_state.is_running = False
        else:
            if "pipeline" not in st.session_state or st.session_state.pipeline is None:
                pipeline = MultiCameraPipeline(model_path="yolov8n.pt", reid_threshold=reid_threshold, conf=conf_threshold)
                st.session_state.pipeline = pipeline
            else:
                pipeline = st.session_state.pipeline
                pipeline.gallery.similarity_threshold = reid_threshold
                pipeline.conf = conf_threshold

            cap1 = cv2.VideoCapture(cam1_source)
            fps_history = []
            frame_idx = 0

            if not cap1.isOpened():
                st.error(f"❌ Failed to connect to live camera at: {cam1_source}. Please verify webcam connection or RTSP address.")
                st.session_state.is_running = False
            else:
                try:
                    while st.session_state.is_running and cap1.isOpened():
                        ret1, frame1 = cap1.read()
                        if not ret1:
                            st.warning("Live camera stream interrupted.")
                            st.session_state.is_running = False
                            break

                        t_start = time.perf_counter()
                        ann1, objs1 = pipeline.process_camera_frame(frame1, cam_id="CAM_LIVE")
                        t_end = time.perf_counter()

                        fps = 1.0 / max(t_end - t_start, 1e-4)
                        fps_history.append(fps)
                        if len(fps_history) > 15:
                            fps_history.pop(0)
                        avg_fps = sum(fps_history) / len(fps_history)

                        render_video_frame(video_placeholder1, ann1, max_w=640)

                        # Update metrics on every frame
                        m_unique.metric("Total Unique Persons Identified", len(pipeline.gallery.identities))
                        m_active.metric("Active in Camera View", len(objs1))
                        m_fps.metric("Processing Rate", f"{avg_fps:.1f} FPS")

                        if frame_idx % 35 == 0:
                            st.session_state.transition_log = list(pipeline.gallery.transition_log)
                            st.session_state.gallery_data = dict(pipeline.gallery.identities)
                            render_tabs(st.session_state.transition_log, st.session_state.gallery_data, include_dossier=False)

                        frame_idx += 1
                        time.sleep(0.015)
                finally:
                    cap1.release()

    else:  # is_multi_cam
        if cam1_source is None or cam2_source is None:
            st.warning("Please configure valid video sources in the sidebar.")
            st.session_state.is_running = False
        else:
            if "pipeline" not in st.session_state or st.session_state.pipeline is None:
                pipeline = MultiCameraPipeline(model_path="yolov8n.pt", reid_threshold=reid_threshold, conf=conf_threshold)
                st.session_state.pipeline = pipeline
            else:
                pipeline = st.session_state.pipeline
                pipeline.gallery.similarity_threshold = reid_threshold
                pipeline.conf = conf_threshold

            cap1 = cv2.VideoCapture(cam1_source)
            cap2 = cv2.VideoCapture(cam2_source)
            fps_history = []
            frame_idx = 0

            try:
                total_f1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                total_f2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                max_total_frames = max(total_f1, total_f2)

                while st.session_state.is_running and cap1.isOpened() and cap2.isOpened():
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()

                    # If one video stream is shorter than the other, loop it so both remain active
                    if not ret1:
                        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret1, frame1 = cap1.read()
                    if not ret2:
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret2, frame2 = cap2.read()

                    if (not ret1 and not ret2) or frame_idx >= max_total_frames:
                        st.session_state.is_running = False
                        st.session_state.transition_log = list(pipeline.gallery.transition_log)
                        st.session_state.gallery_data = dict(pipeline.gallery.identities)
                        render_tabs(st.session_state.transition_log, st.session_state.gallery_data, include_dossier=True)
                        if timeline_placeholder is not None:
                            render_timeline_bar(
                                timeline_placeholder,
                                max_total_frames,
                                max_total_frames,
                                video_fps,
                                is_running=False
                            )
                        break

                    t_start = time.perf_counter()

                    ann1, objs1 = pipeline.process_camera_frame(frame1, cam_id="CAM_01")
                    ann2, objs2 = pipeline.process_camera_frame(frame2, cam_id="CAM_02")

                    t_end = time.perf_counter()
                    fps = 1.0 / max(t_end - t_start, 1e-4)
                    fps_history.append(fps)
                    if len(fps_history) > 15:
                        fps_history.pop(0)
                    avg_fps = sum(fps_history) / len(fps_history)

                    render_video_frame(video_placeholder1, ann1, max_w=640)
                    render_video_frame(video_placeholder2, ann2, max_w=640)

                    if frame_idx % 3 == 0 and timeline_placeholder is not None:
                        render_timeline_bar(
                            timeline_placeholder,
                            frame_idx,
                            max_total_frames,
                            video_fps,
                            is_running=True,
                            avg_fps=avg_fps
                        )

                    if frame_idx % 5 == 0:
                        total_unique = len(pipeline.gallery.identities)
                        m_unique.metric("Unique Person Count", total_unique)
                        m_cam1.metric("Active in Camera 01", len(objs1))
                        m_cam2.metric("Active in Camera 02", len(objs2))
                        m_fps.metric("Processing Rate", f"{avg_fps:.1f} FPS")

                    if frame_idx % 40 == 0:
                        st.session_state.transition_log = list(pipeline.gallery.transition_log)
                        st.session_state.gallery_data = dict(pipeline.gallery.identities)
                        render_tabs(st.session_state.transition_log, st.session_state.gallery_data, include_dossier=False)

                    # Synchronized dual-camera frame skip to advance video at true fast-forward speed
                    if frame_step == 2:
                        cap1.grab()
                        cap2.grab()
                    elif frame_step == 3:
                        cap1.grab()
                        cap1.grab()
                        cap2.grab()
                        cap2.grab()
                    elif frame_step == 4:
                        for _ in range(3):
                            cap1.grab()
                            cap2.grab()
                    elif frame_step == 1.5:
                        if frame_idx % 2 == 0:
                            cap1.grab()
                            cap2.grab()

                    frame_idx += 1

                    # Only throttle if running at 1.0x Real-time speed and completed faster than camera FPS
                    if speed_mult <= 1.0:
                        target_delay = 1.0 / max(10.0, float(video_fps))
                        elapsed = time.perf_counter() - t_start
                        sleep_time = max(0.0, target_delay - elapsed)
                        if sleep_time > 0.003:
                            time.sleep(sleep_time)
            finally:
                cap1.release()
                cap2.release()
else:
    # When feed is stopped / paused, display preview frame at current position
    if is_cctv_file and cam1_source and isinstance(cam1_source, str) and os.path.exists(cam1_source):
        cap_prev = cv2.VideoCapture(cam1_source)
        pos = min(total_frames - 1, max(0, st.session_state.get("current_frame_pos", 0)))
        cap_prev.set(cv2.CAP_PROP_POS_FRAMES, pos)
        r, f = cap_prev.read()
        if r and f is not None:
            render_video_frame(video_placeholder1, f, max_w=640)
            if timeline_placeholder is not None:
                render_timeline_bar(
                    timeline_placeholder,
                    pos,
                    total_frames,
                    video_fps,
                    is_running=False
                )
        cap_prev.release()
    elif is_live_stream:
        st.info("📷 Live Camera Stream Standby. Click **Start Feed** in the sidebar to begin live tracking.")
    elif is_multi_cam and cam1_source and cam2_source:
        if isinstance(cam1_source, str) and os.path.exists(cam1_source) and isinstance(cam2_source, str) and os.path.exists(cam2_source):
            cp1 = cv2.VideoCapture(cam1_source)
            cp2 = cv2.VideoCapture(cam2_source)
            r1, f1 = cp1.read()
            r2, f2 = cp2.read()
            if r1 and f1 is not None and video_placeholder1:
                render_video_frame(video_placeholder1, f1, max_w=640)
            if r2 and f2 is not None and video_placeholder2:
                render_video_frame(video_placeholder2, f2, max_w=640)
            cp1.release()
            cp2.release()
        st.info("Click **Start Feed** in the sidebar to run inference and record transitions.")