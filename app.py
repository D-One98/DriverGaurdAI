import os
import time
import threading
import av
import cv2
import joblib
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

from src.utils import (
    calculate_ear,
    calculate_fatigue_score,
    get_recommendation
)

# ------------------------
# PATHS & CONFIG
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fatigue_model.pkl")

st.set_page_config(page_title="DriverGuard AI", page_icon="🚗", layout="wide")

st.markdown("""
<style>
.main { background-color: #0E1117; }
.metric-card { background-color: #1E2635; padding: 15px; border-radius: 10px; text-align: center; }
.big-title { font-size: 42px; font-weight: bold; }
.subtitle { color: gray; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ------------------------
# SIDEBAR & HEADER
# ------------------------
st.sidebar.title("🛡️ DriverGuard AI")
st.sidebar.markdown("---")
st.sidebar.success("🟢 System Ready")
st.sidebar.info("🧠 Fatigue Model Loaded")
st.sidebar.markdown("---")
st.sidebar.write("Version 1.0 | AI Project")

st.markdown('<div class="big-title">🚗 DriverGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Driver Fatigue Monitoring System</div>', unsafe_allow_html=True)
st.divider()

# Load Model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# ------------------------
# WEBRTC PROCESSOR CLASS
# ------------------------
class FatigueVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.blink_count = 0
        self.closed_frames = 0
        self.total_frames = 0
        self.closed_eye_frames = 0
        self.eye_closed_start = None
        self.session_start = time.time()
        self.EAR_THRESHOLD = 0.20

        # UI Variables (Thread-safe)
        self.status = "ALERT"
        self.fatigue_score = 0
        self.confidence = 100
        self.ear = 0.0
        self.blink_rate = 0.0
        self.closure_duration = 0.0
        self.perclos = 0.0
        self.recommendation = "Stay focused on the road."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.total_frames += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Temporary variables for this frame
        current_ear = 0
        current_status = "ALERT"
        current_fatigue = 0
        current_conf = 100
        current_blink_rate = 0
        current_closure = 0
        current_perclos = 0
        current_rec = ""

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            h, w, _ = img.shape
            eye_points = []

            for idx in LEFT_EYE:
                landmark = face.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                eye_points.append((x, y))
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            current_ear = calculate_ear(eye_points)

            if current_ear < self.EAR_THRESHOLD:
                self.closed_frames += 1
                self.closed_eye_frames += 1
                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()
                current_closure = time.time() - self.eye_closed_start
            else:
                if self.closed_frames >= 2:
                    self.blink_count += 1
                self.closed_frames = 0
                self.eye_closed_start = None

            elapsed = (time.time() - self.session_start) / 60
            if elapsed > 0:
                current_blink_rate = self.blink_count / elapsed

            current_perclos = (self.closed_eye_frames / self.total_frames) * 100

            # Predict
            features = [[current_ear, current_blink_rate, current_closure, current_perclos]]
            prediction = model.predict(features)[0]
            probability = max(model.predict_proba(features)[0])
            current_conf = round(probability * 100, 2)
            current_status = "DROWSY" if prediction == 1 else "ALERT"

            current_fatigue = calculate_fatigue_score(current_ear, current_perclos, current_closure)
            current_rec = get_recommendation(current_fatigue)

            # Draw on frame
            cv2.putText(img, current_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Safely update class attributes for UI to read
        with self.lock:
            self.ear = current_ear
            self.status = current_status
            self.fatigue_score = current_fatigue
            self.confidence = current_conf
            self.blink_rate = current_blink_rate
            self.closure_duration = current_closure
            self.perclos = current_perclos
            if current_rec:
                self.recommendation = current_rec

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------
# STREAMLIT UI LAYOUT
# ------------------------
left_col, right_col = st.columns([2, 1])

with right_col:
    status_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    fatigue_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    confidence_placeholder = st.empty()

st.divider()

e1, e2, e3, e4 = st.columns(4)
ear_placeholder = e1.empty()
blink_placeholder = e2.empty()
closure_placeholder = e3.empty()
perclos_placeholder = e4.empty()
recommendation_placeholder = st.empty()

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with left_col:
    ctx = webrtc_streamer(
        key="driverguard",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=FatigueVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ------------------------
# UI POLLING LOOP
# ------------------------
# This loop runs on the main Streamlit thread to update UI based on WebRTC data
if ctx.state.playing and ctx.video_processor:
    while ctx.state.playing:
        with ctx.video_processor.lock:
            p_status = ctx.video_processor.status
            p_fatigue = ctx.video_processor.fatigue_score
            p_conf = ctx.video_processor.confidence
            p_ear = ctx.video_processor.ear
            p_blink = ctx.video_processor.blink_rate
            p_closure = ctx.video_processor.closure_duration
            p_perclos = ctx.video_processor.perclos
            p_rec = ctx.video_processor.recommendation

        # Update Visuals
        color = "#163f25" if p_status == "ALERT" else "#5a1414"
        icon = "🟢" if p_status == "ALERT" else "🔴"
        
        status_placeholder.markdown(
            f"""
            <div style="background:{color}; padding:20px; border-radius:12px; text-align:center; color:white; font-size:28px; font-weight:bold;">
            {icon} {p_status}
            </div>
            """, unsafe_allow_html=True
        )

        fatigue_placeholder.metric("Fatigue Score", f"{p_fatigue}%")
        confidence_placeholder.metric("Confidence", f"{p_conf}%")
        ear_placeholder.metric("EAR", f"{p_ear:.2f}")
        blink_placeholder.metric("Blink Rate", f"{p_blink:.1f}/min")
        closure_placeholder.metric("Eye Closure", f"{p_closure:.2f}s")
        perclos_placeholder.metric("PERCLOS", f"{p_perclos:.1f}%")
        
        recommendation_placeholder.warning(p_rec)

        # Small sleep to prevent UI freeze and high CPU usage
        time.sleep(0.1)