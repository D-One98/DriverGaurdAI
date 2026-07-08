from src.utils import (
    calculate_ear,
    calculate_fatigue_score,
    get_recommendation
)
import streamlit as st
import cv2
import mediapipe as mp
import time
import joblib
import os
import av
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------
# PATH & CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fatigue_model.pkl")

st.set_page_config(
    page_title="DriverGuard AI",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0E1117; }
.metric-card { background-color: #1E2635; padding: 15px; border-radius: 10px; text-align: center; }
.big-title { font-size: 42px; font-weight: bold; }
.subtitle { color: gray; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🚗 DriverGuard AI")
st.sidebar.markdown("---")
st.sidebar.success("🟢 Camera Connected")
st.sidebar.info("🧠 Fatigue Model Loaded")
st.sidebar.markdown("---")
st.sidebar.write("Version 1.0")
st.sidebar.write("AI Internship Project")

st.markdown('<div class="big-title">🚗 DriverGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Driver Fatigue Monitoring System</div>', unsafe_allow_html=True)
st.divider()

print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("EXISTS =", os.path.exists(MODEL_PATH))

model = joblib.load(MODEL_PATH)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.20

# -------------------------
# WEBRTC VIDEO PROCESSOR
# -------------------------
class FatigueDetector(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe inside the processor to ensure thread safety
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )
        
        # State variables for tracking blinks and fatigue
        self.blink_count = 0
        self.closed_frames = 0
        self.total_frames = 0
        self.closed_eye_frames = 0
        self.eye_closed_start = None
        self.session_start = time.time()
        
        # Thread-safe lock for updating metrics
        self.lock = threading.Lock()
        self.metrics = {
            "status": "ALERT", "fatigue_score": 0, "confidence": 100,
            "ear": 0.0, "blink_rate": 0.0, "closure_duration": 0.0,
            "perclos": 0.0, "recommendation": "Initializing..."
        }

    def recv(self, frame):
        # Convert av.VideoFrame to numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")
        self.total_frames += 1
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        # Default values if no face is detected
        status = "ALERT"
        fatigue_score = 0
        confidence = 100
        ear = 0
        blink_rate = 0
        closure_duration = 0
        perclos = 0
        recommendation = "No face detected"

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
            
            ear = calculate_ear(eye_points)
            
            if ear < EAR_THRESHOLD:
                self.closed_frames += 1
                self.closed_eye_frames += 1
                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()
                closure_duration = time.time() - self.eye_closed_start
            else:
                if self.closed_frames >= 2:
                    self.blink_count += 1
                self.closed_frames = 0
                self.eye_closed_start = None
            
            elapsed_minutes = (time.time() - self.session_start) / 60
            if elapsed_minutes > 0:
                blink_rate = self.blink_count / elapsed_minutes
            
            perclos = (self.closed_eye_frames / self.total_frames) * 100
            
            features = [[ear, blink_rate, closure_duration, perclos]]
            prediction = model.predict(features)[0]
            probability = max(model.predict_proba(features)[0])
            confidence = round(probability * 100, 2)
            status = "DROWSY" if prediction == 1 else "ALERT"
            
            fatigue_score = calculate_fatigue_score(ear, perclos, closure_duration)
            recommendation = get_recommendation(fatigue_score)
            
            cv2.putText(img, f"{status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Update metrics thread-safely
        with self.lock:
            self.metrics = {
                "status": status, "fatigue_score": fatigue_score, "confidence": confidence,
                "ear": ear, "blink_rate": blink_rate, "closure_duration": closure_duration,
                "perclos": perclos, "recommendation": recommendation
            }
            
        # Return the annotated frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# STREAMLIT UI LAYOUT
# -------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    # The WebRTC component handles camera permissions and streaming
    ctx = webrtc_streamer(
        key="driver-guard",
        video_processor_factory=FatigueDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Crucial for heavy processing like MediaPipe
    )

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

# -------------------------
# REAL-TIME UI UPDATES
# -------------------------
if ctx.state.playing:
    # Safely read the latest metrics from the background thread
    with ctx.processor.lock:
        metrics = ctx.processor.metrics.copy()
        
    
    status = metrics["status"]
    if status == "ALERT":
        status_placeholder.markdown(
            f"""<div style="background:#163f25; padding:20px; border-radius:12px; text-align:center; color:white; font-size:28px; font-weight:bold;">🟢 {status}</div>""",
            unsafe_allow_html=True
        )
    else:
        status_placeholder.markdown(
            f"""<div style="background:#5a1414; padding:20px; border-radius:12px; text-align:center; color:white; font-size:28px; font-weight:bold;">🔴 {status}</div>""",
            unsafe_allow_html=True
        )
        
    fatigue_placeholder.metric("Fatigue Score", f"{metrics['fatigue_score']}%")
    confidence_placeholder.metric("Confidence", f"{metrics['confidence']}%")
    ear_placeholder.metric("EAR", f"{metrics['ear']:.2f}")
    blink_placeholder.metric("Blink Rate", f"{metrics['blink_rate']:.1f}/min")
    closure_placeholder.metric("Eye Closure", f"{metrics['closure_duration']:.2f}s")
    perclos_placeholder.metric("PERCLOS", f"{metrics['perclos']:.1f}%")
    recommendation_placeholder.warning(f" {metrics['recommendation']}")
    
    # Rerun the script every 0.1s to update the dashboard metrics in real-time
    time.sleep(0.1)
    st.rerun() # Note: Use st.experimental_rerun() if you are on Streamlit < 1.27
else:
    status_placeholder.info("Click 'START' in the video player to begin monitoring.")
    recommendation_placeholder.info("Waiting for camera to start...")