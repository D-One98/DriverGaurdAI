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

import av
import threading
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fatigue_model.pkl"
)


# PAGE CONFIG


st.set_page_config(
    page_title="DriverGuard AI",
    page_icon=" ",
    layout="wide"
)


st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #1E2635;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    color: gray;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.title(" DriverGuard AI")

st.sidebar.markdown("---")

st.sidebar.success("🟢 Camera Connected")

st.sidebar.info(" Fatigue Model Loaded")

st.sidebar.markdown("---")

st.sidebar.write("Version 1.0")

st.sidebar.write("AI Internship Project")


st.markdown(
    '<div class="big-title">🚗 DriverGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time Driver Fatigue Monitoring System</div>',
    unsafe_allow_html=True
)

st.divider()


print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("EXISTS =", os.path.exists(MODEL_PATH))

model = joblib.load(MODEL_PATH)


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]

if st.button("▶ Start Monitoring"):

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


    class VideoProcessor(VideoProcessorBase):

        def __init__(self):

            self.lock = threading.Lock()

            self.blink_count = 0
            self.closed_frames = 0
            self.total_frames = 0
            self.closed_eye_frames = 0

            self.eye_closed_start = None
            self.session_start = time.time()

            self.EAR_THRESHOLD = 0.20

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            self.total_frames += 1

            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb)

            status = "ALERT"

            fatigue_score = 0

            confidence = 100

            ear = 0

            blink_rate = 0

            closure_duration = 0

            perclos = 0

            if results.multi_face_landmarks:

                face = results.multi_face_landmarks[0]

                h, w, _ = img.shape

                eye_points = []

                for idx in LEFT_EYE:

                    landmark = face.landmark[idx]

                    x = int(landmark.x * w)

                    y = int(landmark.y * h)

                    eye_points.append((x, y))

                    cv2.circle(
                        img,
                        (x, y),
                        2,
                        (0,255,0),
                        -1
                    )

                ear = calculate_ear(eye_points)

                if ear < self.EAR_THRESHOLD:

                    self.closed_frames += 1
                    self.closed_eye_frames += 1

                    if self.eye_closed_start is None:
                        self.eye_closed_start = time.time()

                    closure_duration = (
                        time.time() -
                        self.eye_closed_start
                    )

                else:

                    if self.closed_frames >= 2:
                        self.blink_count += 1

                    self.closed_frames = 0
                    self.eye_closed_start = None

                elapsed = (
                    time.time()
                    - self.session_start
                ) / 60

                if elapsed > 0:

                    blink_rate = (
                        self.blink_count /
                        elapsed
                    )

                perclos = (
                    self.closed_eye_frames /
                    self.total_frames
                ) * 100

                features = [[
                    ear,
                    blink_rate,
                    closure_duration,
                    perclos
                ]]

                prediction = model.predict(features)[0]

                probability = max(
                    model.predict_proba(features)[0]
                )

                confidence = round(
                    probability * 100,
                    2
                )

                status = (
                    "DROWSY"
                    if prediction == 1
                    else "ALERT"
                )

                fatigue_score = calculate_fatigue_score(
                    ear,
                    perclos,
                    closure_duration
                )

                recommendation = get_recommendation(
                    fatigue_score
                )

                with self.lock:

                    status_placeholder.markdown(
                        f"""
                        <div style="
                        background:{
                        '#163f25'
                        if status=='ALERT'
                        else '#5a1414'
                        };
                        padding:20px;
                        border-radius:12px;
                        text-align:center;
                        color:white;
                        font-size:28px;
                        font-weight:bold;">
                        {'🟢' if status=='ALERT' else '🔴'}
                        {status}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    fatigue_placeholder.metric(
                        "Fatigue Score",
                        f"{fatigue_score}%"
                    )

                    confidence_placeholder.metric(
                        "Confidence",
                        f"{confidence}%"
                    )

                    ear_placeholder.metric(
                        "EAR",
                        f"{ear:.2f}"
                    )

                    blink_placeholder.metric(
                        "Blink Rate",
                        f"{blink_rate:.1f}/min"
                    )

                    closure_placeholder.metric(
                        "Eye Closure",
                        f"{closure_duration:.2f}s"
                    )

                    perclos_placeholder.metric(
                        "PERCLOS",
                        f"{perclos:.1f}%"
                    )

                    recommendation_placeholder.warning(
                        recommendation
                    )

                cv2.putText(
                    img,
                    status,
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


    rtc_config = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )

    with left_col:

        webrtc_streamer(
            key="driverguard",
            rtc_configuration=rtc_config,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )