import streamlit as st
import cv2
import mediapipe as mp
import time
import joblib

from src.utils import (
    calculate_ear,
    calculate_fatigue_score,
    get_recommendation
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



model = joblib.load("models/fatigue_model.pkl")


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33,160,158,133,153,144]



if st.button("▶ Start Monitoring"):

    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    # Top Metrics

    m1,m2,m3 = st.columns(3)

    status_placeholder = m1.empty()
    fatigue_placeholder = m2.empty()
    confidence_placeholder = m3.empty()

    st.divider()

  

    e1,e2,e3,e4 = st.columns(4)

    ear_placeholder = e1.empty()
    blink_placeholder = e2.empty()
    closure_placeholder = e3.empty()
    perclos_placeholder = e4.empty()

    recommendation_placeholder = st.empty()

    blink_count = 0
    closed_frames = 0
    total_frames = 0
    closed_eye_frames = 0

    EAR_THRESHOLD = 0.20

    eye_closed_start = None

    session_start = time.time()

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        rgb = cv2.cvtColor(
            frame,
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

            h,w,_ = frame.shape

            eye_points = []

            for idx in LEFT_EYE:

                landmark = face.landmark[idx]

                x = int(landmark.x * w)

                y = int(landmark.y * h)

                eye_points.append((x,y))

                cv2.circle(
                    frame,
                    (x,y),
                    2,
                    (0,255,0),
                    -1
                )

            ear = calculate_ear(
                eye_points
            )

            if ear < EAR_THRESHOLD:

                closed_frames += 1

                closed_eye_frames += 1

                if eye_closed_start is None:
                    eye_closed_start = time.time()

                closure_duration = (
                    time.time()
                    - eye_closed_start
                )

            else:

                if closed_frames >= 2:
                    blink_count += 1

                closed_frames = 0

                eye_closed_start = None

            elapsed_minutes = (
                time.time()
                - session_start
            ) / 60

            if elapsed_minutes > 0:

                blink_rate = (
                    blink_count
                    / elapsed_minutes
                )

            perclos = (
                closed_eye_frames
                / total_frames
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

            # ------------------------
            # STATUS
            # ------------------------

            if status == "ALERT":

                status_placeholder.success(
                    f"🟢 {status}"
                )

            else:

                status_placeholder.error(
                    f"🔴 {status}"
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
                f" {recommendation}"
            )

            cv2.putText(
                frame,
                f"{status}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

    cap.release()