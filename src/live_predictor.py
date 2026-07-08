import cv2
import mediapipe as mp
import time
import joblib
from scipy.spatial import distance

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fatigue_model.pkl"
)


# LOAD TRAINED MODEL


model = joblib.load("MODEL_PATH")

# EAR FUNCTION


def calculate_ear(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


# EYE LANDMARKS


LEFT_EYE = [33, 160, 158, 133, 153, 144]


# MEDIAPIPE SETUP


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


# CAMERA


cap = cv2.VideoCapture(0)


# VARIABLES


EAR_THRESHOLD = 0.20

blink_count = 0
closed_frames = 0

eye_closed_start = None
closure_duration = 0

total_frames = 0
closed_eye_frames = 0

start_time = time.time()


# MAIN LOOP


while True:

    success, frame = cap.read()

    if not success:
        break

    total_frames += 1

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    status = "ALERT"
    fatigue_score = 0
    perclos = 0
    blink_rate = 0

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        left_eye = []

       
        # EYE LANDMARKS
      

        for idx in LEFT_EYE:

            lm = face.landmark[idx]

            x = int(lm.x * w)
            y = int(lm.y * h)

            left_eye.append((x, y))

            cv2.circle(frame,(x, y), 2,(0, 255, 0),-1 )

      
        # EAR
        

        ear = calculate_ear(left_eye)

       
        # BLINK DETECTION
      

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
            closure_duration = 0

      
        # BLINK RATE
     

        elapsed_minutes = (time.time() - start_time) / 60

        if elapsed_minutes > 0:

            blink_rate = (blink_count /elapsed_minutes)

       
        # PERCLOS
      

        perclos = (closed_eye_frames / total_frames) * 100

    
        # ML PREDICTION
     

        try:

            features = [[
                ear,
                blink_rate,
                closure_duration,
                perclos
            ]]

            prediction = model.predict(
                features
            )[0]

            if prediction == 1:
                status = "DROWSY"
            else:
                status = "ALERT"

        except Exception as e:

            print(e)

        # FATIGUE SCORE
        fatigue_score = ((1 - min(ear, 0.4) / 0.4) * 40 + min(perclos, 100) * 0.4 +min(closure_duration, 5) * 10)

        fatigue_score = max( 0, min(100, fatigue_score))

    
        # STATUS COLOR
    
        status_color = (
            (0, 255, 0)
            if status == "ALERT"
            else (0, 0, 255) )

        # DISPLALY
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Closure: {closure_duration:.2f}s",(20, 160),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255),2)

        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Status: {status}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.putText(frame, f"Fatigue Score: {fatigue_score:.0f}%", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # WARNING
        if fatigue_score > 70:

            cv2.putText(frame, "WARNING! DROWSINESS DETECTED", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("DriverGuard AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP
cap.release()
cv2.destroyAllWindows()