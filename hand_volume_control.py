import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, GUID

# ============= 1️⃣ Train SVM on synthetic gesture data (demo only) =============
def generate_fake_data():
    X, y = [], []
    for i in range(60):
        # "up" gesture (fingers close)
        up = np.random.rand(42) * 0.4
        X.append(up)
        y.append("up")
        # "down" gesture (fingers spread)
        down = np.random.rand(42) * 0.4 + 0.6
        X.append(down)
        y.append("down")
    return np.array(X), np.array(y)

X, y = generate_fake_data()
svm_model = SVC(kernel='rbf')
svm_model.fit(X, y)
print("✅ SVM model trained successfully on synthetic data")

# ============= 2️⃣ Setup Pycaw for Volume Control =============
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(GUID('{5CDF2C82-841E-4546-9722-0CF74078229A}'),
                             CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[:2]

# ============= 3️⃣ Mediapipe Setup =============
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ============= 4️⃣ Webcam Setup =============
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
print("🎥 Starting camera... Press 'q' to quit")

# ============= 5️⃣ Real-time Gesture Detection + Volume Bar =============
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Get current volume (0–1 scale)
    current_vol = volume.GetMasterVolumeLevelScalar()
    volBar = np.interp(current_vol, [0, 1], [400, 150])
    volPer = int(current_vol * 100)
    status = "Waiting..."

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Collect 21 landmark (x,y)
            landmarks = []
            for lm in handLms.landmark:
                landmarks.append(lm.x)
            for lm in handLms.landmark:
                landmarks.append(lm.y)

            # Predict gesture using SVM
            pred = svm_model.predict([landmarks])[0]

            # Volume control logic
            if pred == "up":
                volume.SetMasterVolumeLevelScalar(1.0, None)   # Max volume
                status = "🔊 Volume Up"
            elif pred == "down":
                volume.SetMasterVolumeLevelScalar(0.2, None)   # Low volume
                status = "🔉 Volume Down"
            else:
                status = "✋ Neutral"

            # Update display values
            current_vol = volume.GetMasterVolumeLevelScalar()
            volBar = np.interp(current_vol, [0, 1], [400, 150])
            volPer = int(current_vol * 100)

    # ============= 🎚️ Draw Volume Bar =============
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 3)  # Border
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)  # Fill
    cv2.putText(frame, f'{volPer} %', (40, 440),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, status, (150, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("🖐️ SVM Hand Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Exited successfully.")
