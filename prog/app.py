import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('hand_gesture_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    return None

# use webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # convert frame to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    keypoints = extract_keypoints(frame)
    if keypoints is not None:
        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints)
        predicted_class = np.argmax(prediction)
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # find hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # display result
    cv2.imshow('Hand Gesture Recognition', frame)

    # q for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release webcam
cap.release()
cv2.destroyAllWindows()