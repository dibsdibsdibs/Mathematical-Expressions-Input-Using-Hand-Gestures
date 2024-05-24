import cv2
import mediapipe as mp
# import tensorflow as tf

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# use webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # convert frame to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # find hands
    results = hands.process(frame_rgb)

    # draw hand landmarks
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
