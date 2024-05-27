import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

detected_number_symbols = ""
state = "number"
expression = ""
result = ""
hand_cooldown = 60
isdone = False
error_displayed = False

model_path = os.path.join(os.path.dirname(__file__), 'hand_gesture_model.h5')
model = load_model(model_path)

cv2.namedWindow("Detected Digits", cv2.WINDOW_NORMAL)

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

     # check key presses immediately
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        detected_number_symbols = ""
        error_displayed = False
        clear_image = np.zeros((50, 500, 3), dtype=np.uint8)
        cv2.imshow("Detected Digits", clear_image)
    elif key == ord('x'):
        detected_number_symbols = detected_number_symbols[:-1]
        error_displayed = False
    elif key == ord('v'):
        isdone = False
        detected_number_symbols = ""
        error_displayed = False
        clear_image = np.zeros((50, 500, 3), dtype=np.uint8)
        cv2.imshow("Detected Digits", clear_image)
    elif key == ord('q'):
        break

    # convert frame to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if hand_cooldown == 0:
        keypoints = extract_keypoints(frame)
        if keypoints is not None and isdone == False:
            keypoints = keypoints.reshape(1, -1)
            prediction = model.predict(keypoints)
            predicted_class = np.argmax(prediction)

          
            if predicted_class == 10:
                detected_number_symbols += str ("+")
                
            elif predicted_class == 11:
                detected_number_symbols += str ("-")
                
            elif predicted_class == 12:
                detected_number_symbols += str ("*")
                
            elif predicted_class == 13:
                detected_number_symbols += str ("/")
                
            elif predicted_class == 14:
                detected_number_symbols += str ("(")
                
            elif predicted_class == 15:
                detected_number_symbols += str (")")

            elif predicted_class == 16:
                if detected_number_symbols:
                    try:
                        result = eval(detected_number_symbols)
                        detected_number_symbols += "=" + str(result)
                        isdone = True
                    except Exception:
                        detected_number_symbols = "Error"
                        error_displayed = True
                else:
                    detected_number_symbols = "Error"
                    error_displayed = True
            else:
                if error_displayed:
                    detected_number_symbols = str(predicted_class)
                    error_displayed = False
                else:
                    detected_number_symbols += str(predicted_class)

        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        hand_cooldown = 60

    else:
        hand_cooldown -= 1
        frame_height = frame.shape[0]

    text_image = np.zeros((50, 500, 3), dtype=np.uint8)
    cv2.putText(text_image, f'{detected_number_symbols}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Detected Digits", text_image)

    # find hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # display result
    cv2.imshow('Hand Gesture Recognition', frame)

# release webcam
cap.release()

cv2.destroyAllWindows()

print("Detected digits:", detected_number_symbols)