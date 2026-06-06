import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyttsx3 

engine = pyttsx3.init()
engine.setProperty('rate', 150) 

with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2 
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0) 

sentence = []
current_prediction = ""
last_added_word = ""
frame_counter = 0
STABILITY_THRESHOLD = 20 
CONFIDENCE_THRESHOLD = 0.65 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) 
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    
    results = detector.detect_for_video(mp_image, timestamp)
    predicted_word = ""
    confidence = 0

    if results.hand_landmarks:
        data_aux = []
        x_all = [l.x for hl in results.hand_landmarks for l in hl]
        y_all = [l.y for hl in results.hand_landmarks for l in hl]

        if x_all:
            min_x, min_y = min(x_all), min(y_all)
            for i in range(1):
                if i < len(results.hand_landmarks):
                    for landmark in results.hand_landmarks[i]:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)
                        cx, cy = int(landmark.x * W), int(landmark.y * H)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                else:
                    data_aux.extend([0.0] * 21)

            if len(data_aux) == 42:
                prediction_proba = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(prediction_proba)
                
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_word = str(prediction[0])
                else:
                    predicted_word = "..." 

        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, f"{predicted_word} ({int(confidence*100)}%)", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        last_added_word = ""

    if predicted_word not in ["", "..."]:
        if predicted_word == current_prediction:
            frame_counter += 1
        else:
            frame_counter = 0
            current_prediction = predicted_word
        if frame_counter == STABILITY_THRESHOLD:
            if predicted_word != last_added_word:
                sentence.append(predicted_word)
                last_added_word = predicted_word
            frame_counter = 0

    full_sentence = " ".join(sentence)
    cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, f"Output: {full_sentence}", (20, H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('ASL Word Translator', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'): 
        if sentence:
            engine.say(full_sentence); engine.runAndWait() 
            sentence = [] 
    elif key & 0xFF == ord('c'): sentence = []
    elif key & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()