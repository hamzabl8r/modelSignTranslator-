import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './data'
y_true = []
y_pred = []

print("Starting evaluation...")

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            hand = detection_result.hand_landmarks[0]

            for landmark in hand:
                x_.append(landmark.x)
                y_.append(landmark.y)

            min_x, min_y = min(x_), min(y_)

            for landmark in hand:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                y_pred.append(str(prediction[0]))
                y_true.append(dir_)

if y_true and y_pred:
    accuracy = accuracy_score(y_true, y_pred)
    print(f'\nAccuracy on Test Set: {accuracy * 100:.2f}%')
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))
else:
    print("No valid predictions were made!")