import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SETTINGS ---
DATA_DIR = './data'
data = []
labels = []

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path): continue
    
    print(f'Processing: {dir_}')
    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            data_aux = []
            x_all = [l.x for hl in results.hand_landmarks for l in hl]
            y_all = [l.y for hl in results.hand_landmarks for l in hl]

            min_x, min_y = min(x_all), min(y_all)

            for i in range(1):
                if i < len(results.hand_landmarks):
                    for landmark in results.hand_landmarks[i]:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)
                else:
                    data_aux.extend([0.0] * 21) # Padding for one hand

            if len(data_aux) == 42 :
                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Dataset updated to 84 features.")