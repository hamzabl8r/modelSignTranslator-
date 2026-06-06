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

# Fix 1: Absolute path for hand_landmarker.task
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"hand_landmarker.task not found at {model_path}\n"
        "Download it with:\n"
        "Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -OutFile hand_landmarker.task"
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# Fix 2: Track skipped images for debugging
skipped = 0
total = 0

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f'Processing: {dir_}')
    class_count = 0

    for img_file in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_file)
        img = cv2.imread(img_path)
        total += 1

        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        if not results.hand_landmarks:
            skipped += 1
            continue

        hand = results.hand_landmarks[0]  # one hand only

        # Fix 3: Normalize relative to wrist (landmark 0), not min_x/min_y
        # This is more stable and matches what the model should learn
        wrist_x = hand[0].x
        wrist_y = hand[0].y

        data_aux = []
        for landmark in hand:
            data_aux.append(landmark.x - wrist_x)
            data_aux.append(landmark.y - wrist_y)

        # 21 landmarks × 2 coords = 42 features
        if len(data_aux) == 42:
            data.append(data_aux)
            labels.append(dir_)
            class_count += 1

    print(f'  → {class_count} samples collected')

# Fix 4: Show class distribution before saving
import collections
counter = collections.Counter(labels)
print("\n--- Class Distribution ---")
for sign, count in sorted(counter.items()):
    print(f"  {sign}: {count} samples")

print(f"\nTotal: {len(data)} samples | Skipped: {skipped}/{total} images")

# Save
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset saved to data.pickle (42 features, wrist-normalized)")