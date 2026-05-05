import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load el model mte3ek
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# 2. Config mte3 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

DATA_DIR = './DataSet/test' # el path mte3 dossier el test
y_true = []
y_pred = []

print("Starting evaluation...")

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Nafs el logic mte3 el training: nal9aw min_x w min_y
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

            min_x, min_y = min(x_), min(y_)

            # Na3mlo extract lel landmarks (l-2 hands kif kif)
            for i in range(2):
                if i < len(results.multi_hand_landmarks):
                    for landmark in results.multi_hand_landmarks[i].landmark:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)
                else:
                    data_aux.extend([0.0] * 42) # Padding ken famma id wahda

            # Prediction
            if len(data_aux) == 84:
                prediction = model.predict([np.asarray(data_aux)])
                y_pred.append(str(prediction[0]))
                y_true.append(dir_)

# 3. Affichage mte3 el Results
accuracy = accuracy_score(y_true, y_pred)
print(f'\nAccuracy on Test Set: {accuracy * 100:.2f}%')

# Beche tchouf anahi el kelmet li el model ghalet fihom
from sklearn.metrics import classification_report
print("\nDetailed Report:")
print(classification_report(y_true, y_pred))