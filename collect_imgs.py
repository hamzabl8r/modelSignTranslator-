import os
import cv2
import time

# --- SETTINGS ---
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Your updated list of words
classes = ['HELLO', 'THANK_YOU', 'YES', 'NO', 'YALLA', 'FRIEND', '5ODH', 'NOTHING','CAR'] 
images_per_class = 200  # Total images needed for a strong static model

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    # CHECK: Do we already have enough images for this word?
    existing_imgs = len([name for name in os.listdir(class_path) if name.endswith('.jpg')])
    
    if existing_imgs >= images_per_class:
        print(f'>>> Skipping "{class_name}" (Already have {existing_imgs} images)')
        continue

    print(f'\n--- Collecting for NEW WORD: {class_name} ---')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f'Ready to collect: {class_name}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Press "Q" to start recording', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Collector', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Record loop
    counter = existing_imgs # Start from where we left off if folder was partial
    while counter < images_per_class:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        
        cv2.putText(frame, f'RECORDING: {class_name}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Image {counter}/{images_per_class}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Collector', frame)
        cv2.waitKey(25)
        counter += 1

print("\nAll new words collected!")
cap.release()
cv2.destroyAllWindows()