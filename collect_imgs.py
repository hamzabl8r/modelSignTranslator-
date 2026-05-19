import os
import sys
import cv2

# --- SETTINGS ---
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)


DEFAULT_CLASSES = ['HELLO', 'THANK_YOU', 'YES', 'NO', 'YALLA', 'FRIEND', 'NOTHING', 'CAR',"HOW ARE YOU", "GOOD","BAD", "LOVE","HI"]

images_per_class = 200


# ==================== RESTORE DEFAULT CLASSES ====================

def restore_default_classes():
    """
    Recrée les dossiers manquants pour chaque mot de DEFAULT_CLASSES.
    Si un dossier a été supprimé, il sera recréé ici au prochain lancement.
    """
    restored = []
    for class_name in DEFAULT_CLASSES:
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            restored.append(class_name)

    if restored:
        print(f"🔄 Dossiers recréés (mots par défaut restaurés) : {', '.join(restored)}")
    else:
        print("✅ Tous les mots par défaut sont déjà présents.")


restore_default_classes()

# La liste active = DEFAULT_CLASSES (on collecte uniquement les mots par défaut)
classes = DEFAULT_CLASSES


# ==================== WEBCAM ====================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Could not open webcam. Check your camera index.")
    sys.exit(1)

try:
    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_path, exist_ok=True)

        existing_imgs = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if existing_imgs >= images_per_class:
            print(f'>>> Skipping "{class_name}" (already have {existing_imgs} images)')
            continue

        print(f'\n--- Collecting for: {class_name} ---')

        # Wait for user to press Q to start
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera.")
                sys.exit(1)

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'Ready: {class_name}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Press "Q" to start recording', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Collector', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Record loop — resume from where we left off
        counter = existing_imgs
        while counter < images_per_class:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame read failed during recording.")
                break

            frame = cv2.flip(frame, 1)
            save_path = os.path.join(class_path, f'{counter}.jpg')
            cv2.imwrite(save_path, frame)

            cv2.putText(frame, f'RECORDING: {class_name}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Image {counter + 1}/{images_per_class}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Collector', frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                print(f'⏸  Recording paused at image {counter + 1}.')
                break

            counter += 1

    print("\n✅ All classes processed!")

finally:
    cap.release()
    cv2.destroyAllWindows()