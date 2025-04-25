import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.preprocessing import image

print("[INFO] TensorFlow version:", tf.__version__)


model_json_path = "Facial Expression Recognition.json"
model_weights_path = "fer.h5"

if not os.path.exists(model_json_path):
    print(f"[ERROR] Model JSON file '{model_json_path}' not found.")
    exit()

if not os.path.exists(model_weights_path):
    print(f"[ERROR] Model weights file '{model_weights_path}' not found.")
    exit()


try:
    print("[INFO] Loading model architecture...")
    with open(model_json_path, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json, custom_objects={'Sequential': Sequential})
    print("[INFO] Model architecture loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model architecture: {e}")
    exit()


try:
    model.load_weights(model_weights_path)
    print("[INFO] Model weights loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model weights: {e}")
    exit()


cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print(f"[ERROR] Haar cascade file '{cascade_path}' not found.")
    exit()

face_haar_cascade = cv2.CascadeClassifier(cascade_path)
print("[INFO] Haar cascade loaded successfully.")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Unable to access the webcam.")
    exit()

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
print("[INFO] Webcam started. Press 'q' to exit.")

while True:
    ret, test_img = cap.read()
    if not ret:
        print("[WARNING] Frame capture failed, skipping...")
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]

        try:
            roi_gray = cv2.resize(roi_gray, (48, 48))
        except Exception as e:
            print(f"[WARNING] Face crop resize failed: {e}")
            continue

        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        try:
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]
            cv2.putText(test_img, predicted_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")

    resized_img = cv2.resize(test_img, (1300, 800))
    cv2.imshow('Facial Emotion Analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):
        print("[INFO] Exiting on user request...")
        break

cap.release()
cv2.destroyAllWindows()
