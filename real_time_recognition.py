import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import json
import os
import time  # For unique filenames
from preprocessing_utils import preprocess_image

# Load trained ASL model
model = tf.keras.models.load_model("asl_model.h5")

# Load ASL class labels from JSON file
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}  # Reverse key-value pairs

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Create a folder to save cropped images (if it doesn't exist)
DEBUG_IMAGE_FOLDER = "real_time_video_test_set/O"
os.makedirs(DEBUG_IMAGE_FOLDER, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(0)

MARGIN = 80  # Add padding to ensure the full hand is captured

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a natural selfie-view
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get bounding box of hand
            h, w, c = frame.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Add margin
            x_min = max(0, x_min - MARGIN)
            x_max = min(w, x_max + MARGIN)
            y_min = max(0, y_min - MARGIN)
            y_max = min(h, y_max + MARGIN)
            
            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max].copy()
            
            if hand_img.size != 0:  # Check if the crop is valid
                # Use our robust preprocessing function (without augmentation for inference)
                processed_img = preprocess_image(hand_img, augment=False, target_size=(64, 64))
                processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
                
                # Make prediction
                prediction = model.predict(processed_img)
                predicted_class_idx = np.argmax(prediction[0])
                predicted_letter = class_labels[predicted_class_idx]
                confidence = prediction[0][predicted_class_idx]
                
                # Display detected sign on the frame
                cv2.putText(frame, f"Predicted: {predicted_letter} (Confidence: {confidence:.2f})", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow("ASL Real-Time Recognition", frame)

    # Press "q" to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

print(f"Cropped hand images saved in: {DEBUG_IMAGE_FOLDER}")
