import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import json
import os
import time  # For unique filenames

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
DEBUG_IMAGE_FOLDER = "real_time_video_test_set/H"
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
            # Get image dimensions
            h, w, c = frame.shape

            # Get bounding box coordinates
            x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w) - MARGIN)
            y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h) - MARGIN)
            x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w) + MARGIN)
            y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h) + MARGIN)

            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Check if the extracted image is valid
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                # Save the cropped hand image 
                #timestamp = int(time.time() * 1000)  # Unique timestamp
                #image_path = os.path.join(DEBUG_IMAGE_FOLDER, f"hand_{timestamp}.jpg")
                #cv2.imwrite(image_path, hand_img)

                # Preprocess for prediction
                hand_img = cv2.resize(hand_img, (64, 64)) / 255.0  # Resize & normalize
                hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

                # Predict the ASL sign
                prediction = model.predict(hand_img)
                predicted_index = np.argmax(prediction)
                predicted_label = class_labels.get(predicted_index, "Unknown")

                # Display detected sign on the frame
                cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the webcam feed
    cv2.imshow("ASL Real-Time Recognition", frame)

    # Press "q" to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

print(f"Cropped hand images saved in: {DEBUG_IMAGE_FOLDER}")
