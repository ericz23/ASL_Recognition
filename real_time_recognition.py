import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import json


# Load trained ASL model
model = tf.keras.models.load_model("asl_model.h5")

# Load class labels from JSON file
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}  # Reverse key-value pairs

print("Class Labels Loaded:", class_labels)

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

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
            # Extract hand bounding box
            h, w, c = frame.shape
            MARGIN = 20  # Add padding around the hand
            
            # Crop hand region with margin
            x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w) - MARGIN)
            y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h) - MARGIN)
            x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w) + MARGIN)
            y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h) + MARGIN)

            # Crop the hand region
            hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                # Preprocess the cropped image
                hand_img = cv2.resize(hand_img, (64, 64)) / 255.0  # Resize & normalize
                hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

                # Predict the ASL sign
                prediction = model.predict(hand_img)
                predicted_index = np.argmax(prediction)
                predicted_label = class_labels.get(predicted_index, "Unknown")

                # Display detected sign on the frame
                cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the webcam feed
    cv2.imshow("ASL Real-Time Recognition", frame)

    # Press "q" to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
