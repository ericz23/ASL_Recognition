import tensorflow as tf
import numpy as np
import cv2
import json
import os
from sklearn.metrics import classification_report, accuracy_score

# Load the trained CNN model
model = tf.keras.models.load_model("asl_model.h5")

# Load class labels from JSON file
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}  # Reverse key-value pairs

# Define test dataset directory
TEST_DIR = "real_time_video_test_set/"
IMG_SIZE = (64, 64)  # Target image size

# Initialize lists for storing true labels and predictions
true_labels = []
predicted_labels = []

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Resize & normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Loop through each category (A-Z, space, nothing, del)
for class_name in os.listdir(TEST_DIR):
    class_path = os.path.join(TEST_DIR, class_name)

    # Skip non-directory files
    if not os.path.isdir(class_path):
        continue

    # Get the class index from the label mapping
    if class_name in class_indices:
        class_index = class_indices[class_name]
    else:
        print(f"Warning: {class_name} not found in class_labels.json. Skipping.")
        continue

    # Loop through images in the class folder
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        # Read and preprocess the image
        hand_img = cv2.imread(image_path)
        if hand_img is None:
            print(f"Warning: Unable to read {image_path}. Skipping.")
            continue

        # Mirror the image horizontally
        mirrored_img = cv2.flip(hand_img, 1) 

        # Preprocess both original and mirrored images
        original_processed = preprocess(hand_img)
        mirrored_processed = preprocess(mirrored_img)

        # Predict for original image
        original_pred = model.predict(original_processed)
        original_pred_index = np.argmax(original_pred)

        # Predict for mirrored image (treated as separate test sample)
        mirrored_pred = model.predict(mirrored_processed)
        mirrored_pred_index = np.argmax(mirrored_pred)

        # Store both predictions as separate test samples
        true_labels.append(class_index)
        predicted_labels.append(original_pred_index)

        true_labels.append(class_index)
        predicted_labels.append(mirrored_pred_index)

# Compute overall accuracy
overall_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Compute accuracy per class
print("\nAccuracy Per Class:")
class_accuracies = {}
for class_index, class_name in class_labels.items():
    class_mask = (np.array(true_labels) == class_index)
    class_accuracy = accuracy_score(np.array(true_labels)[class_mask], np.array(predicted_labels)[class_mask])
    class_accuracies[class_name] = class_accuracy
    print(f"{class_name}: {class_accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=list(class_labels.values())))
