import tensorflow as tf
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from preprocessing_utils import preprocess_image
from tqdm import tqdm
import time

# Define paths
EVAL_DIR = "eval_dataset/"  # Use the separate evaluation dataset
BASELINE_MODEL_PATH = "asl_model_baseline.h5"
FINETUNED_MODEL_PATH = "asl_model_finetuned_final.h5"
IMG_SIZE = (64, 64)

# Load class labels
with open("finetune_class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Load the class mapping used for fine-tuning
FINETUNE_CLASS_PATH = "finetune_class_indices.json" 
with open(FINETUNE_CLASS_PATH, "r") as f:
    finetune_class_indices = json.load(f)
finetune_labels = {v: k for k, v in finetune_class_indices.items()}

# Map original class indices to fine-tuned indices
class_map = {}
for class_name, finetune_idx in finetune_class_indices.items():
    if class_name in class_indices:
        orig_idx = class_indices[class_name]
        class_map[orig_idx] = finetune_idx

# Load both models for comparison
baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
finetuned_model = tf.keras.models.load_model(FINETUNED_MODEL_PATH)

# Function to preprocess images
def preprocess(img):
    return preprocess_image(img, augment=False, target_size=IMG_SIZE)

# Store results for both models
results = {
    "baseline": {"true": [], "pred": []},
    "finetuned": {"true": [], "pred": []}
}

# Count total images first
total_images = 0
for class_name in os.listdir(EVAL_DIR):
    class_path = os.path.join(EVAL_DIR, class_name)
    if os.path.isdir(class_path):
        total_images += len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

print(f"Evaluating {total_images} images...")
processed = 0

# Process each class
for class_name in os.listdir(EVAL_DIR):
    class_path = os.path.join(EVAL_DIR, class_name)
    
    # Skip non-directories
    if not os.path.isdir(class_path):
        continue
        
    # Get the class index
    if class_name not in class_indices:
        print(f"Warning: Class {class_name} not found in class_labels.json. Skipping.")
        continue
        
    class_idx = class_indices[class_name]
    img_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing {len(img_files)} images for class {class_name}...")
    
    # Process each image in the class folder
    for img_file in img_files:
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # Preprocess the image
        processed_img = preprocess(img)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Get predictions from both models
        baseline_pred = baseline_model.predict(processed_img, verbose=0)
        baseline_class = np.argmax(baseline_pred[0])
        
        # Get predictions from finetuned model
        finetuned_pred = finetuned_model.predict(processed_img, verbose=0)
        finetuned_finetune_class = np.argmax(finetuned_pred[0])
        
        # Map back to original class space for comparison
        finetuned_class_name = finetune_labels[finetuned_finetune_class]
        finetuned_class = class_indices[finetuned_class_name]
        
        # Store results
        results["baseline"]["true"].append(class_idx)
        results["baseline"]["pred"].append(baseline_class)
        
        results["finetuned"]["true"].append(class_idx)
        results["finetuned"]["pred"].append(finetuned_class)
        
        # Update progress
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{total_images} images ({processed/total_images*100:.1f}%)")

# Calculate metrics
for model_name, data in results.items():
    accuracy = accuracy_score(data["true"], data["pred"])
    print(f"\n{model_name.title()} Model Accuracy: {accuracy:.4f}")
    
    print(f"\n{model_name.title()} Classification Report:")
    class_names = [class_labels.get(i, f"Unknown-{i}") for i in range(max(max(data["true"]), max(data["pred"])) + 1)]
    print(classification_report(data["true"], data["pred"], target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(data["true"], data["pred"])
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name.title()} Model Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")

# Compare the improvement per class
baseline_correct = np.array(results["baseline"]["pred"]) == np.array(results["baseline"]["true"])
finetuned_correct = np.array(results["finetuned"]["pred"]) == np.array(results["finetuned"]["true"])

# Calculate per-class accuracy for both models
class_improvements = {}
for class_idx in set(results["baseline"]["true"]):
    class_name = class_labels.get(class_idx, f"Unknown-{class_idx}")
    baseline_mask = np.array(results["baseline"]["true"]) == class_idx
    
    if np.sum(baseline_mask) == 0:
        continue
        
    baseline_acc = np.mean(baseline_correct[baseline_mask])
    finetuned_acc = np.mean(finetuned_correct[baseline_mask])
    improvement = finetuned_acc - baseline_acc
    
    class_improvements[class_name] = {
        "baseline_acc": baseline_acc,
        "finetuned_acc": finetuned_acc,
        "improvement": improvement
    }

# Plot improvements
plt.figure(figsize=(14, 8))
classes = list(class_improvements.keys())
baseline_accs = [class_improvements[c]["baseline_acc"] for c in classes]
finetuned_accs = [class_improvements[c]["finetuned_acc"] for c in classes]

x = np.arange(len(classes))
width = 0.35

plt.bar(x - width/2, baseline_accs, width, label='Baseline')
plt.bar(x + width/2, finetuned_accs, width, label='Fine-tuned')

plt.xlabel('ASL Sign')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison by Class')
plt.xticks(x, classes, rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

print("\nTransfer learning improvement summary:")
for class_name, data in sorted(class_improvements.items(), key=lambda x: -x[1]["improvement"]):
    print(f"{class_name}: {data['baseline_acc']:.2f} → {data['finetuned_acc']:.2f} ({data['improvement']*100:+.1f}%)") 