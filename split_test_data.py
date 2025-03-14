import os
import shutil
import random
import numpy as np

# Define paths
REAL_TIME_DATA_DIR = "real_time_video_test_set"
FINETUNE_DIR = "finetune_dataset"
EVAL_DIR = "eval_dataset"

# Create directories
os.makedirs(FINETUNE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Track available classes
available_classes = []

# Split each class directory
for class_name in os.listdir(REAL_TIME_DATA_DIR):
    class_path = os.path.join(REAL_TIME_DATA_DIR, class_name)
    
    # Skip non-directory items
    if not os.path.isdir(class_path):
        continue
        
    # Get list of image files
    img_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Skip empty directories
    if not img_files:
        print(f"Skipping empty directory: {class_name}")
        continue
        
    # Create class directories in both finetune and eval
    finetune_class_dir = os.path.join(FINETUNE_DIR, class_name)
    eval_class_dir = os.path.join(EVAL_DIR, class_name)
    os.makedirs(finetune_class_dir, exist_ok=True)
    os.makedirs(eval_class_dir, exist_ok=True)
    
    # Track this class
    available_classes.append(class_name)
    
    # Randomly shuffle files
    random.shuffle(img_files)
    
    # Split files 80/20
    split_idx = int(len(img_files) * 0.8)
    finetune_files = img_files[:split_idx]
    eval_files = img_files[split_idx:]
    
    # Copy files to respective directories
    for f in finetune_files:
        shutil.copy(os.path.join(class_path, f), os.path.join(finetune_class_dir, f))
    
    for f in eval_files:
        shutil.copy(os.path.join(class_path, f), os.path.join(eval_class_dir, f))
    
    print(f"Class {class_name}: {len(finetune_files)} finetune, {len(eval_files)} eval")

print(f"\nAvailable classes for fine-tuning: {len(available_classes)}")
print(", ".join(available_classes)) 