import cv2
import numpy as np
import tensorflow as tf
import random

def preprocess_image(image, augment=False, target_size=(64, 64)):
    """Simplified preprocessing without YCrCb conversion"""
    # Check if image is None
    if image is None:
        raise ValueError("Input image is None")
        
    # Convert to RGB if needed
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # Handle RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3 and isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Apply random transformations during training to improve robustness
    if augment:
        # Ensure image is uint8
        image = image.astype(np.uint8)
        
        # Random noise to simulate different cameras/conditions
        if random.random() > 0.5:
            noise = np.random.normal(0, random.uniform(1, 10), image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        # Random blur to simulate different focus levels
        if random.random() > 0.7:
            image = cv2.GaussianBlur(image, (5, 5), random.uniform(0.5, 1.5))
        
        # Random shadow effect
        if random.random() > 0.8:
            rows, cols = image.shape[:2]
            top_y = np.random.randint(0, rows//2)
            bottom_y = np.random.randint(rows//2, rows)
            shadow_value = random.uniform(0.5, 0.9)
            mask = np.zeros_like(image)
            mask[top_y:bottom_y, :] = 1
            image = (image * (1 - mask * (1 - shadow_value))).astype(np.uint8)
        
        # Simple alternative to histogram equalization
        if random.random() > 0.5:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # for each channel separately
            for i in range(3):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                image[:,:,i] = clahe.apply(image[:,:,i])
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image 