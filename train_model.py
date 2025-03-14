import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from preprocessing_utils import preprocess_image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import cv2

# Define dataset path
DATASET_PATH = "dataset/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = (64, 64)  # Resize images
BATCH_SIZE = 32
EPOCHS = 10  # Increase for better accuracy

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  # Random rotations up to 20 degrees
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Mirror images horizontally
    brightness_range=(0.7, 1.3),  # Vary brightness
    fill_mode='nearest',  # Fill strategy for created pixels
    validation_split=0.2  # 80% train, 20% validation
)

# Validation data should only be rescaled, not augmented
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# For training, use augmentation
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"
)

# For validation, use the separate validation generator
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = len(train_generator.class_indices)

# Define CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # 26 letters A-Z
])

# Compile the Model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Define a custom data generator function that uses our preprocessing
def custom_generator(directory, batch_size, target_size, class_indices, augment=True):
    class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    class_dirs = [d for d in class_dirs if d in class_indices]
    
    # Debug output - check what directories are found
    print(f"Found {len(class_dirs)} class directories: {class_dirs}")
    
    num_classes = len(class_indices)
    
    while True:
        batch_images = []
        batch_labels = []
        
        # Fill a batch
        for _ in range(batch_size):
            # Select a random class
            if not class_dirs:
                raise ValueError(f"No valid class directories found in {directory}. Check your dataset structure.")
                
            class_dir = random.choice(class_dirs)
            class_path = os.path.join(directory, class_dir)
            
            # Select a random image from this class
            img_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not img_files:
                print(f"Warning: No image files found in {class_path}")
                continue  # Skip this iteration and try another class
                
            img_file = random.choice(img_files)
            img_path = os.path.join(class_path, img_file)
            
            # Load and preprocess the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            processed_img = preprocess_image(image, augment=augment, target_size=target_size)
            
            # Get the label
            label = to_categorical(class_indices[class_dir], num_classes=num_classes)
            
            batch_images.append(processed_img)
            batch_labels.append(label)
            
            # If we couldn't fill the batch, break and return what we have
            if len(batch_images) == batch_size:
                break
        
        # If we couldn't get any valid images, raise an error
        if not batch_images:
            raise ValueError("Could not find any valid images for training")
            
        yield np.array(batch_images), np.array(batch_labels)

# Get class indices from directory structure
temp_datagen = ImageDataGenerator()
temp_generator = temp_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="categorical"
)
class_indices = temp_generator.class_indices

# Create custom generators with our preprocessing
train_gen = custom_generator(
    DATASET_PATH, 
    BATCH_SIZE, 
    IMG_SIZE, 
    class_indices, 
    augment=True
)

val_gen = custom_generator(
    DATASET_PATH, 
    BATCH_SIZE, 
    IMG_SIZE, 
    class_indices, 
    augment=False
)

# Update the training call
steps_per_epoch = int(len(temp_generator.filenames) * 0.8 // BATCH_SIZE)
validation_steps = int(len(temp_generator.filenames) * 0.2 // BATCH_SIZE)

model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS
)

# Save the Model
model.save("asl_model.h5")
print("Model saved as asl_model.h5")
