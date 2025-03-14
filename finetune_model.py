import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import json
from preprocessing_utils import preprocess_image

# Constants
FINETUNE_DIR = "finetune_dataset/"  # Use the split fine-tuning dataset
IMG_SIZE = (64, 64)
BATCH_SIZE = 16  # Smaller batch size for fine-tuning
EPOCHS = 100
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning

# Load the baseline model
baseline_model = tf.keras.models.load_model("transfer_learning_baseline_model.h5")

# Load class labels to check which classes we have in our fine-tuning data
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
    
# Check available classes in the fine-tuning dataset
available_classes = [d for d in os.listdir(FINETUNE_DIR) if os.path.isdir(os.path.join(FINETUNE_DIR, d))]
print(f"Fine-tuning on {len(available_classes)} classes: {', '.join(available_classes)}")

# Create a mapping between available classes and indices
available_class_indices = {cls: idx for idx, cls in enumerate(sorted(available_classes))}
print("New class mapping:")
for cls, idx in available_class_indices.items():
    print(f"  {cls} -> {idx}")

# Save this mapping for later evaluation
with open("finetune_class_indices.json", "w") as f:
    json.dump(available_class_indices, f)

# Create data generator with preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=lambda img: preprocess_image(img, augment=True),
    validation_split=0.1  # Small validation split for monitoring progress
)

# Flow from directory with preprocessing
train_generator = datagen.flow_from_directory(
    FINETUNE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=sorted(available_classes)  # Ensure consistent class ordering
)

val_generator = datagen.flow_from_directory(
    FINETUNE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    classes=sorted(available_classes)  # Ensure consistent class ordering
)

# CREATE A NEW MODEL APPROACH
# Get the layers from the base model except the last dense layer
base_layers = baseline_model.layers[:-1]  

# Build a new model
finetuned_model = tf.keras.Sequential()

# Add all layers except the output layer
for layer in base_layers:
    finetuned_model.add(layer)
    layer.trainable = False  # Freeze the layer
    
# Add a new output layer with the correct number of classes
finetuned_model.add(tf.keras.layers.Dense(len(available_classes), activation='softmax', name='fine_tuned_output'))

# Compile the model with a lower learning rate
finetuned_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Print model summary to verify
finetuned_model.summary()

# Create a callback to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "asl_model_finetuned.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Fine-tune the model
history = finetuned_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Save the final model (in case the best one wasn't saved by the callback)
finetuned_model.save("asl_model_finetuned_final.h5")
print("Fine-tuning complete. Model saved.")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('finetune_history.png')
plt.show() 