import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Set dataset path
DATASET_PATH = "C:/Users/Sikkandhar Jabbar/Desktop/alzi1/input"

# Define image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50  # Increased epochs for better convergence
NUM_CLASSES = 4  # Mild, Moderate, Non, Very Mild Dementia

# Data Augmentation & Loading (Stronger Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=0.2  # 80% Train, 20% Validation
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Load Pretrained EfficientNetB3 (Train More Layers)
base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers[:-20]:  # Unfreeze last 20 layers for fine-tuning
    layer.trainable = True

# Build Custom Model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)  # Stronger dropout for generalization
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile Model (Using AdamW Optimizer)
model.compile(optimizer=AdamW(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks for Improved Performance
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
]

# Display Model Summary
model.summary()

# Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save the Model
model.save("efficientnet_alzheimers_improved.h5")

# Save Training History
np.save("training_history_improved.npy", history.history)

# Function to Plot Training Performance
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy Graph
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")

    # Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    plt.show()

# Plot Training Performance
plot_training_history(history)

# Display Model Parameters
total_params = model.count_params()
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
non_trainable_params = total_params - trainable_params

print("\nModel Parameters:")
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Non-Trainable Parameters: {non_trainable_params}")
