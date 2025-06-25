import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras import ImageDataGenerator  # type: ignore

# Configuration
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20

# Dataset path
dataset_path = "data"

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_path, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

# Model using MobileNetV2
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

model = Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy")

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Evaluate model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
