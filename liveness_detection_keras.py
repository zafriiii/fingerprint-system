
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import os

# Configuration
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10

# 1. Data Preparation
dataset_path = 'path_to_fingerprint_dataset'  # Should have 'train/live', 'train/spoof', 'val/live', 'val/spoof'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_path, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 2. Model Architecture (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # For binary classification
])

# 3. Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train Model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 5. Evaluate Model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
