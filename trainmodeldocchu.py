import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "testbiensoxe/datatrain"
TEST_DIR = "testbiensoxe/datatest"

# ================================
# 1️⃣ CẤU HÌNH CNN SIZE 32×32
# ================================
IMG_SIZE = (32, 32)
BATCH_SIZE = 16
EPOCHS = 15

# ================================
# 2️⃣ TẠO DATA GENERATOR
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ================================
# 3️⃣ XÂY DỰNG CNN INPUT 32×32×1
# ================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),


    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ================================
# 4️⃣ HUẤN LUYỆN
# ================================
print("\n Bắt đầu huấn luyện mô hình...\n")
history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen)

# ================================
# 5️⃣ LƯU MODEL
# ================================
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/char_cnn_model.h5"
model.save(MODEL_PATH)

print(f"\n✅ Đã lưu mô hình tại: {MODEL_PATH}")

# ================================
# 6️⃣ IN CLASS INDEX
# ================================
class_indices = train_gen.class_indices
print("\nSố lớp:", train_gen.num_classes)
print("Các lớp:", list(class_indices.keys()))

