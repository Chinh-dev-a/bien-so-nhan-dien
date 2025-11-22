import os
import glob
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# kích thước ký tự
digit_w = 30
digit_h = 60


# --------------------------
# HÀM ĐỌC DỮ LIỆU
# --------------------------
def get_digit_data(path):
    digit_list = []
    label_list = []

    # đọc số 0–9
    for number in range(10):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = cv2.resize(img, (digit_w, digit_h))  # resize đúng chuẩn

            digit_list.append(img)
            label_list.append(number)

    # đọc chữ A–Z
    for number in range(65, 91):
        ch = chr(number)
        for img_org_path in glob.iglob(path + ch + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = cv2.resize(img, (digit_w, digit_h))

            digit_list.append(img)
            label_list.append(number - 55)
            # MAP:
            # 'A' = 65 → class 10
            # 'B' = 66 → class 11
            # ...
            # 'Z' → class 35

    return digit_list, label_list


# --------------------------
# Lấy dữ liệu train + test
# --------------------------

path_train = "datatrain/"
path_test = "datatest/"

digit_train, label_train = get_digit_data(path_train)
digit_test, label_test = get_digit_data(path_test)

digit_train = np.array(digit_train)
digit_test = np.array(digit_test)

# Chuẩn hóa ảnh từ 0–255 → 0–1
digit_train = digit_train.astype("float32") / 255.0
digit_test = digit_test.astype("float32") / 255.0

# reshape thành định dạng CNN (H, W, channel)
digit_train = digit_train.reshape(-1, digit_h, digit_w, 1)
digit_test = digit_test.reshape(-1, digit_h, digit_w, 1)

# chuyển label sang one-hot 36 lớp
num_classes = 36
label_train = to_categorical(label_train, num_classes)
label_test = to_categorical(label_test, num_classes)

# --------------------------
# XÂY DỰNG MÔ HÌNH CNN
# --------------------------

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(digit_h, digit_w, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

# --------------------------
# BIÊN DỊCH MÔ HÌNH
# --------------------------
model.compile(
    optimizer=Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------
# TRAIN
# --------------------------
model.fit(
    digit_train, label_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1
)

# --------------------------
# ĐÁNH GIÁ
# --------------------------
loss, acc = model.evaluate(digit_test, label_test, verbose=0)
print("Accuracy =", acc)

# --------------------------
# LƯU MÔ HÌNH
# --------------------------
model.save("cnn_character_model.h5")
print("Model saved!")
