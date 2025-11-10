import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def docbien(model, class_labels):
    TEST_FOLDER = "kytucut"

    def predict_char(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {image_path}")
            return None

        IMG_SIZE = (32, 32)
        img_resized = cv2.resize(img, IMG_SIZE)
        img_resized = cv2.bitwise_not(img_resized)
        img_input = img_resized.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=(0, -1))
        pred = model.predict(img_input, verbose=0)
        pred_idx = np.argmax(pred, axis=1)[0]
        label = class_labels[pred_idx]
        return label

    filenames = sorted(os.listdir(TEST_FOLDER))
    results = []
    for filename in filenames:
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(TEST_FOLDER, filename)
            label = predict_char(path)
            if label is not None:
                results.append((filename, label))

    plate_number = ''.join([label for _, label in results])
    return plate_number
