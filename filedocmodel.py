import cv2
import numpy as np

def docbien(model, char_img, class_labels):
    # 1. Nếu ảnh đang là BGR → convert sang GRAY
    if len(char_img.shape) == 3:
        img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        img = char_img

    # 2. Resize theo đúng model bạn đang dùng (32×32)
    IMG_SIZE = (32, 32)
    img_resized = cv2.resize(img, IMG_SIZE)

    # 3. Invert để giống lúc bạn train (trắng = nét)
    img_resized = cv2.bitwise_not(img_resized)

    # 4. Normalized
    img_input = img_resized.astype("float32") / 255.0

    # 5. Expand cho đúng shape (1, 32, 32, 1)
    img_input = np.expand_dims(img_input, axis=(0, -1))

    # 6. Predict
    pred = model.predict(img_input, verbose=0)
    pred_idx = np.argmax(pred, axis=1)[0]

    # 7. Trả về nhãn ký tự
    return class_labels[pred_idx]
