import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================================
# 1️⃣ Đường dẫn
# ================================
MODEL_PATH = "models/char_cnn_model.h5"   # Mô hình đã train
TEST_IMG   = 'chudatach/char_12.jpg'     # Ảnh ký tự muốn test

# ================================
# 2️⃣ Định nghĩa class labels (ví dụ)
# Nếu bạn train 1 chữ hoặc 1 số, điền nhãn đó vào list
# ================================
# Ví dụ train các ký tự: 0-9 + A-Z
class_labels = ['0','1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','I','J',
                'K','L','M','N','O','P','Q','R','S','T',
                'U','V','W','X','Y','Z']

# Nếu bạn train chỉ 1 chữ hoặc 1 số, ví dụ 'E':
# class_labels = ['E']

# ================================
# 3️⃣ Load mô hình
# ================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy mô hình: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Mô hình đã tải thành công!")

# ================================
# 4️⃣ Load ảnh và chuẩn hóa
# ================================
img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
# img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
# img=cv2.threshold(img,175,255,cv2.THRESH_BINARY_INV)
if img is None:
    raise ValueError("❌ Không đọc được ảnh!")

# Resize về đúng kích thước train
IMG_SIZE = (112,112)   # phải cùng kích thước với lúc train
img_resized = cv2.resize(img, IMG_SIZE)

# Chuẩn hóa và reshape
img_input = img_resized.astype("float32") / 255.0
img_input = np.expand_dims(img_input, axis=(0,-1))  # shape: (1,32,32,1)

print(f"✅ Ảnh test đã được chuẩn bị (kích thước: {img_input.shape})")

# ================================
# 5️⃣ Dự đoán ký tự
# ================================
pred = model.predict(img_input)
pred_idx = np.argmax(pred, axis=1)[0]
label = class_labels[pred_idx]

print("🔹 Ký tự nhận diện:", label)

# ================================
# 6️⃣ Hiển thị ảnh với nhãn
# ================================
cv2.putText(img, label, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imshow("Kết quả nhận diện", img)
cv2.waitKey(0)
cv2.destroyAllWindows()