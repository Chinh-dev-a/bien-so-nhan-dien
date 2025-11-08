import os
import cv2
import numpy as np
from fontTools.misc.cython import returns
from tensorflow.keras.models import load_model

# ================================
# 1️⃣ Đường dẫn
# ================================
def docbien(model,class_labels):
    # MODEL_PATH = "models/char_cnn_model.h5"   # Mô hình CNN
    TEST_FOLDER = "kytucut"                 # Thư mục chứa ảnh ký tự

    # ================================
    # 2️⃣ Danh sách nhãn (label)
    # ================================
    # class_labels = ['0','1','2','3','4','5','6','7','8','9',
    #                 'A','B','C','D','E','F','G','H',
    #                 'K','L','M','N','P','R','T',
    #                 'U','V','X','Y']
    #
    # # ================================
    # # 3️⃣ Load mô hình
    # # ================================
    # if not os.path.exists(MODEL_PATH):
    #     raise FileNotFoundError(f"❌ Không tìm thấy mô hình: {MODEL_PATH}")
    #
    # model = load_model(MODEL_PATH)
    # print("✅ Mô hình đã tải thành công!")


    # ================================
    # 4️⃣ Hàm dự đoán từng ảnh
    # ================================
    def predict_char(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {image_path}")
            return None

        # Resize về kích thước lúc train
        IMG_SIZE = (32, 32)
        img_resized = cv2.resize(img, IMG_SIZE)
        img_resized = cv2.bitwise_not(img_resized)

        # Chuẩn hóa
        img_input = img_resized.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=(0, -1))  # (1, 32, 32, 1)

        # Dự đoán
        pred = model.predict(img_input, verbose=0)
        pred_idx = np.argmax(pred, axis=1)[0]
        label = class_labels[pred_idx]
        return label


    # ================================
    # 5️⃣ Duyệt ảnh trong thư mục theo thứ tự
    # ================================
    # Nếu file ảnh có dạng: char_1.jpg, char_2.jpg, ... thì nên sắp xếp theo tên
    filenames = sorted(os.listdir(TEST_FOLDER))

    results = []
    for filename in filenames:
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(TEST_FOLDER, filename)
            label = predict_char(path)
            if label is not None:
                results.append((filename, label))
                # print(f"Ảnh {filename:20} ➜ {label}")

    # ================================
    # 6️⃣ Ghép ký tự thành chuỗi (biển số)
    # ================================
    # Lấy nhãn theo thứ tự
    plate_number = ''.join([label for _, label in results])

    # print("\n📋 KẾT QUẢ NHẬN DIỆN THEO THỨ TỰ:")
    # for fname, label in results:
    #     print(f"{fname:<20} → {label}")

    # print("\n🚗 Biển số (ghép lại):", plate_number)
    return plate_number
