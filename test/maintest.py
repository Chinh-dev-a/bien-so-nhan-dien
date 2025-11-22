import cv2
import os
from tensorflow.keras.models import load_model

from test import timbienso        # Hàm phát hiện vùng biển số (trả về ảnh biển số + check)
from tachkytu1 import tachkytu     # Hàm tách ký tự
from testread import docbien      # Hàm đọc ký tự và ghép chuỗi biển số


def main():
    # ================================
    # 1️⃣ Nạp cascade phát hiện biển số
    # ================================
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')

    # ================================
    # 2️⃣ Đọc 1 ảnh đầu vào
    # ================================
    image_path = 'datatestbienso/437.jpg'   # 👉 đổi đường dẫn ảnh tùy bạn
    frame = cv2.imread(image_path)

    if frame is None:
        print("❌ Không đọc được ảnh đầu vào!")
        return

    # ================================
    # 3️⃣ Đường dẫn mô hình + nhãn
    # ================================
    MODEL_PATH = "models/char_cnn_model.h5"
    class_labels = ['0','1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H',
                    'K','L','M','N','P','R','T',
                    'U','V','X','Y']

    # ================================
    # 4️⃣ Load mô hình CNN
    # ================================
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy mô hình: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print("✅ Mô hình đã tải thành công!")
    bienso, check = timbienso(frame, plate_cascade)

    if not check:
        print("⚠️ Không phát hiện được biển số trong ảnh.")
        cv2.imshow("Ảnh gốc", frame)
        cv2.waitKey(0)
        return

    kytu = tachkytu(bienso)
    plate_number = ""
    for i, char_img in enumerate(kytu):
        label = docbien(model, class_labels, char_img)
        plate_number += label
        print(f"Ký tự {i + 1}: {label}")

    print(" Biển số nhận dạng:", plate_number)
    cv2.putText(frame, plate_number, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("Nhan dien bien so", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
