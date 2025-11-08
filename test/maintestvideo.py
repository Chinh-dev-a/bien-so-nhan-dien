import cv2
import os
from tensorflow.keras.models import load_model

from test import timbienso        # Hàm phát hiện vùng biển số (trả về ảnh biển số + check)
from tachchar import tachkytu     # Hàm tách ký tự
from testread import docbien      # Hàm đọc ký tự và ghép chuỗi biển số


def main():
    # ================================
    # 1️⃣ Nạp cascade phát hiện biển số
    # ================================
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')

    # ================================
    # 2️⃣ Chọn nguồn video
    # ================================
    # cap = cv2.VideoCapture(0)  # webcam
    cap = cv2.VideoCapture("video/cv2 (1).mp4")

    MODEL_PATH = "models/char_cnn_model.h5"   # Mô hình CNN
    folder = 'kytucut'                        # Thư mục tạm chứa ký tự tách

    # ================================
    # 3️⃣ Danh sách nhãn (label)
    # ================================
    class_labels = ['0','1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H',
                    'K','L','M','N','P','R','T',
                    'U','V','X','Y']

    # ================================
    # 4️⃣ Load mô hình
    # ================================
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy mô hình: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print("✅ Mô hình đã tải thành công!")

    # ================================
    # 5️⃣ Kiểm tra video
    # ================================
    if not cap.isOpened():
        print("❌ Không mở được video hoặc webcam!")
        return

    # ================================
    # 6️⃣ Xử lý từng khung hình
    # ================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("📹 Video kết thúc hoặc không đọc được khung hình.")
            break

        # Phát hiện biển số
        bienso, check = timbienso(frame, plate_cascade)

        if check is True :
            # Tách ký tự
            tachkytu(bienso)

            # Nhận dạng ký tự
            text = docbien(model, class_labels)

            # Ghi kết quả lên ảnh
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # Xóa file tạm trong thư mục
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Hiển thị video
        cv2.imshow("Nhan dien bien so", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ================================
    # 7️⃣ Giải phóng tài nguyên
    # ================================
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
