# import cv2
# import numpy as np
# import os
#
# def adjust_gamma(image, gamma=0.8):
#     # Xây dựng bảng tra cứu (LookUp Table - LUT)
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#
#     # Áp dụng LUT cho ảnh
#     return cv2.LUT(image, table)
#
# # === Đường dẫn video hoặc webcam ===
# video_path = "video/xesang2.mp4"  # 🔹 Thay bằng đường dẫn video của bạn
# # Nếu muốn dùng webcam thì đổi thành video_path = 0
#
# # === Nạp bộ cascade phát hiện biển số ===
# plate_cascade = cv2.CascadeClassifier('cascade2.xml')  # 🔹 Đường dẫn đến file cascade
#
# # === Mở video hoặc camera ===
# cap = cv2.VideoCapture(video_path)
#
# if not cap.isOpened():
#     print("⚠️ Không mở được video hoặc camera.")
#     exit()
#
# # === Thư mục lưu biển số ===
# save_dir = "plates"
# os.makedirs(save_dir, exist_ok=True)
#
# plate_count = 0  # Đếm số lượng biển số đã lưu
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("📁 Hết video hoặc không đọc được khung hình.")
#         break
#
#     # Chuyển sang grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = adjust_gamma(gray, gamma=0.5)
#
#     # Phát hiện biển số
#     plates = plate_cascade.detectMultiScale(
#         gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
#     )
#
#     for (x, y, w, h) in plates:
#         # Vẽ khung quanh biển số
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, "BIEN SO XE", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#         # Cắt biển số và lưu lại
#         plate_crop = frame[y:y+h, x:x+w]
#         if plate_crop.size > 0:  # tránh lỗi ảnh rỗng
#             plate_filename = os.path.join(save_dir, f"plate_{plate_count}.jpg")
#             cv2.imwrite(plate_filename, plate_crop)
#             plate_count += 1
#             print(f"💾 Đã lưu {plate_filename}")
#
#     # Hiển thị video
#     cv2.imshow("Phat hien bien so xe", frame)
#     # cv2.imshow("Anh xam", gray)  # nếu muốn xem ảnh grayscale
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Giải phóng tài nguyên
# cap.release()
# cv2.destroyAllWindows()
############################################################################
import cv2
import os
from tensorflow.keras.models import load_model

from tachchar import tachkytu
from testread import docbien

def timbienso(image, plate_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('test',gray)
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "BIEN SO XE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        plate_crop = image[y:y + h, x:x + w]

    if len(plates) == 0:
        print(" Không phát hiện được biển số nào trong ảnh.")
        checks = False
        plate_crop = 0
    else:
        print(" Phát hiện biển số trong ảnh.")
        checks = True

    return plate_crop, checks

def main():
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')
    cap = cv2.VideoCapture('video/VideoTest.mp4')
    MODEL_PATH = "models/char_cnn_model.h5"
    folder = 'kytucut'
    class_labels = ['0','1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H',
                    'K','L','M','N','P','R','S','T',
                    'U','V','X','Y']

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f" Không tìm thấy mô hình: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print(" Mô hình đã tải thành công!")

    if not cap.isOpened():
        print(" Không mở được video hoặc webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video kết thúc hoặc không đọc được khung hình.")
            break

        bienso, check = timbienso(frame, plate_cascade)
        if check is True:
            tachkytu(bienso)
            text = docbien(model, class_labels)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2, cv2.LINE_AA)

            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        cv2.imshow("Nhan dien bien so", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
