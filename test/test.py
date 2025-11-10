# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
#
# from tachchar import tachbien
#
# # Đọc ảnh
# # def docbienso(path_img):
# #     img = cv2.imread(path_img)
# #     # 🔹 Thay bằng đường dẫn ảnh biển số
# #
# #     # Chuyển sang ảnh xám
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# #     # Tăng độ tương phản và nhị phân hóa (Adaptive Threshold)
# #     thresh = cv2.adaptiveThreshold(
# #         gray, 255,
# #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #         cv2.THRESH_BINARY,
# #         11, 2
# #     )
# #
# #     # Tìm contour
# #     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     largest_rectangle = [0, 0]  # Lưu contour có diện tích lớn nhất
# #
# #     for cnt in contours:
# #         # Xấp xỉ đa giác (giảm nhiễu)
# #         approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
# #
# #         # Kiểm tra nếu contour là hình có 4 cạnh
# #         if len(approx) == 4:
# #             area = cv2.contourArea(cnt)
# #             if area > largest_rectangle[0]:
# #                 largest_rectangle = [area, cnt, approx]
# #
# #     # Nếu tìm thấy hình chữ nhật lớn nhất
# #     if largest_rectangle[0] != 0:
# #         # Lấy tọa độ hình chữ nhật
# #         x, y, w, h = cv2.boundingRect(largest_rectangle[1])
# #
# #
# #         # Cắt vùng biển số
# #         roi = img[y:y + h, x:x + w]
# #
# #         # Vẽ khung lên ảnh gốc
# #         cv2.drawContours(img, [largest_rectangle[1]], -1, (0, 255, 0), 1)
# #
# #         # Hiển thị kết quả
# #         cv2.imshow('Vung bien so', roi)
# #         cv2.imshow('Anh goc + khung', img)
# #
# #         # Nếu muốn lưu vùng cắt
# #         cv2.imwrite('plate_crop.jpg', roi)
# #         print("💾 Đã lưu vùng biển số: plate_crop.jpg")
# #         # tachbien('plates_crop.jpg')
# #
# #     else:
# #         print("⚠️ Không tìm thấy hình chữ nhật phù hợp!")
# #
# #
# #
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
from tabnanny import check

import cv2
import numpy as np
import os

def adjust_gamma(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# === Đường dẫn ảnh ===
#     image_path = "images/xesang2.jpg"  # 🔹 Thay bằng đường dẫn ảnh của bạn
def timbienso(image,plate_cascade):
    # === Nạp bộ cascade phát hiện biển số ===
    # plate_cascade = cv2.CascadeClassifier('cascade2.xml')

    # === Đọc ảnh ===
    # image = cv2.imread(image_path)
    # if image is None:
    #     print("⚠️ Không đọc được ảnh. Kiểm tra lại đường dẫn!")
    #     exit()

    # === Chuyển sang grayscale và tăng tương phản bằng gamma ===
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = adjust_gamma(gray, gamma=0.5)

    # === Phát hiện biển số ===
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
    )

    # === Thư mục lưu biển số ===
    # save_dir = "plates"
    # os.makedirs(save_dir, exist_ok=True)
    plate_count = 0

    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "BIEN SO XE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        plate_crop = image[y:y+h, x:x+w]
        # if plate_crop.size > 0:
        #     plate_filename = os.path.join(save_dir, f"plate_{plate_count}.jpg")
        #     cv2.imwrite(plate_filename, plate_crop)
        #     plate_count += 1
        #     print(f"💾 Đã lưu {plate_filename}")
    # checks=False
    if len(plates) == 0:
        print("❌ Không phát hiện được biển số nào trong ảnh.")
        checks=False
        plate_crop=0;
    else:
        print(f"✅ Phát hiện  biển số trong ảnh.")  # {len(plates)}
        checks = True

    # === Hiển thị kết quả ===
    # cv2.imshow("Phat hien bien so xe", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return plate_crop,checks
