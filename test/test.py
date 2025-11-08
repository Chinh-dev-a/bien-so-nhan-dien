import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Đọc ảnh
img = cv2.imread('datatestbienso/1042.jpg')
# 🔹 Thay bằng đường dẫn ảnh biển số

# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tăng độ tương phản và nhị phân hóa (Adaptive Threshold)
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# Tìm contour
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

largest_rectangle = [0, 0]  # Lưu contour có diện tích lớn nhất

for cnt in contours:
    # Xấp xỉ đa giác (giảm nhiễu)
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

    # Kiểm tra nếu contour là hình có 4 cạnh
    if len(approx) == 4:
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [area, cnt, approx]

# Nếu tìm thấy hình chữ nhật lớn nhất
if largest_rectangle[0] != 0:
    # Lấy tọa độ hình chữ nhật
    x, y, w, h = cv2.boundingRect(largest_rectangle[1])

    # Cắt vùng biển số
    roi = img[y:y + h, x:x + w]

    # Vẽ khung lên ảnh gốc
    cv2.drawContours(img, [largest_rectangle[1]], -1, (0, 255, 0), 1)

    # Hiển thị kết quả
    cv2.imshow('Vung bien so', roi)
    cv2.imshow('Anh goc + khung', img)

    # Nếu muốn lưu vùng cắt
    cv2.imwrite('plate_crop.jpg', roi)
    print("💾 Đã lưu vùng biển số: plate_crop.jpg")

else:
    print("⚠️ Không tìm thấy hình chữ nhật phù hợp!")

cv2.waitKey(0)
cv2.destroyAllWindows()
