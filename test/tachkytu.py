import cv2
import numpy as np
import os

# === 1. Đọc ảnh biển số ===
img = cv2.imread("plate_crop.jpg")   # ảnh biển số đã cắt
img=cv2.resize(img,(300,175),interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Tiền xử lý
# Làm mượt và tăng tương phản
gray = cv2.bilateralFilter(gray, 9, 75, 75)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('anh nhi phan ',thresh)

# === 3. Tìm contour ký tự ===
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)

# === 4. Tạo thư mục lưu ===
if not os.path.exists("kytucut"):
    os.mkdir("kytucut")

chars = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Lọc kích thước nhiễu
    if h > 40 and w > 10 and h < 300 and w < 200:
        chars.append((x, y, w, h))
        print(chars.append((x,y,w,h)))

# === 5. Sắp xếp ký tự từ trái sang phải ===
chars = sorted(chars, key=lambda x: x[0])

# === 6. Cắt và lưu từng ký tự ===
for i, (x, y, w, h) in enumerate(chars):
    char_img = gray[y:y+h, x:x+w]
    char_img = cv2.resize(char_img, (112, 112))
    cv2.imwrite(f"kytucut/char_{i+1}.jpg", char_img)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(f"✅ Đã tách {len(chars)} ký tự và lưu trong thư mục 'kytucut/'")

# === 7. Hiển thị kết quả ===
cv2.imshow("Ky tu tach duoc", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
