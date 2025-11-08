import cv2
import cv2 as cv
import numpy as np

# Đọc ảnh và xử lý nhị phân
img = cv.imread('biensoxe/plate_10.jpg')
img=cv2.resize(img,(250,150),interpolation=cv2.INTER_NEAREST)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 9, 75, 80)
cv2.imshow('anh da giam nhieu',gray)
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
th
cv2.imshow('anh nhi phan',thresh)

# Tìm contour
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

# # Tính chu vi
# pcontours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]

# Tính chu vi
peri = cv.arcLength(cnt, True)

# Duyệt thử epsilon từ nhỏ đến lớn để tìm khi có 4 điểm
approx = None
for i in range(1, 100):
    epsilon = i/100 * peri   # epsilon = 1%, 2%, ..., 99% chu vi
    approx_temp = cv.approxPolyDP(cnt, epsilon, True)
    if len(approx_temp) == 4:
        approx = approx_temp
        print(f"Tìm được epsilon = {epsilon:.3f}, có {len(approx)} đỉnh.")
        break

# Vẽ kết quả
if approx is not None:
    cv.drawContours(img, [approx], -1, (0, 255, 0), 3)
    for p in approx:
        cv.circle(img, tuple(p[0]), 5, (0, 0, 255), -1)
else:
    print("Không tìm thấy epsilon phù hợp (N != 4).")

cv.imshow('Approximation', img)
cv.waitKey(0)
cv.destroyAllWindows()
