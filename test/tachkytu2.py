import cv2
import numpy as np
import os

# Ngưỡng chồng lấn (IOU) và Ngưỡng khoảng cách
IOU_THRESHOLD = 1

def get_iou(boxA, boxB):
    # boxA, boxB là (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def adjust_gamma(image, gamma=0.8):
    # Xây dựng bảng tra cứu (LookUp Table - LUT)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Áp dụng LUT cho ảnh
    return cv2.LUT(image, table)




# 1. Đọc ảnh biển số
img_path = "plates/plate_172.jpg"
img = cv2.imread(img_path)
# if img is None:
#     print(f"Lỗi: Không tìm thấy ảnh tại đường dẫn {img_path}")
#     exit()
# kernel = np.ones((3,3),np.uint8)
img = cv2.resize(img, (400, 250), interpolation=cv2.INTER_AREA)

# (h, w, d) = img.shape
# # tính tâm ảnh
# center = (w // 2, h // 2)
# quay ảnh 45 độ tỉ lệ 0.5
# m = cv2.getRotationMatrix2D(center, -4,1)
# thực hiện lệnh quay
# img = cv2.warpAffine(img, m, (w, h))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = adjust_gamma(gray, gamma=0.4)
cv2.imshow('anh xam', gray)
img_result = img.copy()
# 2. Tiền xử lý
gray = cv2.bilateralFilter(gray, 7, 75, 75)

cv2.imshow('anh da gian nhieu va giam choi', gray)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((3,3),np.uint8)
thresh = cv2.erode(thresh,kernel,iterations = 1)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Ảnh nhị phân', thresh)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)

# 3. Tìm contour ký tự
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Lọc kích thước cơ bản
potential_chars = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Lọc kích thước (chỉnh lại ngưỡng phù hợp với ảnh 200x150)
    if h > 40 and w > 10 and h < 140 and w < 100 and w / h < 1.5:
        potential_chars.append((x, y, w, h))

# 5. Loại bỏ các hộp giới hạn bị trùng lặp
final_chars = []
potential_chars.sort(key=lambda b: b[2] * b[3], reverse=True)

for box1 in potential_chars:
    is_duplicate = False
    for box2 in final_chars:
        if get_iou(box1, box2) > IOU_THRESHOLD:
            is_duplicate = True
            break
    if not is_duplicate:
        final_chars.append(box1)

# =======================================================
# === BƯỚC MỚI: Sắp xếp theo thứ tự (Trên-Dưới, Trái-Phải) ===
# =======================================================

# 6.1. Tìm ranh giới hàng (Boundary)
# Sắp xếp các ký tự theo tọa độ Y
final_chars.sort(key=lambda b: b[1])

# Giả sử ranh giới nằm ở giữa chiều cao của ảnh
# (Đây là cách đơn giản nhất, thường hiệu quả với biển số đã cắt chuẩn)
H_plate = img.shape[0]  # Chiều cao ảnh 150
Y_mid_point = H_plate / 2

# Nếu biển số chỉ có 1 hàng (thường là xe ô tô) thì không cần chia
if len(final_chars) > 5 and final_chars[0][1] < Y_mid_point < final_chars[-1][1]:

    # Phân loại ký tự vào hai hàng
    top_row = []
    bottom_row = []

    for box in final_chars:
        # Dựa vào tọa độ Y của đỉnh trên (box[1])
        if box[1] < Y_mid_point:
            top_row.append(box)
        else:
            bottom_row.append(box)

    # 6.2. Sắp xếp lại: Sắp xếp từng hàng theo tọa độ X
    top_row = sorted(top_row, key=lambda x: x[0])
    bottom_row = sorted(bottom_row, key=lambda x: x[0])

    # Kết hợp: Hàng trên (Trái -> Phải) + Hàng dưới (Trái -> Phải)
    sorted_chars = top_row + bottom_row

    print(f"Đã chia thành Hàng Trên ({len(top_row)} ký tự) và Hàng Dưới ({len(bottom_row)} ký tự).")

else:
    # Nếu không phải biển số 2 hàng hoặc có quá ít ký tự, sắp xếp tất cả theo X
    sorted_chars = sorted(final_chars, key=lambda x: x[0])
    print("Xử lý như biển số 1 hàng (sắp xếp Trái -> Phải).")

# 7. Cắt và lưu từng ký tự theo thứ tự đã sắp xếp
if not os.path.exists("kytucut"):
    os.mkdir("kytucut")

for i, (x, y, w, h) in enumerate(sorted_chars):
    char_img = gray[y:y + h, x:x + w]
    char_img_resized = cv2.resize(char_img, (112, 112), interpolation=cv2.INTER_AREA)

    # Lưu ảnh với tên file theo thứ tự i+1
    cv2.imwrite(f"kytucut/char_{i + 1}.jpg", char_img_resized)

    # Vẽ hộp giới hạn và đánh số thứ tự
    cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Thêm số thứ tự lên ảnh
    cv2.putText(img_result, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print(f"✅ Đã tách và sắp xếp {len(sorted_chars)} ký tự theo thứ tự (Trên-Xuống, Trái-Phải).")

# 8. Hiển thị kết quả
cv2.imshow("Ky tu tach duoc (Đã sắp xếp)", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()