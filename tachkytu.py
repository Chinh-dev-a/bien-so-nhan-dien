import cv2
import numpy as np
import os

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def tachkytu(img):
    img = cv2.resize(img, (400, 250), interpolation=cv2.INTER_AREA)
    img=adjust_gamma(img,0.8)
    img_result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#chuyen anh sang xam
    # Giảm nhiễu bằng phương pháp lọc song phương
    noise_removal = cv2.bilateralFilter(gray, 12, 30, 30)
    # cân bằng histogram
    equal_histogram = cv2.equalizeHist(noise_removal)
    # tạo nhân 3*3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dùng thuật toán mở loại bỏ nhiễu với ảnh đầu vào là ảnh cân bằng, và nhân kích cở 3 * 3
    morphology_img = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel)
    _, thresh = cv2.threshold(morphology_img, 125, 255, cv2.THRESH_BINARY_INV)
    # Tạo mask (bắt buộc phải lớn hơn ảnh 2 pixel mỗi chiều)
    h, w = thresh.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill từ điểm góc (0,0): nền trắng -> thành đen (0)
    cv2.floodFill(thresh, mask, (0, 0), 0)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_chars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 50 and w > 10 and h < 200 and w < 100 and w / h < 1.5:
            potential_chars.append((x, y, w, h))

    H_plate = img_result.shape[0]
    Y_mid_point = H_plate / 2

    top_row = []
    bottom_row = []

    # --- 1) Tính tâm và phân loại hàng ---
    for (x, y, w, h) in potential_chars:
        cx = x + w / 2
        cy = y + h / 2

        if cy < Y_mid_point:
            top_row.append((x, y, w, h, cx, cy))
        else:
            bottom_row.append((x, y, w, h, cx, cy))

    # --- 2) Sắp xếp trong từng hàng theo tọa độ tâm cx ---
    top_row_sorted = sorted(top_row, key=lambda b: b[4])
    bottom_row_sorted = sorted(bottom_row, key=lambda b: b[4])

    # --- 3) Gộp 2 hàng theo đúng thứ tự ---
    if len(top_row) > 0 and len(bottom_row) > 0:
        # print(f"Biển số 2 hàng: Top={len(top_row)} ký tự, Bottom={len(bottom_row)} ký tự")
        sorted_chars = top_row_sorted + bottom_row_sorted
    else:
        #Biển số 1 hàng — Sắp xếp theo tâm trái → phải.
        all_chars = top_row + bottom_row
        sorted_chars = sorted(all_chars, key=lambda b: b[4])

    # (tuỳ chọn) bỏ cx, cy để giữ lại (x, y, w, h)
    sorted_chars = [(x, y, w, h) for (x, y, w, h, cx, cy) in sorted_chars]

    if not os.path.exists("kytucut"):
        os.mkdir("kytucut")

    chars_list = []

    for i, (x, y, w, h) in enumerate(sorted_chars):
        x -= 10
        y -= 10
        w += 15
        h += 15

        char_img = thresh[y:y + h, x:x + w]

        if char_img.size == 0 or w < 5 or h < 5:
            print(f"Bỏ qua ký tự rỗng tại ({x},{y})")
            continue

        # Resize về 32x32
        char_img_resized = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_AREA)

        # Thêm vào list để return
        chars_list.append(char_img_resized)

        # (tùy chọn) vẽ khung xem trực quan
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_result, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('anh xu ly tach ky tu nhi phan', thresh)
    cv2.imshow('anh cat ky tu bien so', img_result)

    print(f"Đã tách và sắp xếp {len(chars_list)} ký tự.")

    return chars_list

if __name__ == "__main__":
    img=cv2.imread("test/bienso/databienso/16_plate1.jpg")
    chars = tachkytu(img)
    print(len(chars))
    x=0
    while x!=len(chars):
        name = f"Ky tu {x + 1}"
        cv2.imshow(name, chars[x])
        x+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
