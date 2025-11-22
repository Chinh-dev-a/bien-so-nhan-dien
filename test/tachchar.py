# import cv2
# import numpy as np
# import math
# import os
#
#
# def preprocess_char(char_crop, new_w=112, new_h=112):
#     """
#     Căn giữa ký tự vào nền trắng vuông và resize về kích thước chuẩn
#     """
#     h, w = char_crop.shape
#     size = max(w, h)
#     square = np.ones((size, size), dtype=np.uint8) * 255
#
#     # Căn giữa
#     x_offset = (size - w) // 2
#     y_offset = (size - h) // 2
#     square[y_offset:y_offset + h, x_offset:x_offset + w] = char_crop
#
#     # Resize về kích thước chuẩn
#     char_final = cv2.resize(square, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#     return char_final
#
#
# def tachkytu(img_path, save_dir="kytucut", debug=True):
#     """
#     Tách tất cả ký tự trong ảnh biển số và lưu từng ký tự riêng biệt.
#     Đồng thời vẽ khung ký tự trên ảnh gốc.
#     """
#     # Đọc ảnh
#     img_color = cv2.imread(img_path)
#     # img_color =cv2.resize(img_color,(400,250),cv2.INTER_AREA)
#     if img_color is None:
#         print("❌ Không đọc được ảnh:", img_path)
#         return
#
#     gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#     gray=cv2.resize(gray,(400,250),interpolation=cv2.INTERSECT_NONE)
#
#     # Tiền xử lý - Canny
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 80, 200)
#
#     # Tìm contour
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     char_regions = []
#     h_img, w_img = gray.shape
#
#     # Lọc contour ký tự
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#
#         # Bộ lọc loại bỏ nhiễu (tuỳ ảnh biển số)
#         if h < 20 or w < 10 or h > 0.9 * h_img or w > 0.5 * w_img:
#             continue
#         if h / w < 1 or h / w > 5:
#             continue
#
#         char_regions.append((x, y, w, h))
#
#     if not char_regions:
#         print("Không tìm thấy ký tự nào.")
#         return
#
#     # Sắp xếp ký tự từ trái sang phải, trên xuống dưới
#     char_regions = sorted(char_regions, key=lambda r: (r[1] // 10, r[0]))
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Cắt, căn giữa và lưu
#     for idx, (x, y, w, h) in enumerate(char_regions):
#         char_crop = gray[y:y + h, x:x + w]
#         _, char_bin = cv2.threshold(char_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         char_bin = 255 - char_bin  # đảo màu: ký tự trắng, nền đen
#
#         char_final = preprocess_char(char_bin)
#         save_path = os.path.join(save_dir, f"char_{idx + 1}.jpg")
#         cv2.imwrite(save_path, char_final)
#
#         # Vẽ khung ký tự lên ảnh gốc
#         cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(img_color, str(idx + 1), (x, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#     # Hiển thị kết quả
#     if debug:
#         cv2.imshow("Edges (Canny)", edges)
#         cv2.imshow("Bien so co khung ky tu", img_color)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     print(f"✅ Đã tách {len(char_regions)} ký tự và lưu vào thư mục '{save_dir}/'")
#
#
# if __name__ == "__main__":
#     tachkytu("bienso/databienso/705_plate1.jpg")
#####################################################################
import cv2
import numpy as np
import os

# def get_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = boxA[2] * boxA[3]
#     boxBArea = boxB[2] * boxB[3]
#     unionArea = float(boxAArea + boxBArea - interArea)
#     if unionArea == 0:
#         return 0
#     iou = interArea / unionArea
#     return iou

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
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
    # gray_denoised = cv2.bilateralFilter(gray, 7, 75, 75)
    _, thresh = cv2.threshold(morphology_img, 125, 255, cv2.THRESH_BINARY_INV)
    # Tạo mask (bắt buộc phải lớn hơn ảnh 2 pixel mỗi chiều)
    h, w = thresh.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill từ điểm góc (0,0): nền trắng -> thành đen (0)
    cv2.floodFill(thresh, mask, (0, 0), 0)
    # edges = cv2.Canny(morphology_img, 30, 200)
    # combined = cv2.bitwise_or(thresh, edges)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    # # chuyển hình ảnh từ ảnh RGB sang ảnh xám
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Giảm nhiễu bằng phương pháp lọc song phương
    # noise_removal = cv2.bilateralFilter(gray_img, 12, 30, 30)
    # # cân bằng histogram
    # equal_histogram = cv2.equalizeHist(noise_removal)
    # # tạo nhân 3*3
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # dùng thuật toán mở loại bỏ nhiễu với ảnh đầu vào là ảnh cân bằng, và nhân kích cở 3 * 3
    # morphology_img = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel)
    # # tìm biên ảnh bằng phương pháp Canny với đầu vào là ảnh đã qua xóa nhiễu
    # edged_img = cv2.Canny(morphology_img, 30, 200)
    # cv2.imshow('anh xu ly tach ky tu nhi phan',thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_chars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 50 and w > 10 and h < 200 and w < 100 and w / h < 1.5:
            potential_chars.append((x, y, w, h))

    # final_chars = []
    # potential_chars.sort(key=lambda b: b[2] * b[3], reverse=True)
    # for box1 in potential_chars:
    #     is_duplicate = False
    #     for box2 in final_chars:
    #         if get_iou(box1, box2) > IOU_THRESHOLD:
    #             is_duplicate = True
    #             break
    #     if not is_duplicate:
    #         final_chars.append(box1)
    #########################################
#sap xep tam on
    # potential_chars.sort(key=lambda b: b[1])
    # H_plate = img_result.shape[0]
    # Y_mid_point = H_plate / 2
    #
    # if len(potential_chars) > 5 and potential_chars[0][1] < Y_mid_point < potential_chars[-1][1]:
    #     top_row = []
    #     bottom_row = []
    #     for box in potential_chars:
    #         if box[1] < Y_mid_point:
    #             top_row.append(box)
    #         else:
    #             bottom_row.append(box)
    #     top_row = sorted(top_row, key=lambda x: x[0])
    #     bottom_row = sorted(bottom_row, key=lambda x: x[0])
    #     sorted_chars = top_row + bottom_row
    #     print(f"Đã chia thành Hàng Trên ({len(top_row)} ký tự) và Hàng Dưới ({len(bottom_row)} ký tự).")
    # else:
    #     sorted_chars = sorted(potential_chars, key=lambda x: x[0])
    #     print("Xử lý như biển số 1 hàng (sắp xếp Trái -> Phải).")
    ###############################################################
    # potential_chars = [(x, y, w, h), ...]

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
        print(f"Biển số 2 hàng: Top={len(top_row)} ký tự, Bottom={len(bottom_row)} ký tự")
        sorted_chars = top_row_sorted + bottom_row_sorted
    else:
        print("Biển số 1 hàng — Sắp xếp theo tâm trái → phải.")
        all_chars = top_row + bottom_row
        sorted_chars = sorted(all_chars, key=lambda b: b[4])

    # (tuỳ chọn) bỏ cx, cy để giữ lại (x, y, w, h)
    sorted_chars = [(x, y, w, h) for (x, y, w, h, cx, cy) in sorted_chars]

    if not os.path.exists("kytucut"):
        os.mkdir("kytucut")

    for i, (x, y, w, h) in enumerate(sorted_chars):
        x-=10
        y-=10
        w+=15
        h+=15
        char_img = thresh[y:y + h, x:x + w]
        if char_img.size == 0 or w < 5 or h < 5:
            print(f" Bỏ qua ký tự rỗng tại vị trí ({x}, {y})")
            continue
        char_img_resized = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"kytucut/char_{i + 1}.jpg", char_img_resized)
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_result, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('anh xu ly tach ky tu ni phan', thresh)
    cv2.imshow('anh cat ky tu bien so',img_result)
    print(f"Đã tách và sắp xếp {len(sorted_chars)} ký tự.")