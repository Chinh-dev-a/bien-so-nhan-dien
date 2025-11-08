import cv2
import numpy as np
import os

def sort_contours_left_to_right(contours):
    # trả về contours đã sắp xếp theo tâm x tăng dần
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours_with_boxes = zip(contours, bounding_boxes)
    sorted_items = sorted(contours_with_boxes, key=lambda b: b[1][0])
    sorted_contours = [item[0] for item in sorted_items]
    sorted_boxes = [item[1] for item in sorted_items]
    return sorted_contours, sorted_boxes

def segment_plate_characters(plate_img,
                             save_dir="chars",
                             debug=False,
                             min_area=100,         # lọc contour theo diện tích nhỏ nhất
                             min_w=8, min_h=20,    # lọc theo chiều rộng / cao tối thiểu
                             max_h_ratio=0.95):    # chiều cao tối đa so với ảnh biển
    """
    plate_img: BGR image (có thể là ảnh crop vùng biển số)
    save_dir: thư mục để lưu ký tự (nếu muốn)
    debug: nếu True, sẽ hiển thị vài bước trung gian
    Trả về: danh sách ảnh ký tự (grayscale, đã resize về 28x28)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Resize theo chiều ngang nếu quá lớn để xử lý nhanh (không bắt buộc)
    h0, w0 = plate_img.shape[:2]
    target_w = 400
    if w0 > target_w:
        scale = target_w / w0
        plate = cv2.resize(plate_img, (target_w, int(h0*scale)), interpolation=cv2.INTER_AREA)
    else:
        plate = plate_img.copy()

    # 2) Chuyển sang grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 3) Tăng tương phản với top-hat và black-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)

    # 4) Blur nhẹ và adaptive threshold / Otsu
    blur = cv2.GaussianBlur(enhanced, (3,3), 0)
    # Thử Otsu
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert nếu nền trắng ký tự đen -> muốn ký tự trắng nền đen để dễ xử lý
    # Kiểm tra tỉ lệ pixel trắng để quyết invert
    white_ratio = np.sum(th == 255) / (th.shape[0] * th.shape[1])
    if white_ratio > 0.5:
        th = cv2.bitwise_not(th)

    # 5) Morphology: đóng để kết nối phần rời rạc của ký tự, mở nhẹ để loại nhiễu
    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, rect_k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, rect_k, iterations=1)

    if debug:
        cv2.imshow("gray", gray)
        cv2.imshow("enhanced", enhanced)
        cv2.imshow("th", th)
        cv2.waitKey(0)

    # 6) Tìm contours (ngoại vi)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7) Lọc contour theo kích thước + tỉ lệ (loại bỏ nhiễu)
    plate_h, plate_w = th.shape
    chars = []
    saved = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Lọc theo diện tích/ kích thước hợp lý
        if area < min_area:
            continue
        if w < min_w or h < min_h:
            continue
        if h > plate_h * max_h_ratio:
            # Một số trường hợp phải cắt phần viền; nếu quá cao có thể là toàn bộ plate -> bỏ
            continue

        # Lọc theo tỉ lệ: ký tự thường cao hơn rộng (tùy biển thực tế có thể điều chỉnh)
        aspect = w / float(h)
        if aspect > 1.2:  # nếu quá rộng, có thể là 2 ký tự dính -> vẫn giữ nhưng có thể bẻ tiếp
            pass

        # Crop ký tự với padding nhỏ
        pad_x = max(1, int(w * 0.12))
        pad_y = max(1, int(h * 0.15))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(plate_w, x + w + pad_x)
        y2 = min(plate_h, y + h + pad_y)
        char_img = th[y1:y2, x1:x2]

        # Resize về kích thước chuẩn (ví dụ 28x28) giữ tỉ lệ bằng padding nếu cần
        size = 28
        hC, wC = char_img.shape
        # Tạo ảnh nền vuông
        if hC > wC:
            new_h = size
            new_w = int(wC * (size / hC))
        else:
            new_w = size
            new_h = int(hC * (size / wC))
        if new_w == 0: new_w = 1
        if new_h == 0: new_h = 1
        resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # đặt vào nền trắng đen (ký tự trắng nền đen)
        canvas = np.zeros((size, size), dtype=np.uint8)
        # center
        x_off = (size - new_w) // 2
        y_off = (size - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        chars.append((x, canvas))  # lưu kèm vị trí x để sắp xếp sau
        # không lưu file ở đây, lưu sau khi sort

    # 8) Sắp xếp ký tự theo x (từ trái sang phải)
    if len(chars) == 0:
        if debug:
            print("Không tìm thấy ký tự nào.")
        return []

    chars_sorted = sorted(chars, key=lambda item: item[0])
    char_images = []
    for i, (xpos, img) in enumerate(chars_sorted):
        # Lưu từng ký tự
        filename = os.path.join(save_dir, f"char_{i}.png")
        cv2.imwrite(filename, img)
        char_images.append(img)
        if debug:
            print(f"Saved {filename}")

    if debug:
        # show all chars
        for i, img in enumerate(char_images):
            cv2.imshow(f"char_{i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return char_images

if __name__ == "__main__":
    # Ví dụ sử dụng
    plate_image_path = "plates/plate_1071.jpg"  # thay bằng ảnh biển số đã crop của bạn
    if not os.path.exists(plate_image_path):
        print("Hãy đặt ảnh biển số đã crop vào:", plate_image_path)
        exit()

    img = cv2.imread(plate_image_path)
    chars = segment_plate_characters(img, save_dir="chars", debug=True)
    print(f"Đã tách được {len(chars)} ký tự. Lưu trong thư mục 'chars/'.")
