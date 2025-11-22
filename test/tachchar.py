import cv2
import numpy as np
import math
import os


def preprocess_char(char_crop, new_w=112, new_h=112):
    """
    Căn giữa ký tự vào nền trắng vuông và resize về kích thước chuẩn
    """
    h, w = char_crop.shape
    size = max(w, h)
    square = np.ones((size, size), dtype=np.uint8) * 255

    # Căn giữa
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = char_crop

    # Resize về kích thước chuẩn
    char_final = cv2.resize(square, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return char_final


def tachkytu(img_path, save_dir="kytucut", debug=True):
    """
    Tách tất cả ký tự trong ảnh biển số và lưu từng ký tự riêng biệt.
    Đồng thời vẽ khung ký tự trên ảnh gốc.
    """
    # Đọc ảnh
    img_color = cv2.imread(img_path)
    # img_color =cv2.resize(img_color,(400,250),cv2.INTER_AREA)
    if img_color is None:
        print("❌ Không đọc được ảnh:", img_path)
        return

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(400,250),interpolation=cv2.INTERSECT_NONE)

    # Tiền xử lý - Canny
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 200)

    # Tìm contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_regions = []
    h_img, w_img = gray.shape

    # Lọc contour ký tự
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Bộ lọc loại bỏ nhiễu (tuỳ ảnh biển số)
        if h < 20 or w < 10 or h > 0.9 * h_img or w > 0.5 * w_img:
            continue
        if h / w < 1 or h / w > 5:
            continue

        char_regions.append((x, y, w, h))

    if not char_regions:
        print("Không tìm thấy ký tự nào.")
        return

    # Sắp xếp ký tự từ trái sang phải, trên xuống dưới
    char_regions = sorted(char_regions, key=lambda r: (r[1] // 10, r[0]))

    os.makedirs(save_dir, exist_ok=True)

    # Cắt, căn giữa và lưu
    for idx, (x, y, w, h) in enumerate(char_regions):
        char_crop = gray[y:y + h, x:x + w]
        _, char_bin = cv2.threshold(char_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        char_bin = 255 - char_bin  # đảo màu: ký tự trắng, nền đen

        char_final = preprocess_char(char_bin)
        save_path = os.path.join(save_dir, f"char_{idx + 1}.jpg")
        cv2.imwrite(save_path, char_final)

        # Vẽ khung ký tự lên ảnh gốc
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_color, str(idx + 1), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Hiển thị kết quả
    if debug:
        cv2.imshow("Edges (Canny)", edges)
        cv2.imshow("Bien so co khung ky tu", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"✅ Đã tách {len(char_regions)} ký tự và lưu vào thư mục '{save_dir}/'")


if __name__ == "__main__":
    tachkytu("bienso/databienso/705_plate1.jpg")
