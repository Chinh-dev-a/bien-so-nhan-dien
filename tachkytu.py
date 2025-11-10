import cv2
import numpy as np
import os

IOU_THRESHOLD = 0.5

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0
    iou = interArea / unionArea
    return iou

def adjust_gamma(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def tachkytu(img):
    img = cv2.resize(img, (400, 250), interpolation=cv2.INTER_AREA)#resize anh
    img=adjust_gamma(img,1)
    img_result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#chuyen anh sang xam
    gray_denoised = cv2.bilateralFilter(gray, 7, 75, 75)
    _, thresh_before_skew = cv2.threshold(gray_denoised, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh_before_skew = cv2.erode(thresh_before_skew, kernel, iterations=1)
    cv2.imshow('anh xu ly tach ky tu ni phan',thresh_before_skew)
    contours, _ = cv2.findContours(thresh_before_skew, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_chars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 50 and w > 10 and h < 200 and w < 100 and w / h < 1.5:
            potential_chars.append((x, y, w, h))

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

    final_chars.sort(key=lambda b: b[1])
    H_plate = img_result.shape[0]
    Y_mid_point = H_plate / 2

    if len(final_chars) > 5 and final_chars[0][1] < Y_mid_point < final_chars[-1][1]:
        top_row = []
        bottom_row = []
        for box in final_chars:
            if box[1] < Y_mid_point:
                top_row.append(box)
            else:
                bottom_row.append(box)
        top_row = sorted(top_row, key=lambda x: x[0])
        bottom_row = sorted(bottom_row, key=lambda x: x[0])
        sorted_chars = top_row + bottom_row
        print(f"Đã chia thành Hàng Trên ({len(top_row)} ký tự) và Hàng Dưới ({len(bottom_row)} ký tự).")
    else:
        sorted_chars = sorted(final_chars, key=lambda x: x[0])
        print("Xử lý như biển số 1 hàng (sắp xếp Trái -> Phải).")

    if not os.path.exists("kytucut"):
        os.mkdir("kytucut")

    for i, (x, y, w, h) in enumerate(sorted_chars):
        x-=10
        y-=10
        h+=15
        w+=15
        char_img = thresh_before_skew[y:y + h, x:x + w]
        char_img_resized = cv2.resize(char_img, (112, 112), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"kytucut/char_{i + 1}.jpg", char_img_resized)
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_result, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('anh xu ly tach ky tu ni phan', thresh_before_skew)
    cv2.imshow('anh cat ky tu bien so',img_result)
    print(f"✅ Đã tách và sắp xếp {len(sorted_chars)} ký tự.")
