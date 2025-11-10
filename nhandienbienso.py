import cv2
import numpy as np
import os

# def adjust_gamma(image, gamma=0.8):
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)

def timbienso(image, plate_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('test',gray)
    # gray = adjust_gamma(gray, gamma=0.5)

    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )

    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "BIEN SO XE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        plate_crop = image[y:y + h, x:x + w]

    if len(plates) == 0:
        print("❌ Không phát hiện được biển số nào trong ảnh.")
        checks = False
        plate_crop = 0
    else:
        print("✅ Phát hiện biển số trong ảnh.")
        checks = True

    return plate_crop, checks
