import cv2
import os
from tensorflow.keras.models import load_model
from tachkytu import tachkytu
from filedocmodel import docbien

def timbienso(image, plate_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "BIEN SO XE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        plate_crop = image[y:y + h, x:x + w]

    if len(plates) == 0:
        print("Không phát hiện được biển số.")
        return None, False
    else:
        return plate_crop, True


def main():
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')
    IMAGE_PATH = 'test/datatestbienso/16.jpg'
    MODEL_PATH = "models/char_cnn_model.h5"

    class_labels = ['0','1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H',
                    'K','L','M','N','P','R','T',
                    'U','V','X','Y']

    model = load_model(MODEL_PATH)
    print("✔ Đã tải mô hình CNN!")

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Không đọc được ảnh!")
        return

    bienso, check = timbienso(frame, plate_cascade)

    if check:
        kytu = tachkytu(bienso)
        plate_number = ""
        for i, char_img in enumerate(kytu):
            label = docbien(model, char_img, class_labels)
            plate_number += label
            # print(f"Ký tự {i + 1}: {label}")

        print("Biển số nhận được:", plate_number)

        cv2.putText(frame, plate_number, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:
        print("Không phát hiện biển số.")

    cv2.imshow("Nhan dien bien so", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
