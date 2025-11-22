import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tachkytu import tachkytu

def docbien(model, char_img, class_labels):
    # 1. Nếu ảnh đang là BGR → convert sang GRAY
    if len(char_img.shape) == 3:
        img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        img = char_img

    # 2. Resize theo đúng model bạn đang dùng (32×32)
    IMG_SIZE = (32, 32)
    img_resized = cv2.resize(img, IMG_SIZE)

    # 3. Invert để giống lúc bạn train (trắng = nét)
    img_resized = cv2.bitwise_not(img_resized)

    # 4. Normalized
    img_input = img_resized.astype("float32") / 255.0

    # 5. Expand cho đúng shape (1, 32, 32, 1)
    img_input = np.expand_dims(img_input, axis=(0, -1))

    # 6. Predict
    pred = model.predict(img_input, verbose=0)
    pred_idx = np.argmax(pred, axis=1)[0]

    # 7. Trả về nhãn ký tự
    return class_labels[pred_idx]


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
