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
    #Phát hiện biển số trong khung hình (frame) bằng HaarCascade.
    # Chuyển đổi sang ảnh grayscale để phát hiện
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Khởi tạo mặc định
    plate_crop = None
    check = False

    # Chỉ xử lý biển số đầu tiên được tìm thấy (nếu có nhiều biển số)
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "BIEN SO XE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Cắt vùng chứa biển số
        plate_crop = image[y:y + h, x:x + w]
        check = True
        break  # Phát hiện và xử lý biển số đầu tiên, sau đó thoát khỏi vòng lặp

    return plate_crop, check


def main():
    # Tải bộ phân loại Haar Cascade để phát hiện biển số
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')
    IMAGE_PATH = 'test/datatestbienso/16.jpg'  # Đường dẫn đến ảnh kiểm thử
    MODEL_PATH = "models/char_cnn_model.h5"  # Đường dẫn đến mô hình CNN

    # Các nhãn ký tự được sử dụng trong quá trình huấn luyện mô hình
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                    'K', 'L', 'M', 'N', 'P', 'R', 'T',
                    'U', 'V', 'X', 'Y']

    # Tải mô hình nhận diện ký tự
    model = load_model(MODEL_PATH)
    print("Đã tải mô hình CNN!")

    # Đọc ảnh đầu vào
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Không đọc được ảnh!")
        return

    # 1. Phát hiện biển số trong ảnh
    bienso, check = timbienso(frame, plate_cascade)

    if check:
        # 2. Tách các ký tự từ ảnh biển số đã cắt
        kytu = tachkytu(bienso)
        plate_number = ""

        # 3. Vòng lặp nhận diện từng ký tự
        for i, char_img in enumerate(kytu):
            # Gọi hàm docbien để dự đoán ký tự
            label = docbien(model, char_img, class_labels)
            plate_number += label
            # print(f"Ký tự {i + 1}: {label}") # Tùy chọn: In từng ký tự

        # In kết quả cuối cùng
        print("Biển số nhận được:", plate_number)

        # Hiển thị biển số nhận diện lên ảnh
        cv2.putText(frame, plate_number, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:
        print("Không phát hiện biển số.")

    # Hiển thị ảnh kết quả và chờ người dùng nhấn phím bất kỳ
    cv2.imshow("Nhan dien bien so", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
