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
    img_resized = cv2.resize(img, (32, 32))

    # 3. Invert để giống lúc bạn train (trắng = nét)
    img_resized = cv2.bitwise_not(img_resized)

    # 4. chuan hoa chuyen qua anh den trang
    img_input = img_resized.astype("float32") / 255.0

    # 5. dua anh dau vao cho đúng shape (1, 32, 32, 1)
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
    # Khởi tạo HaarCascade cho biển số xe
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')
    # VIDEO_SOURCE = 0  # Cam may tinh
    VIDEO_SOURCE = "test/video/VideoTest.mp4"  # Đường dẫn video
    MODEL_PATH = "models/char_cnn_model.h5"

    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                    'K', 'L', 'M', 'N', 'P', 'R', 'T',
                    'U', 'V', 'X', 'Y']

    # Tải mô hình CNN
    try:
        model = load_model(MODEL_PATH)
        print("Đã tải mô hình CNN!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Khởi tạo đối tượng đọc video
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Không thể mở nguồn video: {VIDEO_SOURCE}")
        return

    print("Bắt đầu xử lý video...")

    # Bắt đầu vòng lặp xử lý từng khung hình
    while True:
        # Đọc từng khung hình
        ret, frame = cap.read()

        # Kiểm tra xem khung hình có được đọc thành công không
        if not ret:
            print("Đã hết video hoặc không thể đọc khung hình.")
            break

        # Sao chép khung hình để hiển thị kết quả
        display_frame = frame.copy()

        # Phát hiện và cắt biển số
        bienso, check = timbienso(display_frame, plate_cascade)
        plate_number = ""

        if check and bienso is not None:
                # Tách ký tự (giả định hàm tachkytu đã được định nghĩa đúng)
                kytu = tachkytu(bienso)

                # Nhận diện từng ký tự
                for i, char_img in enumerate(kytu):
                    label = docbien(model, char_img, class_labels)
                    plate_number += label

                # In và hiển thị kết quả nhận diện
                print("Biển số nhận được:", plate_number)

                cv2.putText(display_frame, plate_number, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Khong phat hien bien so", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị khung hình
        cv2.imshow("Nhan dien bien so tu Video", display_frame)

        # Thoát vòng lặp khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng đối tượng đọc video và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

