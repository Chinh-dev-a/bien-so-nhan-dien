import cv2
import os
from tensorflow.keras.models import load_model
from nhandienbienso import timbienso
from tachkytu import tachkytu
from filedocmodel import docbien

def main():
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')
    IMAGE_PATH = 'test/datatestbienso/16.jpg'
    MODEL_PATH = "testbiensoxe/models/char_cnn_model.h5"
    folder = 'kytucut'
    class_labels = ['0','1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H',
                    'K','L','M','N','P','R','T',
                    'U','V','X','Y']

    # Kiểm tra model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f" Không tìm thấy mô hình: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Mô hình đã tải thành công!")

    # Đọc ảnh
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Không đọc được ảnh! Kiểm tra lại đường dẫn.")
        return

    # Nhận diện biển số
    bienso, check = timbienso(frame, plate_cascade)
    if check is True:
        # Tách ký tự
        tachkytu(bienso)

        # Đọc ký tự và ghép chuỗi
        text = docbien(model, class_labels)

        # Hiển thị kết quả
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)
        print(f"Biển số nhận được: {text}")

        # Dọn thư mục ký tự
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        print("Không phát hiện được biển số trong ảnh.")

    # Hiển thị ảnh kết quả
    cv2.imshow("Nhan dien bien so", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
