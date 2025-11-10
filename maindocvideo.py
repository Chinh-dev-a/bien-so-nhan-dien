import cv2
import os
from tensorflow.keras.models import load_model


from nhandienbienso import timbienso
from tachkytu import tachkytu
from filedocmodel import docbien

def main():
    plate_cascade = cv2.CascadeClassifier('cascade2.xml')
    cap = cv2.VideoCapture('test/video/xesang2.mp4')
    MODEL_PATH = "models/char_cnn_model.h5"
    folder = 'kytucut'
    class_labels = ['0','1','2','3','4','5','6','7','8','9',
                    'A','B','C','D','E','F','G','H',
                    'K','L','M','N','P','R','T',
                    'U','V','X','Y']

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy mô hình: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print("✅ Mô hình đã tải thành công!")

    if not cap.isOpened():
        print("❌ Không mở được video hoặc webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📹 Video kết thúc hoặc không đọc được khung hình.")
            break

        bienso, check = timbienso(frame, plate_cascade)
        if check is True:
            tachkytu(bienso)
            text = docbien(model, class_labels)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2, cv2.LINE_AA)

            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        cv2.imshow("Nhan dien bien so", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
