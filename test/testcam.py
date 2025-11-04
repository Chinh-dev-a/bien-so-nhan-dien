import cv2

# === Đường dẫn video ===
video_path = "video/xetoi.mp4"   # 🔹 Thay bằng đường dẫn video của bạn
# Nếu muốn dùng webcam, đặt video_path = 0

# === Nạp bộ cascade phát hiện biển số ===
plate_cascade = cv2.CascadeClassifier('cascade2.xml')  # 🔹 Đường dẫn file cascade

# === Mở video hoặc camera ===
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("⚠️ Không mở được video hoặc camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("📁 Hết video hoặc không đọc được khung hình.")
        break

    # Chuyển sang grayscale để xử lý
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện biển số trong từng khung hình
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
    )

    # Vẽ khung quanh vùng phát hiện
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "BIEN SO XE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị video
    cv2.imshow("Phat hien bien so xe", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
