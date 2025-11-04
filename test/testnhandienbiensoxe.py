import cv2
import os

# Đường dẫn thư mục chứa ảnh gốc
input_folder = "./datatestbienso"   # 🔹 Thay bằng thư mục của bạn, ví dụ: "D:/images"

# Tạo thư mục lưu ảnh kết quả
output_folder = "./bienso/databienso"
os.makedirs(output_folder, exist_ok=True)

# Nạp bộ phân loại Haar Cascade cho biển số xe
plate_cascade = cv2.CascadeClassifier('cascade2.xml')
soloi=0
soluongkhongphathienduocanh=0

# Duyệt qua tất cả ảnh trong thư mục
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_folder, file_name)
        print(f"🔹 Đang xử lý: {img_path}")

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {file_name}")
            soluongkhongphathienduocanh +=1
            continue

        # Chuyển ảnh sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện vùng có biển số
        plates = plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25)
        )

        # Nếu không tìm thấy biển số
        if len(plates) == 0:
            print(f"❌ Không phát hiện biển số trong: {file_name}")
            soloi+=1
            continue

        # Cắt và lưu từng biển số phát hiện được
        for i, (x, y, w, h) in enumerate(plates):
            plate_crop = img[y:y + h, x:x + w]
            save_name = f"{os.path.splitext(file_name)[0]}_plate{i + 1}.jpg"
            save_path = os.path.join(output_folder, save_name)

            cv2.imwrite(save_path, plate_crop)
            print(f"✅ Lưu biển số: {save_path}")

print("\n🎯 Hoàn tất! Tất cả biển số đã được cắt và lưu trong thư mục:\n", output_folder)
print("so anh khong phat hien bien so la :",soloi)
print("so anh khong phat hien duoc anh la :",soluongkhongphathienduocanh)
