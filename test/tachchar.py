# # import cv2
# # import numpy as np
# # import os
# #
# # # Ngưỡng chồng lấn (IOU) và Ngưỡng khoảng cách
# # IOU_THRESHOLD = 1
# #
# # def get_iou(boxA, boxB):
# #     # boxA, boxB là (x, y, w, h)
# #     xA = max(boxA[0], boxB[0])
# #     yA = max(boxA[1], boxB[1])
# #     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
# #     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
# #     interArea = max(0, xB - xA) * max(0, yB - yA)
# #     boxAArea = boxA[2] * boxA[3]
# #     boxBArea = boxB[2] * boxB[3]
# #     iou = interArea / float(boxAArea + boxBArea - interArea)
# #     return iou
# #
# #
# # def adjust_gamma(image, gamma=0.8):
# #     # Xây dựng bảng tra cứu (LookUp Table - LUT)
# #     invGamma = 1.0 / gamma
# #     table = np.array([((i / 255.0) ** invGamma) * 255
# #                       for i in np.arange(0, 256)]).astype("uint8")
# #
# #     # Áp dụng LUT cho ảnh
# #     return cv2.LUT(image, table)
# #
# #
# #
# # def tachbien(img_path):
# # # 1. Đọc ảnh biển số
# # #     img_path = "plates/plate_172.jpg"
# #     img = cv2.imread(img_path)
# #     # if img is None:
# #     #     print(f"Lỗi: Không tìm thấy ảnh tại đường dẫn {img_path}")
# #     #     exit()
# #     # kernel = np.ones((3,3),np.uint8)
# #     img = cv2.resize(img, (400, 250), interpolation=cv2.INTER_AREA)
# #
# #     # (h, w, d) = img.shape
# #     # # tính tâm ảnh
# #     # center = (w // 2, h // 2)
# #     # quay ảnh 45 độ tỉ lệ 0.5
# #     # m = cv2.getRotationMatrix2D(center, -4,1)
# #     # thực hiện lệnh quay
# #     # img = cv2.warpAffine(img, m, (w, h))
# #
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = adjust_gamma(gray, gamma=0.4)
# #     cv2.imshow('anh xam', gray)
# #     img_result = img.copy()
# #     # 2. Tiền xử lý
# #     gray = cv2.bilateralFilter(gray, 7, 75, 75)
# #
# #     cv2.imshow('anh da gian nhieu va giam choi', gray)
# #     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
# #     assert img is not None, "file could not be read, check with os.path.exists()"
# #     kernel = np.ones((3,3),np.uint8)
# #     thresh = cv2.erode(thresh,kernel,iterations = 1)
# #     # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# #     cv2.imshow('Ảnh nhị phân', thresh)
# #     # thresh = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)
# #
# #     # 3. Tìm contour ký tự
# #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     # 4. Lọc kích thước cơ bản
# #     potential_chars = []
# #     for c in contours:
# #         x, y, w, h = cv2.boundingRect(c)
# #         # Lọc kích thước (chỉnh lại ngưỡng phù hợp với ảnh 200x150)
# #         if h > 40 and w > 10 and h < 140 and w < 100 and w / h < 1.5:
# #             potential_chars.append((x, y, w, h))
# #
# #     # 5. Loại bỏ các hộp giới hạn bị trùng lặp
# #     final_chars = []
# #     potential_chars.sort(key=lambda b: b[2] * b[3], reverse=True)
# #
# #     for box1 in potential_chars:
# #         is_duplicate = False
# #         for box2 in final_chars:
# #             if get_iou(box1, box2) > IOU_THRESHOLD:
# #                 is_duplicate = True
# #                 break
# #         if not is_duplicate:
# #             final_chars.append(box1)
# #
# #     # =======================================================
# #     # === BƯỚC MỚI: Sắp xếp theo thứ tự (Trên-Dưới, Trái-Phải) ===
# #     # =======================================================
# #
# #     # 6.1. Tìm ranh giới hàng (Boundary)
# #     # Sắp xếp các ký tự theo tọa độ Y
# #     final_chars.sort(key=lambda b: b[1])
# #
# #     # Giả sử ranh giới nằm ở giữa chiều cao của ảnh
# #     # (Đây là cách đơn giản nhất, thường hiệu quả với biển số đã cắt chuẩn)
# #     H_plate = img.shape[0]  # Chiều cao ảnh 150
# #     Y_mid_point = H_plate / 2
# #
# #     # Nếu biển số chỉ có 1 hàng (thường là xe ô tô) thì không cần chia
# #     if len(final_chars) > 5 and final_chars[0][1] < Y_mid_point < final_chars[-1][1]:
# #
# #         # Phân loại ký tự vào hai hàng
# #         top_row = []
# #         bottom_row = []
# #
# #         for box in final_chars:
# #             # Dựa vào tọa độ Y của đỉnh trên (box[1])
# #             if box[1] < Y_mid_point:
# #                 top_row.append(box)
# #             else:
# #                 bottom_row.append(box)
# #
# #         # 6.2. Sắp xếp lại: Sắp xếp từng hàng theo tọa độ X
# #         top_row = sorted(top_row, key=lambda x: x[0])
# #         bottom_row = sorted(bottom_row, key=lambda x: x[0])
# #
# #         # Kết hợp: Hàng trên (Trái -> Phải) + Hàng dưới (Trái -> Phải)
# #         sorted_chars = top_row + bottom_row
# #
# #         print(f"Đã chia thành Hàng Trên ({len(top_row)} ký tự) và Hàng Dưới ({len(bottom_row)} ký tự).")
# #
# #     else:
# #         # Nếu không phải biển số 2 hàng hoặc có quá ít ký tự, sắp xếp tất cả theo X
# #         sorted_chars = sorted(final_chars, key=lambda x: x[0])
# #         print("Xử lý như biển số 1 hàng (sắp xếp Trái -> Phải).")
# #
# #     # 7. Cắt và lưu từng ký tự theo thứ tự đã sắp xếp
# #     if not os.path.exists("kytucut"):
# #         os.mkdir("kytucut")
# #
# #     for i, (x, y, w, h) in enumerate(sorted_chars):
# #         char_img = gray[y:y + h, x:x + w]
# #         char_img_resized = cv2.resize(char_img, (112, 112), interpolation=cv2.INTER_AREA)
# #
# #         # Lưu ảnh với tên file theo thứ tự i+1
# #         cv2.imwrite(f"kytucut/char_{i + 1}.jpg", char_img_resized)
# #
# #         # Vẽ hộp giới hạn và đánh số thứ tự
# #         cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #         # Thêm số thứ tự lên ảnh
# #         cv2.putText(img_result, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# #
# #     # print(f"✅ Đã tách và sắp xếp {len(sorted_chars)} ký tự theo thứ tự (Trên-Xuống, Trái-Phải).")
# #
# #     # 8. Hiển thị kết quả
# #     cv2.imshow("Ky tu tach duoc (Đã sắp xếp)", img_result)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #
# # def main():
# #     tachbien('plate_crop.jpg')
# #
# # if __name__ == "__main__":
# #     main()
# #     # return img_result
#
# import cv2
# import numpy as np
# import os
#
# # Ngưỡng chồng lấn (IOU)
# IOU_THRESHOLD = 0.5
#
#
# def get_iou(boxA, boxB):
#     # boxA, boxB là (x, y, w, h)
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = boxA[2] * boxA[3]
#     boxBArea = boxB[2] * boxB[3]
#     unionArea = float(boxAArea + boxBArea - interArea)
#     if unionArea == 0:
#         return 0
#     iou = interArea / unionArea
#     return iou
#
#
# def adjust_gamma(image, gamma=0.8):
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)
#
#
# # --- HÀM XOAY ẢNH CHÍNH NGHIÊNG (DESKEW) ĐÃ ĐIỀU CHỈNH ---
# # --- HÀM XOAY ẢNH CHÍNH NGHIÊNG (DESKEW) ĐIỀU CHỈNH THEO CÔNG THỨC MỚI ---
# # def deskew_image_by_corners(image):
# #     coords = cv2.findNonZero(image)
# #     if coords is None or len(coords) < 100:
# #         M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), 0, 1.0)
# #         return image, 0, M
# #
# #     rect = cv2.minAreaRect(coords)
# #     box = cv2.boxPoints(rect) # Lấy 4 đỉnh góc (float)
# #     box = np.int0(box)        # Chuyển về int
# #
# #     # 1. Tìm 2 đỉnh dưới cùng (có tọa độ Y lớn nhất)
# #     # Sắp xếp các đỉnh theo tọa độ Y giảm dần (lớn nhất ở đầu)
# #     sorted_y = box[np.argsort(box[:, 1])]
# #
# #     # Hai đỉnh dưới cùng là 2 đỉnh có Y lớn nhất
# #     # Ta lấy 2 đỉnh này ra và sắp xếp lại theo X để đảm bảo A luôn là đỉnh bên trái
# #     corner_A_B = sorted_y[-2:] # Lấy 2 đỉnh có Y lớn nhất
# #     corner_A_B = corner_A_B[np.argsort(corner_A_B[:, 0])] # Sắp xếp theo X
# #
# #     A = corner_A_B[0] # Đỉnh A (Bên trái, Y lớn)
# #     B = corner_A_B[1] # Đỉnh B (Bên phải, Y lớn)
# #
# #     x1, y1 = A[0], A[1]
# #     x2, y2 = B[0], B[1]
# #
# #     # 2. Tính toán cạnh đối và cạnh kề của tam giác ABC (với C là hình chiếu của A trên trục ngang qua B)
# #     # Cạnh đối (độ chênh lệch Y): y_diff = y1 - y2
# #     # Cạnh kề (độ chênh lệch X): x_diff = x2 - x1
# #
# #     y_diff = y1 - y2
# #     x_diff = x2 - x1
# #
# #     # 3. Tính góc quay (Góc alpha)
# #     # Sử dụng hàm atan2 để có được góc chính xác trong khoảng [-180, 180]
# #     # np.rad2deg chuyển từ radian sang độ
# #     angle = np.rad2deg(np.arctan2(y_diff, x_diff))
# #
# #     # 4. Xoay ảnh theo góc quay đã tính.
# #     # Nếu y1 > y2 (A cao hơn B) -> y_diff > 0 -> angle > 0 (xoay ngược chiều kim đồng hồ)
# #     # Nếu y1 < y2 (A thấp hơn B) -> y_diff < 0 -> angle < 0 (xoay thuận chiều kim đồng hồ)
# #     # Việc tính toán góc đã tự động bao gồm logic "ngược lại điểm A nằm cao hơn điểm B ta cho góc quay âm"
# #
# #     # Lấy tâm ảnh
# #     (h, w) = image.shape[:2]
# #     center = (w // 2, h // 2)
# #
# #     # Ma trận xoay (góc xoay là angle)
# #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
# #
# #     # Áp dụng phép biến đổi affine
# #     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# #
# #     # Trả về ảnh đã xoay, góc xoay, VÀ ma trận xoay M
# #     return rotated, angle, M
#
#
# # 1. Đọc ảnh biển số
# # img_path = "plates/plate_711.jpg"  # Thay bằng đường dẫn ảnh của bạn
# def  tachkytu(img):
#     # img_path='bienso/databienso/603_plate1.jpg'
#     # img = cv2.imread(img_path)
#     # if img is None:
#     #     print(f"Lỗi: Không tìm thấy ảnh tại đường dẫn {img_path}")
#     #     exit()
#
#     img = cv2.resize(img, (400, 250), interpolation=cv2.INTER_AREA)
#     img_result = img.copy()  # Dùng để vẽ kết quả
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('Ảnh gốc xám', gray)
#
#     # 2. Tiền xử lý
#     # 2.1 Giảm chói/Tăng tương phản bằng Gamma
#     # gray_gamma = adjust_gamma(gray, gamma=1)  # Gamma = 0.2 là khá mạnh, có thể thử 0.5-0.8
#     # cv2.imshow('Ảnh giảm chói', gray_gamma)
#
#     # 2.2 Giảm nhiễu
#     gray_denoised = cv2.bilateralFilter(gray , 7, 75, 75)
#     # cv2.imshow('Ảnh đã giảm nhiễu', gray_denoised)
#
#     # 2.3 Ngưỡng hóa nhị phân
#     _, thresh_before_skew = cv2.threshold(gray_denoised, 150, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow('Ảnh nhị phân (trước xoay)', thresh_before_skew)
#     kernel = np.ones((5, 5), np.uint8)
#     thresh_before_skew = cv2.erode(thresh_before_skew, kernel, iterations=1)
#
#     # 2.4 XOAY ẢNH CHỈNH NGHIÊNG (DESKEW)
#     # thresh_rotated, skew_angle, M = deskew_image_by_corners(thresh_before_skew)
#     # Áp dụng ma trận xoay M cho ảnh màu kết quả để vẽ contour
#     # (h, w) = img_result.shape[:2]
#     # img_result_rotated = cv2.warpAffine(img_result.copy(), M, (w, h), flags=cv2.INTER_CUBIC,
#     #                                     borderMode=cv2.BORDER_REPLICATE)
#
#     # cv2.imshow('Ảnh nhị phân (sau xoay)', thresh_before_skew)
#     # cv2.imshow('Ảnh gốc đã xoay thẳng', img_result_rotated)  # Hiển thị ảnh màu đã xoay
#     # print(f"Đã xoay ảnh một góc: {skew_angle:.2f} độ.")
#
#     # 3. Tìm contour ký tự
#     contours, _ = cv2.findContours(thresh_before_skew, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 4. Lọc kích thước cơ bản
#     potential_chars = []
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         # Ngưỡng lọc có thể cần điều chỉnh lại một chút sau khi xoay và resize
#         # Ví dụ, nếu ảnh 400x250, h > 50 và w > 10 có thể là tốt
#         # if h > 50 and w > 10 and h < 200 and w < 100 and w / h < 1.5:
#         #     potential_chars.append((x, y, w, h))
#
#     # 5. Loại bỏ các hộp giới hạn bị trùng lặp
#     final_chars = []
#     potential_chars.sort(key=lambda b: b[2] * b[3], reverse=True)
#
#     for box1 in potential_chars:
#         is_duplicate = False
#         for box2 in final_chars:
#             if get_iou(box1, box2) > IOU_THRESHOLD:
#                 is_duplicate = True
#                 break
#         if not is_duplicate:
#             final_chars.append(box1)
#
#     # 6. Sắp xếp theo thứ tự (Trên-Dưới, Trái-Phải)
#     final_chars.sort(key=lambda b: b[1])
#     H_plate = img_result.shape[0]
#     Y_mid_point = H_plate / 2
#
#     if len(final_chars) > 5 and final_chars[0][1] < Y_mid_point < final_chars[-1][1]:
#         top_row = []
#         bottom_row = []
#         for box in final_chars:
#             if box[1] < Y_mid_point:
#                 top_row.append(box)
#             else:
#                 bottom_row.append(box)
#
#         top_row = sorted(top_row, key=lambda x: x[0])
#         bottom_row = sorted(bottom_row, key=lambda x: x[0])
#         sorted_chars = top_row + bottom_row
#         print(f"Đã chia thành Hàng Trên ({len(top_row)} ký tự) và Hàng Dưới ({len(bottom_row)} ký tự).")
#     else:
#         sorted_chars = sorted(final_chars, key=lambda x: x[0])
#         print("Xử lý như biển số 1 hàng (sắp xếp Trái -> Phải).")
#
#     # 7. Cắt và lưu từng ký tự theo thứ tự đã sắp xếp
#     if not os.path.exists("kytucut"):
#         os.mkdir("kytucut")
#
#     for i, (x, y, w, h) in enumerate(sorted_chars):
#         char_img = thresh_before_skew[y:y + h, x:x + w]
#         char_img_resized = cv2.resize(char_img, (112, 112), interpolation=cv2.INTER_AREA)
#
#         cv2.imwrite(f"kytucut/char_{i + 1}.jpg", char_img_resized)
#
#         # Vẽ hộp giới hạn và đánh số thứ tự lên ảnh màu đã xoay
#         cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(img_result, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     print(f"✅ Đã tách và sắp xếp {len(sorted_chars)} ký tự.")
#
#     # 8. Hiển thị kết quả
#     # cv2.imshow("Ky tu tach duoc (Đã xoay & sắp xếp)", img_result)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import os
#
# from tensorflow.python.ops.gen_lookup_ops import anonymous_mutable_hash_table
#
# IOU_THRESHOLD = 0.5
#
# # ================================
# # Hàm tính IoU giữa 2 bounding box
# # ================================
# def get_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = boxA[2] * boxA[3]
#     boxBArea = boxB[2] * boxB[3]
#     unionArea = float(boxAArea + boxBArea - interArea)
#     if unionArea == 0:
#         return 0
#     iou = interArea / unionArea
#     return iou
#
#
# # ================================
# # Hàm chỉnh gamma (tăng/giảm sáng)
# # ================================
# def adjust_gamma(image, gamma=0.8):
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)
#
#
# # ================================
# # Hàm tách ký tự từ ảnh biển số
# # ================================
# def tachkytu(img):
#     img = cv2.resize(img, (400, 250), interpolation=cv2.INTER_AREA)
#     img = adjust_gamma(img, 1)
#     img_result = img.copy()
#
#     # 1️⃣ Chuyển ảnh sang xám
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 2️⃣ Giảm nhiễu
#     gray_denoised = cv2.bilateralFilter(gray, 7, 75, 75)
#
#     # 3️⃣ Phát hiện biên bằng Canny
#     edges = cv2.Canny(gray_denoised, 50, 150)
#     cv2.imshow("Canny Edge", edges)
#
#     # 4️⃣ Nhị phân đảo màu (nền đen - chữ trắng)
#     _, thresh = cv2.threshold(gray_denoised, 150, 255, cv2.THRESH_BINARY_INV)
#
#     # 5️⃣ Kết hợp biên và nhị phân
#     combined = cv2.bitwise_or(thresh, edges)
#     kernel = np.ones((3, 3), np.uint8)
#     combined = cv2.erode(combined, kernel, iterations=1)
#     cv2.imshow('Anh nhi phan ket hop Canny', combined)
#
#     # 6️⃣ Tìm contour
#     contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     potential_chars = []
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         if h > 50 and w > 10 and h < 200 and w < 100 and w / h < 1.5:
#             potential_chars.append((x, y, w, h))
#
#     # 7️⃣ Lọc trùng bằng IoU
#     final_chars = []
#     potential_chars.sort(key=lambda b: b[2] * b[3], reverse=True)
#     for box1 in potential_chars:
#         is_duplicate = False
#         for box2 in final_chars:
#             if get_iou(box1, box2) > IOU_THRESHOLD:
#                 is_duplicate = True
#                 break
#         if not is_duplicate:
#             final_chars.append(box1)
#
#     # 8️⃣ Sắp xếp ký tự theo hàng
#     final_chars.sort(key=lambda b: b[1])
#     H_plate = img_result.shape[0]
#     Y_mid_point = H_plate / 2
#
#     if len(final_chars) > 5 and final_chars[0][1] < Y_mid_point < final_chars[-1][1]:
#         top_row = []
#         bottom_row = []
#         for box in final_chars:
#             if box[1] < Y_mid_point:
#                 top_row.append(box)
#             else:
#                 bottom_row.append(box)
#         top_row = sorted(top_row, key=lambda x: x[0])
#         bottom_row = sorted(bottom_row, key=lambda x: x[0])
#         sorted_chars = top_row + bottom_row
#         print(f"Đã chia thành Hàng Trên ({len(top_row)} ký tự) và Hàng Dưới ({len(bottom_row)} ký tự).")
#     else:
#         sorted_chars = sorted(final_chars, key=lambda x: x[0])
#         print("Xử lý như biển số 1 hàng (sắp xếp Trái -> Phải).")
#
#     # 9️⃣ Lưu & trả về ảnh ký tự
#     if not os.path.exists("kytucut"):
#         os.mkdir("kytucut")
#
#     char_images = []
#     for i, (x, y, w, h) in enumerate(sorted_chars):
#         x = max(0, x - 10)
#         y = max(0, y - 10)
#         w = min(img.shape[1] - x, w + 15)
#         h = min(img.shape[0] - y, h + 15)
#
#         char_img = combined[y:y + h, x:x + w]
#         char_img_resized = cv2.resize(char_img, (112, 112), interpolation=cv2.INTER_AREA)
#         char_images.append(char_img_resized)
#
#         cv2.imwrite(f"kytucut/char_{i + 1}.jpg", char_img_resized)
#         cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(img_result, str(i + 1), (x, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     cv2.imshow('Anh cat ky tu bien so', img_result)
#     print(f"✅ Đã tách và sắp xếp {len(sorted_chars)} ký tự.")
#
#     return char_images
#
# def main():
#     image_path = 'bienso/databienso/217_plate1.jpg'
#     img=tachkytu(image_path)
#
# if __name__=='__main__':
#     main()


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
    return interArea / unionArea

def adjust_gamma(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def tachkytu(img):
    img = cv2.resize(img, (400, 250), interpolation=cv2.INTER_AREA)
    img = adjust_gamma(img, 0.6)
    img_result = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.bilateralFilter(gray, 7, 75, 75)
    _, thresh = cv2.threshold(gray_denoised, 150, 255, cv2.THRESH_BINARY_INV)

    # Tăng biên bằng Canny để dễ tách ký tự nghiêng
    edges = cv2.Canny(thresh, 100, 200)
    cv2.imshow("Edges Canny", edges)

    # Kết hợp edges vào ảnh nhị phân
    combined = cv2.bitwise_or(thresh, edges)
    cv2.imshow("Binary + Canny", combined)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.erode(combined, kernel, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_chars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 50 < h < 200 and 10 < w < 100 and w / h < 1.5:
            potential_chars.append((x, y, w, h, c))

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

    # --- Vẽ bounding box ---
    for i, (x, y, w, h, cnt) in enumerate(final_chars):
        # (a) Bounding rectangle thường (xanh lá)
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # (b) Bounding rectangle xoay (đỏ)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_result, [box], 0, (0, 0, 255), 2)

        cv2.putText(img_result, f"{i+1}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('Bounding Rects (Green=Straight, Red=Rotated)', img_result)
    print(f"✅ Đã vẽ {len(final_chars)} ký tự với cả 2 loại bounding box.")
