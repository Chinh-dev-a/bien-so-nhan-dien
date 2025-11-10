# # # # import cv2
# # # # import numpy as np
# # # #
# # # # # Đọc ảnh
# # # # img = cv2.imread('datatestbienso/127.jpg')
# # # # if img is None:
# # # #     print("Không tìm thấy ảnh 'bss.jpg'")
# # # #     exit()
# # # #
# # # # # Chuyển sang ảnh xám
# # # # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # # #
# # # # # Làm mờ để giảm nhiễu
# # # # gray = cv2.GaussianBlur(gray, (5, 5), 0)
# # # #
# # # # # Ngưỡng hóa ảnh (adaptive threshold)
# # # # thresh = cv2.adaptiveThreshold(
# # # #     gray, 255,
# # # #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# # # #     cv2.THRESH_BINARY,
# # # #     11, 2
# # # # )
# # # #
# # # # # Tìm các đường viền (contours)
# # # # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # # #
# # # # # Biến lưu vùng chữ nhật lớn nhất
# # # # largest_rectangle = None
# # # # max_area = 0
# # # #
# # # # # Duyệt tất cả contour
# # # # for cnt in contours:
# # # #     # Xấp xỉ hình đa giác từ contour
# # # #     approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
# # # #
# # # #     # Kiểm tra nếu có 4 cạnh (hình chữ nhật hoặc gần chữ nhật)
# # # #     if len(approx) == 4:
# # # #         area = cv2.contourArea(cnt)
# # # #         if area > max_area:
# # # #             max_area = area
# # # #             largest_rectangle = approx
# # # #
# # # # # Nếu tìm thấy vùng chữ nhật lớn nhất
# # # # if largest_rectangle is not None:
# # # #     # Vẽ khung xanh quanh vùng chữ nhật
# # # #     cv2.drawContours(img, [largest_rectangle], -1, (0, 255, 0), 4)
# # # #
# # # #     # Cắt vùng chữ nhật ra khỏi ảnh
# # # #     x, y, w, h = cv2.boundingRect(largest_rectangle)
# # # #     cropped = img[y:y + h, x:x + w]
# # # #
# # # #     # Hiển thị ảnh kết quả
# # # #     cv2.imshow('Khung chữ nhật', img)
# # # #     cv2.imshow('Vùng cắt', cropped)
# # # # else:
# # # #     print("Không tìm thấy vùng hình chữ nhật nào.")
# # # #
# # # # cv2.waitKey(0)
# # # # cv2.destroyAllWindows()
# # #
# # # import cv2
# # # import numpy as np
# # #
# # # # ---- CẤU HÌNH NGUỒN VIDEO ----
# # # # Nếu bạn muốn dùng webcam: để 0
# # # # Nếu bạn muốn đọc từ file video: đổi thành đường dẫn, ví dụ: 'video.mp4'
# # # cap = cv2.VideoCapture("video/xetoi.mp4")
# # #
# # # if not cap.isOpened():
# # #     print("Không mở được camera hoặc video.")
# # #     exit()
# # #
# # # while True:
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         print("Kết thúc video hoặc không đọc được khung hình.")
# # #         break
# # #
# # #     # --- Tiền xử lý ảnh ---
# # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #     gray = cv2.GaussianBlur(gray, (5, 5), 0)
# # #     thresh = cv2.adaptiveThreshold(
# # #         gray, 255,
# # #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# # #         cv2.THRESH_BINARY,
# # #         11, 2
# # #     )
# # #
# # #     # --- Tìm contour ---
# # #     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # #     largest_rectangle = None
# # #     max_area = 0
# # #
# # #     # --- Duyệt và tìm vùng chữ nhật lớn nhất ---
# # #     for cnt in contours:
# # #         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
# # #         if len(approx) == 4:  # nếu là hình tứ giác
# # #             area = cv2.contourArea(cnt)
# # #             if area > max_area:
# # #                 max_area = area
# # #                 largest_rectangle = approx
# # #
# # #     # --- Vẽ và cắt vùng chữ nhật ---
# # #     if largest_rectangle is not None:
# # #         cv2.drawContours(frame, [largest_rectangle], -1, (0, 255, 0), 3)
# # #         x, y, w, h = cv2.boundingRect(largest_rectangle)
# # #         roi = frame[y:y+h, x:x+w]
# # #         cv2.putText(frame, "Rectangle detected", (x, y - 10),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# # #
# # #     # --- Hiển thị khung hình ---
# # #     cv2.imshow('Detect Rectangle', frame)
# # #
# # #     # Nhấn ESC để thoát
# # #     if cv2.waitKey(1) & 0xFF == 27:
# # #         break
# # #
# # # # ---- Kết thúc ----
# # # cap.release()
# # # cv2.destroyAllWindows()
# #
# #
# # import cv2
# # import numpy as np
# #
# # # ---- Nguồn video hoặc camera ----
# # cap = cv2.VideoCapture(0)   # 0 = webcam, hoặc thay bằng 'video.mp4'
# #
# # if not cap.isOpened():
# #     print("Không mở được camera hoặc video.")
# #     exit()
# #
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     # 1️⃣ Chuyển sang ảnh xám
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #
# #     # 2️⃣ Tăng độ tương phản (dùng CLAHE)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #     contrast = clahe.apply(gray)
# #
# #     # 3️⃣ Giảm nhiễu bằng bộ lọc Gauss
# #     blur = cv2.GaussianBlur(contrast, (5, 5), 0)
# #
# #     # 4️⃣ Nhị phân hóa ảnh bằng ngưỡng động (adaptive threshold)
# #     thresh = cv2.adaptiveThreshold(
# #         blur, 255,
# #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #         cv2.THRESH_BINARY,
# #         11, 2
# #     )
# #
# #     # 5️⃣ Phát hiện biên bằng Canny
# #     edges = cv2.Canny(thresh, 100, 200)
# #
# #     # 6️⃣ Tìm contour
# #     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     # 7️⃣ Lọc vùng có khả năng là biển số xe
# #     for cnt in contours:
# #         x, y, w, h = cv2.boundingRect(cnt)
# #         aspect_ratio = w / float(h)
# #         area = cv2.contourArea(cnt)
# #
# #         # điều kiện lọc cơ bản (tùy chỉnh cho phù hợp)
# #         if 2000 < area < 20000 and 2 < aspect_ratio < 5:
# #             # Vẽ khung bao quanh vùng nghi ngờ là biển số
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #             cv2.putText(frame, "Bien so xe", (x, y - 5),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# #
# #     # Hiển thị từng bước xử lý
# #     cv2.imshow("Anh goc", frame)
# #     cv2.imshow("Anh canny", edges)
# #
# #     # Nhấn ESC để thoát
# #     if cv2.waitKey(1) & 0xFF == 27:
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
#
#
#
# import cv2
# import numpy as np
#
# # ---- Mở camera hoặc video ----
# cap = cv2.VideoCapture('video/xetoi.mp4')  # Thay bằng 'video.mp4' nếu bạn có file video
#
# if not cap.isOpened():
#     print("Không mở được camera hoặc video.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 1️⃣ Chuyển sang ảnh xám
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 2️⃣ Tăng độ tương phản bằng CLAHE
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
#
#     # 3️⃣ Dùng các phép hình thái học để làm nổi bật chi tiết
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
#
#     # Top-hat: làm nổi vùng sáng nhỏ trên nền tối
#     tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
#
#     # Black-hat: làm nổi vùng tối nhỏ trên nền sáng
#     blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
#
#     # Kết hợp hai ảnh để tăng tương phản tổng thể
#     contrast = cv2.add(enhanced, tophat)
#     contrast = cv2.subtract(contrast, blackhat)
#
#     # 4️⃣ Giảm nhiễu bằng bộ lọc Gauss
#     blur = cv2.GaussianBlur(contrast, (5, 5), 0)
#
#     # 5️⃣ Nhị phân hóa ảnh với ngưỡng động
#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )
#
#     # 6️⃣ Phát hiện biên bằng Canny
#     edges = cv2.Canny(thresh, 100, 200)
#
#     # 7️⃣ Tìm và lọc contour theo tỉ lệ khung hình
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = w / float(h)
#         area = cv2.contourArea(cnt)
#
#         # Điều kiện lọc biển số (tuỳ chỉnh theo dữ liệu thực tế)
#         if 2000 < area < 20000 and 2 < aspect_ratio < 5:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, "Bien so xe", (x, y - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#     # 8️⃣ Hiển thị kết quả
#     cv2.imshow("Anh goc", frame)
#     cv2.imshow("Anh canny", edges)
#     cv2.imshow("Top-hat", tophat)
#     cv2.imshow("Black-hat", blackhat)
#     cv2.imshow("Tang tuong phan", contrast)
#
#     # Nhấn ESC để thoát
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
# # --- Bước 1: Đọc clip đầu vào ---
# video_path = "video/xetoi.mp4"  # Đường dẫn tới video
# cap = cv2.VideoCapture(video_path)
#
# frame_index = 0
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Kết thúc khi đọc hết clip
#
#     frame_index += 1
#     print(f"Đang xử lý frame {frame_index}...")
#
#     # --- Bước 2: Chuyển ảnh sang ảnh xám ---
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # --- Bước 3: Tăng độ tương phản bằng các phép toán hình thái học ---
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
#     blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
#     enhanced = cv2.add(gray, tophat)
#     enhanced = cv2.subtract(enhanced, blackhat)
#
#     # --- Bước 4: Giảm nhiễu bằng bộ lọc Gauss ---
#     blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
#
#     # --- Bước 5: Lấy ngưỡng bằng phương pháp Adaptive Threshold ---
#     binary = cv2.adaptiveThreshold(
#         blurred, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         19, 9
#     )
#
#     # --- Bước 6: Phát hiện cạnh bằng thuật toán Canny ---
#     edges = cv2.Canny(binary, 100, 200)
#
#     # --- Bước 7: Tìm các đường bao (contours) ---
#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     # --- Bước 8: Lọc theo tỉ lệ và diện tích để tìm vùng khả nghi là biển số ---
#     possible_plates = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = w / float(h)
#         area = w * h
#
#         # Điều kiện lọc biển số (có thể điều chỉnh tùy trường hợp thực tế)
#         if 2 < aspect_ratio < 6 and 1000 < area < 30000:
#             possible_plates.append((x, y, w, h))
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # --- Bước 9: Hiển thị kết quả ---
#     cv2.imshow("Frame goc", frame)
#     cv2.imshow("Anh xam", gray)
#     cv2.imshow("Bien so nghi ngo", edges)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# import math
#
# import cv2
# import numpy as np
#
# import Preprocess
#
# ADAPTIVE_THRESH_BLOCK_SIZE = 19
# ADAPTIVE_THRESH_WEIGHT = 9
#
# n = 1
#
# Min_char = 0.01
# Max_char = 0.09
#
# RESIZED_IMAGE_WIDTH = 20
# RESIZED_IMAGE_HEIGHT = 30
#
# img = cv2.imread("datatestbienso/1002.jpg")
# img = cv2.resize(img, dsize=(1920, 1080))
#
# ###################### If you want to try increasing the contrast #############
# # img2 = cv2.imread("1.jpg")
# # imgGrayscaleplate2, _ = Preprocess.preprocess(img)
# # imgThreshplate2 = cv2.adaptiveThreshold(imgGrayscaleplate2, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE ,ADAPTIVE_THRESH_WEIGHT )
# # cv2.imshow("imgThreshplate2",imgThreshplate2)
# ###############################################################
#
# ######## Upload KNN model ######################
# npaClassifications = np.loadtxt("classifications.txt", np.float32)
# npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
# npaClassifications = npaClassifications.reshape(
#     (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
# kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
# kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
# #########################
#
# ################ Image Preprocessing #################
# imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
# canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
# kernel = np.ones((3, 3), np.uint8)
# dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation
# # cv2.imshow("dilated_image",dilated_image)
#
# ###########################################
#
# ###### Draw contour and filter out the license plate  #############
# contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất
# # cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # Vẽ tất cả các ctour trong hình lớn
#
# screenCnt = []
# for c in contours:
#     peri = cv2.arcLength(c, True)  # Tính chu vi
#     approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
#     [x, y, w, h] = cv2.boundingRect(approx.copy())
#     ratio = w / h
#     # cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
#     # cv2.putText(img, str(ratio), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
#     if (len(approx) == 4):
#         screenCnt.append(approx)
#
#         cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
#
# if screenCnt is None:
#     detected = 0
#     print("No plate detected")
# else:
#     detected = 1
#
# if detected == 1:
#
#     for screenCnt in screenCnt:
#         cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe
#
#         ############## Find the angle of the license plate #####################
#         (x1, y1) = screenCnt[0, 0]
#         (x2, y2) = screenCnt[1, 0]
#         (x3, y3) = screenCnt[2, 0]
#         (x4, y4) = screenCnt[3, 0]
#         array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#         sorted_array = array.sort(reverse=True, key=lambda x: x[1])
#         (x1, y1) = array[0]
#         (x2, y2) = array[1]
#         doi = abs(y1 - y2)
#         ke = abs(x1 - x2)
#         angle = math.atan(doi / ke) * (180.0 / math.pi)
#
#         ####################################
#
#         ########## Crop out the license plate and align it to the right angle ################
#
#         mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
#         new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
#         # cv2.imshow("new_image",new_image)
#
#         # Cropping
#         (x, y) = np.where(mask == 255)
#         (topx, topy) = (np.min(x), np.min(y))
#         (bottomx, bottomy) = (np.max(x), np.max(y))
#
#         roi = img[topx:bottomx, topy:bottomy]
#         imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
#         ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2
#
#         if x1 < x2:
#             rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
#         else:
#             rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)
#
#         roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
#         imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
#         roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
#         imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
#
#         ####################################
#
#         #################### Prepocessing and Character segmentation ####################
#         kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
#         cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         cv2.imshow(str(n + 20), thre_mor)
#         cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số
#
#         ##################### Filter out characters #################
#         char_x_ind = {}
#         char_x = []
#         height, width, _ = roi.shape
#         roiarea = height * width
#
#         for ind, cnt in enumerate(cont):
#             (x, y, w, h) = cv2.boundingRect(cont[ind])
#             ratiochar = w / h
#             char_area = w * h
#             # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
#             # cv2.putText(roi, str(ratiochar), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
#
#             if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
#                 if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
#                     x = x + 1
#                 char_x.append(x)
#                 char_x_ind[x] = ind
#
#                 # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
#
#         ############ Character recognition ##########################
#
#         char_x = sorted(char_x)
#         strFinalString = ""
#         first_line = ""
#         second_line = ""
#
#         for i in char_x:
#             (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
#             cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             imgROI = thre_mor[y:y + h, x:x + w]  # Crop the characters
#
#             imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image
#             npaROIResized = imgROIResized.reshape(
#                 (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
#
#             npaROIResized = np.float32(npaROIResized)
#             _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest;
#             strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
#             cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
#
#             if (y < height / 3):  # decide 1 or 2-line license plate
#                 first_line = first_line + strCurrentChar
#             else:
#                 second_line = second_line + strCurrentChar
#
#         print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
#         roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
#         cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#
#         # cv2.putText(img, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
#         n = n + 1
#
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
# cv2.imshow('License plate', img)
#
# cv2.waitKey(0)

import cv2
import numpy as np

# ================================
# 1️⃣ Đọc ảnh và chuyển sang xám
# ================================
img = cv2.imread('bienso/databienso/803_plate1.jpg')
img= cv2.resize(img,(400,250),interpolation=cv2.INTER_CUBIC)
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ================================
# 2️⃣ Giảm nhiễu bằng Gaussian Blur
# ================================
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)


# ================================
# 3️⃣ Phát hiện biên bằng Canny
# ================================
edges = cv2.Canny(blur, 50, 150)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 3)

# Làm rõ vùng biên bằng phép giãn và co
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=3)
edges = cv2.erode(edges, kernel, iterations=1)

# ================================
# 4️⃣ Tìm contour
# ================================
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ================================
# 8️⃣ Hiển thị kết quả
# ================================
cv2.imshow("1. Anh goc", img)
cv2.imshow("2. Bien canny", edges)
cv2.imshow("3. Khung tu giac tim duoc", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
