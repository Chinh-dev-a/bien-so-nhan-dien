import cv2
import numpy as np
from google.  collab.patches import cv2_imshow

original_image = cv2.imread("/content/building.jpeg")
cv2_imshow(original_image)
color_to_gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
cv2_imshow(color_to_gray_image)
kernelSizes = [(3, 3), (5, 5), (7, 7)]
for kernelSize in kernelSizes:
	# apply an "opening" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(color_to_gray_image, cv2.MORPH_OPEN, kernel)
cv2_imshow(opening)
    # apply an "opening" operation
closing = cv2.morphologyEx(color_to_gray_image, cv2.MORPH_CLOSE, kernel)
cv2_imshow(closing)

ret,thresh_img_1 = cv2.threshold(color_to_gray_image,127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_contours = np.zeros(original_image.shape)
cv2.drawContours(image_contours, contours, -1, (255,0,0), 3)
cv2_imshow(image_contours)


ret,thresh_img_2 = cv2.threshold(opening,127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_contours = np.zeros(original_image.shape)
cv2.drawContours(image_contours, contours, -1, (0,255,0), 3)
cv2_imshow(image_contours)

ret,thresh_img_3 = cv2.threshold(closing,127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_contours = np.zeros(original_image.shape)
cv2.drawContours(image_contours, contours, -1, (0,0,255), 3)
cv2_imshow(image_contours)
