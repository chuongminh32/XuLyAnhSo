import cv2
import numpy as np
import time

# Đọc và hiển thị ảnh gốc
img_path = "G:/code/PY/ImagesProcessing/buoi1/images/img1.png"
img = cv2.imread(img_path)
cv2.imshow("HW2.1 - Original", img)
cv2.waitKey(0)

# Tách 3 kênh màu và hiển thị từng kênh
b, g, r = cv2.split(img)
zeros = np.zeros_like(b)
cv2.imshow("HW2.2 - Blue",  cv2.merge([b, zeros, zeros]))
cv2.imshow("HW2.2 - Green", cv2.merge([zeros, g, zeros]))
cv2.imshow("HW2.2 - Red",   cv2.merge([zeros, zeros, r]))
cv2.waitKey(0)

# Chuyển sang ảnh xám và hiển thị
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("HW2.3 - Gray", gray)
cv2.waitKey(0)

# Xoay ảnh 100 lần, mỗi lần 5°, dừng 0.1s (thoát nếu nhấn ESC)
h, w = img.shape[:2]
center = (w // 2, h // 2)
for i in range(100):
    angle = i * 5
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h))
    cv2.imshow("HW2.4 - Rotating", rotated)
    if cv2.waitKey(100) & 0xFF == 27:  # ESC -> thoát
        break

# Cắt 1/4 kích thước từ tâm ảnh
ch, cw = h // 4, w // 4
center_crop = img[ch: h - ch, cw: w - cw]
cv2.imshow("HW2.5 - Center Crop", center_crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
