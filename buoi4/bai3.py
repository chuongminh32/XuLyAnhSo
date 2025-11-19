import cv2
import numpy as np

image_path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi4/img1.jpg"

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
else:
    #  Thiết lập Ánh xạ
    r1, s1 = 50, 10
    r2, s2 = 150, 240

    #  Xây dựng LUT (Lookup Table)
    lut = np.zeros(256, dtype=np.uint8)

    # Tính độ dốc cho đoạn tuyến tính ở giữa
    slope = (s2 - s1) / (r2 - r1)

    for i in range(256):
        if i < r1:
            # Giá trị < 50 được ánh xạ về 0
            lut[i] = 0
        elif i > r2:
            # Giá trị > 150 được ánh xạ về 255
            lut[i] = 255
        else:
            # Ánh xạ tuyến tính cho đoạn [50, 150] -> [10, 240]
            s = s1 + slope * (i - r1)
            lut[i] = np.uint8(round(s))

    #  Áp dụng Biến đổi
    img_lut = cv2.LUT(img_gray, lut)

    cv2.imshow("Bai 3: Anh Goc", img_gray)
    cv2.imshow("Bai 3: Bien doi (LUT)", img_lut)

    print("Đã chạy Bài 3. Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
