import cv2
import numpy as np

image_path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi4/img1.jpg"

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
else:
    #  Cân bằng Histogram Toàn cục
    img_global_eq = cv2.equalizeHist(img_gray)

    #  Cân bằng Histogram Thích ứng Cục bộ (CLAHE)
    # Tạo đối tượng CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Áp dụng CLAHE
    img_clahe = clahe.apply(img_gray)

    #  Hiển thị
    cv2.imshow("Bai 4: Anh Goc", img_gray)
    cv2.imshow("Bai 4: Can bang Global (equalizeHist)", img_global_eq)
    cv2.imshow("Bai 4: Can bang Cuc bo (CLAHE)", img_clahe)

    print("Đã chạy Bài 4. Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
