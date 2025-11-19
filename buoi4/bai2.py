import cv2
import numpy as np

image_path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi4/img1.jpg"

gamma_val = 0.5

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
else:
    # Chuyển ảnh sang kiểu float32 để tính toán
    f = img_gray.astype(np.float32)

    # Tính g = c * ln(1 + f)
    # Thêm 1 để tránh log(0)
    f_plus_1 = f + 1.0

    # Tính logarit tự nhiên
    log_transform = cv2.log(f_plus_1)

    # Tìm f_max
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(f)

    # Tìm hằng số c = 255 / log(1 + f_max)
    c = 255 / np.log(1 + max_val)

    # Áp dụng hằng số c và chuẩn hóa về 0-255
    img_log = cv2.convertScaleAbs(log_transform, alpha=c)

    # Chuẩn hóa ảnh về dải [0, 1]
    f_norm = img_gray / 255.0

    # Áp dụng g = f^gamma
    gamma_transform = cv2.pow(f_norm, gamma_val)

    # Chuyển kết quả về 0-255
    img_gamma = cv2.convertScaleAbs(gamma_transform, alpha=255)

    cv2.imshow("Bien doi Logarit", img_log)
    cv2.imshow("Bien doi Gamma", img_gamma)

    print(f"Đã chạy Bài 2 (với Gamma={gamma_val}). Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
