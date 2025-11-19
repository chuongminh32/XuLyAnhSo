import cv2
import numpy as np

image_path = input("Vui lòng nhập đường dẫn đến ảnh: ")

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
else:
    img_negative = cv2.subtract(255, img_gray)
    cv2.imshow("Bai1: Anh Goc (Grayscale)", img_gray)
    cv2.imshow("Bai1: Anh Am Ban (Negative)", img_negative)

    print("Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
