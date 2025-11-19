import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi5/img1.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    exit("Lỗi: Không tìm thấy ảnh!")


def add_salt_pepper(image, noise_ratio=0.05):
    noisy_image = image.copy()
    h, w = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = 0  # Tiêu
        else:
            noisy_image[row, col] = 255  # Muối
    return noisy_image


# Bài 1
b1_avg = cv2.blur(img, (5, 5))
b1_gauss = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
# Bài 2
noisy = add_salt_pepper(img)
b2_avg_denoise = cv2.blur(noisy, (5, 5))
b2_med_denoise = cv2.medianBlur(noisy, 5)

plt.figure(figsize=(12, 6))
items = [
    (img, "Goc"),
    (b1_avg, "B1: Avg 5x5"),
    (b1_gauss, "B1: Gauss sig=1.5"),
    (noisy, "B2: Nhieu muoi tieu"),
    (b2_avg_denoise, "B2: Khu Avg"),
    (b2_med_denoise, "B2: Khu Median"),
]
for i, (im, title) in enumerate(items):
    plt.subplot(2, 3, i + 1)
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")

print("Đang hiện Bài 1 & 2. Hãy ĐÓNG cửa sổ đồ thị để chạy Bài 3...")
plt.show()

win_name = "Bai 3: Interactive Filter (Nhan 'q' de thoat)"
cv2.namedWindow(win_name)
cv2.createTrackbar("K Size", win_name, 1, 20, lambda x: None)
cv2.createTrackbar("Type", win_name, 0, 2, lambda x: None)

cv2.imshow(win_name, img)

while True:
    try:
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Đã đóng cửa sổ bằng nút X.")
            break
    except:
        break

    # Lấy giá trị thanh trượt
    try:
        k = cv2.getTrackbarPos("K Size", win_name)
        ftype = cv2.getTrackbarPos("Type", win_name)
    except cv2.error:
        break  # Thoát nếu không tìm thấy trackbar (phòng hờ)

    k = k if k % 2 != 0 else k + 1  # Đảm bảo k là số lẻ

    if ftype == 0:
        res, txt = cv2.blur(img, (k, k)), f"Average: {k}x{k}"
    elif ftype == 1:
        res, txt = cv2.medianBlur(img, k), f"Median: {k}"
    else:
        res, txt = cv2.GaussianBlur(img, (k, k), 0), f"Gaussian: {k}x{k}"

    # Vẽ chữ lên ảnh hiển thị
    view = res.copy()
    cv2.putText(view, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    cv2.imshow(win_name, view)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
