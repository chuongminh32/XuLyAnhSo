import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"G:/hcmute/semeter1-term2-2526/XLAS/code/buoi5/img1.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, 255, -1)
    cv2.GaussianBlur(img, (21, 21), 0, dst=img)  # Làm mờ để test làm sắc nét


# Hàm hỗ trợ hiển thị
def show_plot(figure_num, title, images, subtitles, rows, cols):
    plt.figure(figure_num, figsize=(14, 8))
    plt.suptitle(title, fontsize=16, color="blue")
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(subtitles[i])
        plt.axis("off")


# bai 1: SOBEL FILTER (Đạo hàm bậc 1)
# Lưu ý: Dùng cv2.CV_64F để giữ giá trị âm của đạo hàm, sau đó lấy trị tuyệt đối
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Tính Magnitude (Độ lớn vector gradient): sqrt(x^2 + y^2)
magnitude = cv2.magnitude(sobel_x, sobel_y)

# Chuyển về uint8 để hiển thị
sobel_x_view = cv2.convertScaleAbs(sobel_x)
sobel_y_view = cv2.convertScaleAbs(sobel_y)
magnitude_view = cv2.convertScaleAbs(magnitude)

# bai 2: LAPLACIAN FILTER & SHARPENING
# Laplacian là đạo hàm bậc 2, cực nhạy với biên
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_view = cv2.convertScaleAbs(laplacian)

# Công thức làm sắc nét cơ bản: Ảnh gốc - Laplacian (tùy kernel)
# Ở đây dùng phép trừ có trọng số để tránh nhiễu quá mức
img_sharpened_lap = cv2.addWeighted(img, 1, cv2.convertScaleAbs(laplacian), -0.5, 0)


# bai 3: HIGH-BOOST FILTERING
# B1: Tạo ảnh mờ (Blurred)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# B2: Tạo Mask (Gốc - Mờ) -> Đây là phần chi tiết/cạnh
mask = cv2.subtract(img, blurred)
# B3: Cộng lại vào ảnh gốc theo công thức: G = Gốc + k * Mask
# Hoặc dùng addWeighted: Result = (1+k)*Img - k*Blur
k_values = [1.5, 3.0, 5.0]
highboost_results = []
highboost_titles = []

for k in k_values:
    # Công thức High-boost tính nhanh bằng addWeighted
    # alpha = 1 + k, beta = -k
    hb = cv2.addWeighted(img, 1.0 + k, blurred, -k, 0)
    highboost_results.append(hb)
    highboost_titles.append(f"High-boost (k={k})")

# Show nhóm 1: Đạo hàm bậc 1 (Sobel) & Bậc 2 (Laplacian)
show_plot(
    1,
    "Bài 1, 2 & 4: Sobel, Laplacian & Magnitude",
    [
        img,
        sobel_x_view,
        sobel_y_view,
        magnitude_view,
        laplacian_view,
        img_sharpened_lap,
    ],
    [
        "Ảnh Gốc",
        "Sobel X",
        "Sobel Y",
        "Gradient Magnitude",
        "Laplacian Edges",
        "Sharpened (Gốc - Lap)",
    ],
    2,
    3,
)

# Show nhóm 2: High-boost Filtering
# Gom ảnh gốc và mask vào để dễ so sánh
hb_imgs = [img, mask] + highboost_results
hb_lbls = ["Ảnh Gốc", "Unsharp Mask (Chi tiết)"] + highboost_titles

show_plot(2, "Bài 3: Unsharp Masking & High-boost Filtering", hb_imgs, hb_lbls, 2, 3)

plt.show()
