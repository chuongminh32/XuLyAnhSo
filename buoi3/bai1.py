import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh xám
path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi2/Gemini_Generated_Image_ywvle3ywvle3ywvl.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
k = 8
L = 2 ** k

# Ép kiểu sang int để tránh tràn số
neg = (L - 1) - img.astype(int)

# Đảm bảo giá trị nằm trong [0,255], rồi chuyển về uint8
neg = np.clip(neg, 0, 255).astype(np.uint8)

# Biến đổi log: S = c * log(1 + r)
c_log = 255 / np.log(1 + np.max(img))
log_img = np.uint8(c_log * np.log(1 + img))

#  Biến đổi mũ (Power-law): S = c * r^γ
gamma = 0.5
c_gamma = 255 / (np.max(img) ** gamma)
pow_img = np.uint8(c_gamma * (img ** gamma))

# Hiển thị
titles = ['Gốc', 'Âm bản', 'Log', f'Power-law (γ={gamma})']
imgs = [img, neg, log_img, pow_img]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
