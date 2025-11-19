import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi5/img1.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    exit("Không tìm thấy ảnh!")
else:
    # Chập ảnh với 3x3 tùy chọn
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    res_bai1 = cv2.filter2D(img, -1, kernel)

    # so sánh padding
    pad = 50  
    # Zero Padding(viền đen): (srx, top, bottom, left, right, border_type, value)
    pad_zero = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    # Replicate Padding: Lặp lại pixel rìa
    pad_repl = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # minh họa sự khác nhau giữa convulatioin vs correlation
    # Mask bất đối xứng (để thấy sự khác biệt)
    k_asym = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # Correlation: Mặc định của filter2D
    res_corr = cv2.filter2D(img, -1, k_asym)
    # Convolution: Phải lật (flip) kernel 180 độ trước khi tính
    res_conv = cv2.filter2D(img, -1, cv2.flip(k_asym, -1))

    imgs = [img, res_bai1, pad_zero, pad_repl, res_corr, res_conv]
    titles = [
        "Gốc",
        "Mask 3x3",
        "Zero Padding",
        "Replicate Padding",
        "Correlation",
        "Convolution",
    ]

    plt.figure(figsize=(12, 7))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
