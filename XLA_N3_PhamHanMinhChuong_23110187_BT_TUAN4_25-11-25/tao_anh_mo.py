import cv2
import numpy as np
import os
import glob

# Thư mục nguồn (Đổi thành đường dẫn thư mục ảnh gốc của bạn)
INPUT_DIR = "images_original"
# Thư mục đích: nơi chứa ảnh mờ nhẹ cho Bài tập 2
OUTPUT_DIR = "images_blurred_color"
# Tần số cắt. D0 lớn (30-50) sẽ tạo ảnh mờ NHẸ hơn D0 nhỏ (10-20).
D0 = 30
MAX_IMAGES_TO_PROCESS = 10

# --- HÀM TẠO MẶT NẠ LỌC THÔNG THẤP GAUSS ---


def gaussian_low_pass_filter_mask(rows, cols, D0):
    """Tạo mặt nạ Lọc Thông Thấp Gauss (GLPF) 1 kênh."""

    # Tính tọa độ trung tâm và khoảng cách D(u,v)
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(cols)
    v = np.arange(rows)
    U, V = np.meshgrid(u, v)
    D = np.sqrt((U - center_col) ** 2 + (V - center_row) ** 2)

    # Công thức GLPF
    H = np.exp(-(D**2) / (2 * (D0**2)))

    return H.astype(np.float32)


# --- .HÀM XỬ LÝ DFT/IDFT CHO MỘT KÊNH MÀU ---


def process_single_channel(channel, H_mask_2ch):
    """Áp dụng DFT, Lọc và IDFT cho một kênh màu."""

    #  Biến đổi Fourier và Dịch Tâm (DFT & fftshift)
    dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    #  Áp dụng Lọc
    dft_filtered = dft_shifted * H_mask_2ch

    # Biến đổi Fourier Ngược (ifftshift & IDFT)
    dft_ishift = np.fft.ifftshift(dft_filtered)
    img_back = cv2.idft(dft_ishift)

    # Lấy biên độ và chuẩn hóa
    channel_processed = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(channel_processed, channel_processed, 0, 255, cv2.NORM_MINMAX)

    return channel_processed


def blur_and_save_color_images(input_dir, output_dir, D0, max_count):

    print(f"Bắt đầu tạo {max_count} ảnh màu mờ nhẹ từ thư mục: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Lấy danh sách ảnh
    search_path = os.path.join(input_dir, "*.*")
    image_paths = glob.glob(search_path)[:max_count]

    if not image_paths:
        print(f"Lỗi: Thư mục '{input_dir}' trống hoặc không chứa file ảnh.")
        print("Vui lòng kiểm tra đường dẫn và nội dung thư mục.")
        return

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"-> Xử lý ảnh {i+1}/{len(image_paths)}: {filename}")

        # Đọc ảnh màu (Mặc định là BGR)
        img_color = cv2.imread(img_path)
        if img_color is None:
            print(f"Bỏ qua: Không thể đọc ảnh {filename}")
            continue

        rows, cols, _ = img_color.shape

        # Tách các kênh màu
        b, g, r = cv2.split(img_color)

        # Chuyển sang float32
        b_float = np.float32(b)
        g_float = np.float32(g)
        r_float = np.float32(r)

        # Tạo Mask chung cho tất cả các kênh
        H_mask_1ch = gaussian_low_pass_filter_mask(rows, cols, D0)
        H_mask_2ch = cv2.merge([H_mask_1ch, H_mask_1ch])

        # Xử lý từng kênh
        b_processed = process_single_channel(b_float, H_mask_2ch)
        g_processed = process_single_channel(g_float, H_mask_2ch)
        r_processed = process_single_channel(r_float, H_mask_2ch)

        # Ghép các kênh lại và lưu ảnh
        img_processed_color = cv2.merge([b_processed, g_processed, r_processed])

        # Lưu ảnh
        output_name = f"color_blurred_{filename}"
        output_path = os.path.join(output_dir, output_name)
        # Chuyển sang uint8 trước khi lưu
        cv2.imwrite(output_path, img_processed_color.astype(np.uint8))

    print(
        f"\nHoàn thành! {len(image_paths)} ảnh màu mờ nhẹ đã được lưu trong thư mục '{output_dir}'."
    )


if __name__ == "__main__":
    INPUT_DIR = r"G:\hcmute\semeter1-term2-2526\XLAS\code\buoi8_25-11-25\anh"

    blur_and_save_color_images(INPUT_DIR, OUTPUT_DIR, D0, MAX_IMAGES_TO_PROCESS)
