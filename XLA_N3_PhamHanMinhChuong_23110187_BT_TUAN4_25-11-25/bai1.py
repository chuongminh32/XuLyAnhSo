import cv2
import numpy as np
import os

INPUT_DIR = r"G:\hcmute\semeter1-term2-2526\XLAS\code\XLA_N3_PhamHanMinhChuong_23110187_BT_TUAN4_25-11-25\images_bai1"
OUTPUT_DIR = r"G:\hcmute\semeter1-term2-2526\XLAS\code\XLA_N3_PhamHanMinhChuong_23110187_BT_TUAN4_25-11-25\output_lam_min_bai1"
D0_VALUES = [10, 30, 60]
N_BWT_VALUES = [2, 4]


def create_d0_filter_mask(rows, cols):
    """Tạo ma trận khoảng cách D(u,v) từ tâm."""
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(cols)
    v = np.arange(rows)
    U, V = np.meshgrid(u, v)
    D = np.sqrt((U - center_col) ** 2 + (V - center_row) ** 2)
    return D


# Lọc Thông Thấp Lý Tưởng (Ideal Low-pass Filter - ILPF)
def ideal_low_pass_filter(D, D0):
    """Công thức: H = 1 nếu D <= D0, 0 nếu D > D0."""
    H = np.zeros(D.shape, dtype=np.float32)
    H[D <= D0] = 1
    return cv2.merge([H, H])


# Lọc Thông Thấp Butterworth (Butterworth Low-pass Filter - BLPF)
def butterworth_low_pass_filter(D, D0, n):
    """Công thức: H(u,v) = 1 / [1 + (D/D0)^(2n)]"""
    H = 1.0 / (1.0 + np.power(D / D0, 2 * n))
    return cv2.merge([H, H])


# Lọc Thông Thấp Gauss (Gaussian Low-pass Filter - GLPF)
def gaussian_low_pass_filter(D, D0):
    """Công thức: H(u,v) = exp(-D^2 / (2*D0^2))"""
    H = np.exp(-(D**2) / (2 * (D0**2)))
    return cv2.merge([H, H])


def process_and_save_image(img_path, filename):
    """Thực hiện DFT, lọc và IDFT cho một ảnh."""

    # Đọc ảnh (grayscale) và chuyển sang float32
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Bỏ qua: Không thể đọc ảnh tại {img_path}")
        return

    img_float32 = np.float32(img)
    rows, cols = img_float32.shape

    # Thực hiện DFT và dịch tâm phổ
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # Tạo ma trận khoảng cách D(u,v) chung
    D = create_d0_filter_mask(rows, cols)

    # Khởi tạo danh sách lọc
    filters = {
        "Ideal_LPF": lambda D0: ideal_low_pass_filter(D, D0),
        "Gaussian_LPF": lambda D0: gaussian_low_pass_filter(D, D0),
    }

    # Thêm Butterworth với các bậc n
    for n in N_BWT_VALUES:
        # Gán lambda function với giá trị n cố định
        filters[f"Butterworth_LPF_n{n}"] = lambda D0, n=n: butterworth_low_pass_filter(
            D, D0, n
        )

    for filter_name, filter_func in filters.items():

        # Tạo thư mục con
        output_sub_dir = os.path.join(OUTPUT_DIR, filter_name.split("_n")[0])
        os.makedirs(output_sub_dir, exist_ok=True)

        for D0 in D0_VALUES:
            # Cài đặt và Áp dụng bộ lọc
            H_mask = filter_func(D0)
            dft_filtered = dft_shifted * H_mask

            # Biến đổi Fourier ngược và chuẩn hóa
            dft_ishift = np.fft.ifftshift(dft_filtered)
            img_back = cv2.idft(dft_ishift)
            img_processed = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            cv2.normalize(img_processed, img_processed, 0, 255, cv2.NORM_MINMAX)

            # Lưu ảnh kết quả
            base_name, ext = os.path.splitext(filename)

            if "Butterworth" in filter_name:
                output_name = f"BWT_D{D0}_{filter_name.split('_n')[1]}_{base_name}{ext}"
            else:
                output_name = f"{filter_name.split('_LPF')[0]}_D{D0}_{base_name}{ext}"

            cv2.imwrite(
                os.path.join(output_sub_dir, output_name),
                img_processed.astype(np.uint8),
            )


def main():
    print("Bắt đầu xử lý ảnh miền tần số (Lọc làm mịn)...")

    # Tạo thư mục output chính
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lấy danh sách ảnh trong thư mục INPUT_DIR
    try:
        image_files = [
            f
            for f in os.listdir(INPUT_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    except FileNotFoundError:
        print(
            f"Lỗi: Không tìm thấy thư mục đầu vào tại '{INPUT_DIR}'. Vui lòng kiểm tra lại đường dẫn."
        )
        return

    if not image_files:
        print(f"Lỗi: Thư mục '{INPUT_DIR}' trống hoặc không chứa file ảnh.")
        return

    for filename in image_files:
        img_path = os.path.join(INPUT_DIR, filename)
        print(f"-> Đang xử lý: {filename}")
        process_and_save_image(img_path, filename)

    print(f"\nHoàn thành! Kết quả đã được lưu trong thư mục '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
