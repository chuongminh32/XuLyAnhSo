import cv2
import numpy as np
import os

INPUT_DIR = r"G:\hcmute\semeter1-term2-2526\XLAS\code\XLA_N3_PhamHanMinhChuong_23110187_BT_TUAN4_25-11-25\images_bai2"
OUTPUT_DIR = (
    r"G:\hcmute\semeter1-term2-2526\XLAS\code\XLA_N3_PhamHanMinhChuong_23110187_BT_TUAN4_25-11-25\output_lam_net_bai2"
)

D0_VALUES = [5, 15, 30]  # Giá trị D0 cho các bộ lọc thông cao
N_BWT_VALUES = [1, 2]  # Bậc n cho Butterworth High-pass


def create_d0_filter_mask(rows, cols):
    """Tạo ma trận khoảng cách D(u,v) từ tâm."""
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(cols)
    v = np.arange(rows)
    U, V = np.meshgrid(u, v)
    D = np.sqrt((U - center_col) ** 2 + (V - center_row) ** 2)
    return D


def merge_to_two_channel(H):
    """Chuyển mask 1 kênh sang 2 kênh (thực + ảo) cho DFT."""
    return cv2.merge([H, H])


# --- CÀI ĐẶT BỘ LỌC THÔNG CAO (HPF = 1 - LPF) ---
def ideal_high_pass_filter(D, D0):
    """Lọc Thông Cao Lý Tưởng (IHPF)."""
    H_lpf = np.zeros(D.shape, dtype=np.float32)
    H_lpf[D <= D0] = 1
    H_hpf = 1.0 - H_lpf
    return merge_to_two_channel(H_hpf)


def butterworth_high_pass_filter(D, D0, n):
    """Lọc Thông Cao Butterworth (BHPF)."""
    H_lpf = 1.0 / (1.0 + np.power(D / D0, 2 * n))
    H_hpf = 1.0 - H_lpf
    return merge_to_two_channel(H_hpf)


def gaussian_high_pass_filter(D, D0):
    """Lọc Thông Cao Gauss (GHPF)."""
    H_lpf = np.exp(-(D**2) / (2 * (D0**2)))
    H_hpf = 1.0 - H_lpf
    return merge_to_two_channel(H_hpf)


def laplacian_filter_frequency(D):
    """Bộ lọc Laplace trong miền Tần số: H(u,v) = -(u^2 + v^2) = -D^2."""
    H = -(D**2)
    H = H / np.max(np.abs(H))  # Chuẩn hóa để tránh giá trị quá lớn
    return merge_to_two_channel(H)


# --- HÀM XỬ LÝ VÀ LƯU ẢNH ---
def process_and_save_image(img_path, filename):
    """Thực hiện DFT, lọc và IDFT cho một ảnh."""

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

    # Khai báo tất cả các bộ lọc cần áp dụng
    filters = []
    for D0 in D0_VALUES:
        filters.append(("Ideal_HPF", D0, ideal_high_pass_filter(D, D0)))
        filters.append(("Gaussian_HPF", D0, gaussian_high_pass_filter(D, D0)))
    for n in N_BWT_VALUES:
        for D0 in D0_VALUES:
            filters.append(
                (f"Butterworth_HPF_n{n}", D0, butterworth_high_pass_filter(D, D0, n))
            )

    H_laplace = laplacian_filter_frequency(D)
    filters.append(("Laplace_HPF", 0, H_laplace))

    # Lặp và áp dụng từng bộ lọc
    for filter_name, D0, H_mask in filters:

        # Áp dụng bộ lọc
        dft_filtered = dft_shifted * H_mask

        # Biến đổi Fourier ngược và chuẩn hóa
        dft_ishift = np.fft.ifftshift(dft_filtered)
        img_back = cv2.idft(dft_ishift)
        img_processed = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        cv2.normalize(img_processed, img_processed, 0, 255, cv2.NORM_MINMAX)

        # Lưu ảnh kết quả
        base_name, ext = os.path.splitext(filename)
        # Tạo thư mục con
        output_sub_dir = os.path.join(
            OUTPUT_DIR, filter_name.split("_n")[0].replace("_HPF", "")
        )
        os.makedirs(output_sub_dir, exist_ok=True)

        if D0 == 0:  # Laplace
            output_name = f"Laplace_{base_name}{ext}"
        elif "Butterworth" in filter_name:
            n_val = filter_name.split("_n")[1]
            output_name = f"BWT_D{D0}_{n_val}_{base_name}{ext}"
        else:
            output_name = f"{filter_name.split('_HPF')[0]}_D{D0}_{base_name}{ext}"

        cv2.imwrite(
            os.path.join(output_sub_dir, output_name), img_processed.astype(np.uint8)
        )


def main():
    print("Bắt đầu xử lý ảnh miền tần số (Lọc làm nét)...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lấy danh sách ảnh
    image_files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

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
