import cv2, sys

path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi2/Gemini_Generated_Image_ywvle3ywvle3ywvl.png"
img = cv2.imread(path)
if img is None:
    sys.exit(f"Không thể mở ảnh: {path}")

edges = cv2.Canny(
    cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 100, 200
)

cv2.imshow("Original", img)
cv2.imshow("Edges (Canny)", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
