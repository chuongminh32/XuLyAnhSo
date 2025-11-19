import cv2, sys

img = cv2.imread(
    "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi2/Gemini_Generated_Image_ywvle3ywvle3ywvl.png"
)
if img is None:
    sys.exit("Không thể mở ảnh")

edges = cv2.Canny(
    cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 100, 200
)
h, w = img.shape[:2]
cropped = img[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
