import cv2, sys

path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi2/Gemini_Generated_Image_ywvle3ywvle3ywvl.png"
img = cv2.imread(path)
if img is None:
    sys.exit(f"Không thể mở ảnh: {path}")

blur = cv2.GaussianBlur(img, (5, 5), 0)

cv2.imshow("Color", img)
cv2.imshow("Blurred", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
