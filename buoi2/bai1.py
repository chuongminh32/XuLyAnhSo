import cv2

path = "G:/hcmute/semeter1-term2-2526/XLAS/code/buoi2/Gemini_Generated_Image_ywvle3ywvle3ywvl.png"

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Color", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Black_and_White", bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
