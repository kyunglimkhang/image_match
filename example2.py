import cv2
import numpy as np

image = cv2.imread('image/couples.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('image/couple2.png', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

startX, startY = max_loc # 만약 cv.TM_SQDIFF 혹은 cv.TM_SQDIFF_NORMED를 사용했을경우 최솟값을 사용해야한다.
endX, endY = startX + w, startY + h
cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 1)

cv2.imwrite('example2_result.png', image)