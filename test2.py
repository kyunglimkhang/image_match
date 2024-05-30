import cv2
import numpy as np

image = cv2.imread('couples.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('couple.png', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.95 # 임계치 설정
box_loc = np.where(result >= threshold) # 임계치 이상의 값들만 사용

for box in zip(*box_loc[::-1]):
    startX, startY = box
    endX, endY = startX + w, startY + h
    cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 1)

cv2.imwrite('result.png', image)