import cv2
import numpy as np

image = cv2.imread('image/screen_2.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('image/native_image.png', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv2.matchTemplate(image, template, cv2.TM_CCORR)
threshold = 0.95 # 임계치 설정
box_loc = np.where(result >= threshold) # 임계치 이상의 값들만 사용

# Check if any coordinates were found
template_found = len(box_loc[0]) > 0

print(f"Template found: {template_found}")

for box in zip(*box_loc[::-1]):
    startX, startY = box
    endX, endY = startX + w, startY + h
    cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 1)

cv2.imwrite('example1_result.png', image)