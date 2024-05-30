import cv2
import numpy as np

image = cv2.imread("couples.png", cv2.IMREAD_GRAYSCALE)
target = cv2.imread("couple.png", cv2.IMREAD_GRAYSCALE)
result_image = cv2.imread("couples.png")
w, h = target.shape[::-1]


result = cv2.matchTemplate(image, target, cv2.TM_SQDIFF_NORMED)
threshold = 0.80 # 임계치 설정
box_loc = np.where(result >= threshold) # 임계치 이상의 값들만 사용

for box in zip(*box_loc[::-1]):
    startX, startY = box
    endX, endY = startX + w, startY + h
    cv2.rectangle(result_image, (startX, startY), (endX, endY), (0,0,255), 1)

# flag = False
# for i in result:
#     if i.any() > threshold:
#         flag = True

# print("result", flag)

# minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
# x, y = minLoc
# h, w = target.shape

# result_image = cv2.rectangle(result_image, (x, y), (x +  w, y + h) , (0, 0, 255), 1)
cv2.imshow("result_image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()