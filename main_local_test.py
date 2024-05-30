import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

template = cv.imread('image/test_banner.png',cv.IMREAD_GRAYSCALE)          # queryImage
image = cv.imread('image/test_screenshot.png',cv.IMREAD_GRAYSCALE)          # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(template, None)
kp2, des2 = sift.detectAndCompute(image, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)     # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

# KNN matching
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test as per Lowe's paper
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Set a threshold for the number of good matches
min_good_matches = 50

# Print and return the result
template_found = len(good_matches) >= min_good_matches
print(f"Template found: {template_found}, (good_matches : {len(good_matches)})")

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(template, kp1, image, kp2, matches, None, **draw_params)
plt.imshow(img3,),plt.show()



#http://127.0.0.1:5000/match?screen_image=/Users/kyunglimkang/Sites/side/imageMatchAPI/image/screen_2.png&target_image=/Users/kyunglimkang/Sites/side/imageMatchAPI/image/native_image_2.png