from flask import Flask, request, jsonify
import cv2 as cv
import base64
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

@app.route("/")
def index():
	return "Hello, World!"

@app.route('/match', methods=['POST'])
def match_images():
    try:
        data = request.get_json()

        # Get image URLs from the request
        screen_image_url = data.get('screen_image', '')
        target_image_url = data.get('target_image', '')

        if not screen_image_url or not target_image_url:
            return jsonify({'error': 'Missing image URL'})

        # Download images from URLs
        screen_image = download_image(screen_image_url)
        target_image = download_image(target_image_url)

        response = template_matching(screen_image, target_image)

        return response

    except Exception as e:
        return jsonify({'error': str(e)})

def template_matching(screen_image, target_image):        

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(target_image, None)
    kp2, des2 = sift.detectAndCompute(screen_image, None)

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
    match_result = len(good_matches) >= min_good_matches
    
    if (match_result):
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

        img3 = cv.drawMatchesKnn(target_image, kp1, screen_image, kp2, matches, None, **draw_params)

    return jsonify({
            'Template found': match_result,
            'good_matches': len(good_matches)
        })


def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        # Read the image from the response content
        image_data = BytesIO(response.content)
        image = cv.imdecode(np.frombuffer(image_data.read(), np.uint8), cv.IMREAD_GRAYSCALE)
        return image
    else:
        raise Exception(f"Failed to download image from URL: {image_url}")
