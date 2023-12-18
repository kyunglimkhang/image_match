from flask import Flask, request, jsonify
import cv2 as cv
import base64
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
	return "Hello, World!"

@app.route('/match', methods=['POST'])
def match_images():
    try:
        data = request.get_json()

        # Get base64-encoded image data from the request
        screen_image_base64 = data.get('screen_image', '')
        target_image_base64 = data.get('target_image', '')

        if not screen_image_base64 or not target_image_base64:
            return jsonify({'error': 'Missing image or template data'})

        # Decode base64-encoded images
        screen_image = decode_base64_image(screen_image_base64)
        target_image = decode_base64_image(target_image_base64)

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

    return jsonify({
            'Template found': match_result,
            'good_matches': len(good_matches)
        })

def decode_base64_image(encoded_image):
    # Remove the data URL prefix (e.g., 'data:image/png;base64,')
    encoded_image = encoded_image.split(',')[1]

    # Decode base64 and convert to NumPy array
    decoded_image = np.frombuffer(base64.b64decode(encoded_image), dtype=np.uint8)

    # Decode the image using OpenCV
    image = cv.imdecode(decoded_image, cv.IMREAD_GRAYSCALE)

    return image

if __name__ == '__main__':
    app.run(debug=True)
