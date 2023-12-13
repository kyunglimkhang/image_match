from flask import Flask, request, jsonify
import cv2 as cv

app = Flask(__name__)

@app.route("/")
def index():
	return "Hello, World!"

@app.route('/match', methods=['GET'])
def match_images():
    try:
        # Get image data and template data from the request
        screen_image = request.args.get('screen_image', '')
        target_image = request.args.get('target_image', '')

        if not screen_image or not target_image:
            return jsonify({'error': 'Missing image or template data'})

        response = template_matching(screen_image, target_image)

        return response

    except Exception as e:
        return jsonify({'error': str(e)})

def template_matching(screen_image, target_image):

    screen = cv.imread(screen_image, cv.IMREAD_GRAYSCALE)          
    target = cv.imread(target_image, cv.IMREAD_GRAYSCALE)          

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(target, None)
    kp2, des2 = sift.detectAndCompute(screen, None)

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

if __name__ == '__main__':
    app.run(debug=True)
