from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import dlib
import numpy as np
import face_recognition
import pymysql
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Auto-download model if not exists
MODEL_URL = "https://drive.google.com/uc?export=download&id=1cJaW1v_8vWXSeia97_3a5MWWMkIutzNE"
MODEL_NAME = "shape_predictor_68_face_landmarks.dat"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_NAME}...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded.")

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)
# detector = dlib.get_frontal_face_detector()
# predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
# predictor = dlib.shape_predictor(predictor_path)

# Thresholds
EAR_THRESHOLD = 0.3
FRAME_COUNT = 3
FACE_MATCH_THRESHOLD = 0.45

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
    B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
    C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
    return (A + B) / (2.0 * C)

# Blink Detection for Liveness
def detect_blink(gray, face):
    landmarks = predictor(gray, face)
    left_eye = landmarks.parts()[36:42]
    right_eye = landmarks.parts()[42:48]

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0

    return ear < EAR_THRESHOLD

# Extract Face Encoding
def extract_face_encoding(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if len(face_encodings) == 0:
        return None
    return face_encodings[0]

# Extract Iris Features
def extract_iris_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, False

    blink_detected = False
    for face in faces:
        if detect_blink(gray, face):
            blink_detected = True
        landmarks = predictor(gray, face)
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]
        left_eye_points = np.array([[p.x, p.y] for p in left_eye])
        right_eye_points = np.array([[p.x, p.y] for p in right_eye])

        iris_features = np.concatenate((left_eye_points, right_eye_points), axis=0)
        return iris_features, blink_detected

    return None, False

@app.route('/verify_with_stored_data', methods=['POST'])
def verify_with_stored_data():
    try:
        data = request.get_json()
        image_data = data.get('image', '').split(',')[1]
        stored_iris_data = data.get('stored_iris_data', '')
        
        if not image_data or not stored_iris_data:
            return jsonify({
                "status": "error", 
                "message": "Missing data. Please provide both image and stored iris data."
            })

        # Decode and convert image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Perform iris extraction and liveness detection
        iris_features, blink_detected = extract_iris_features(image)

        if not blink_detected:
            return jsonify({
                "status": "error", 
                "message": "Liveness check failed. Please blink your eyes and try again."
            })

        # Extract face encoding
        captured_encoding = extract_face_encoding(image)
        if captured_encoding is None:
            return jsonify({
                "status": "error", 
                "message": "No face detected in the image. Please try again."
            })

        # Parse the stored encoding from the passed data
        try:
            stored_encoding = np.array([float(x) for x in stored_iris_data.split(",")])
        except Exception as e:
            return jsonify({
                "status": "error", 
                "message": f"Invalid stored data format: {str(e)}"
            })

        # Perform comparison
        distance = np.linalg.norm(captured_encoding - stored_encoding)

        if distance < FACE_MATCH_THRESHOLD:
            return jsonify({
                "status": "success", 
                "message": "Face Verified Successfully!",
                "distance": float(distance)
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Face verification failed. Please try again.",
                "distance": float(distance)
            })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Verification error: {str(e)}"})        

if __name__ == '__main__':
    app.run(debug=True)