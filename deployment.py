import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load("svm_model.joblib")

def preprocess_image(image):
    # Resize the image to 64x64
    resized_img = cv2.resize(image, (64, 64))
    # Flatten the image into a feature vector
    flattened_img = resized_img.flatten()
    # Standardize the feature vector
    flattened_img = flattened_img.reshape(1, -1)
    return flattened_img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read the image file
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)