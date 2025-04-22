from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os

from model_loader import ModelLoader
from predictor import Predictor
from image_processor import ImageProcessor
from device_manager import DeviceManager

app = Flask(__name__, static_folder="../frontend/static", template_folder="../frontend")
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = DeviceManager.get_device()
model_loader = ModelLoader('model.pth', device)
model = model_loader.get_model()
image_processor = ImageProcessor()
predictor = Predictor(model, device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image_tensor = image_processor.load_image(filepath)
    prediction, probability = predictor.predict(image_tensor)

    result = "AI Generated Image" if prediction == 1 else "Human Captured Image"
    print(f"Prediction: {result}, Probability: {probability:.4f}")
    return jsonify({"result": result, "probability": f"{probability:.4f}"})


@app.route('/uploads/<path:path>')
def serve_static(path):
    return send_from_directory("../frontend/static", path)

if __name__ == "__main__":
    app.run(debug=True)