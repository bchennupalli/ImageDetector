from model_loader import ModelLoader
from predictor import Predictor
from image_processor import ImageProcessor
from device_manager import DeviceManager
import os

if __name__ == "__main__":
    try:
        # Select the best available device
        device = DeviceManager.get_device()

        # Define paths
        image_path = "images/44faf85e-a4a6-43a0-92cb-b88ef4f62c39.webp"
        model_path = "model.pth"

        # Ensure the model and image exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load the model
        model_loader = ModelLoader(model_path, device)
        model = model_loader.get_model()

        # Process the image
        image_processor = ImageProcessor()
        image_tensor = image_processor.load_image(image_path)

        if image_tensor is None:
            raise ValueError("Image processing failed. Check the image format and path.")

        # Predict the image class
        predictor = Predictor(model, device)
        prediction, probability = predictor.predict(image_tensor)

        if prediction is not None:
            if prediction == 0:
                print("Human Captured Image")
            else:
                print("AI Generated Image")
        else:
            print("Prediction failed.")

    except Exception as e:
        print(f"Error: {e}")
