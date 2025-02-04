from PIL import Image
from torchvision import transforms


class ImageProcessor:
    """
    Handles image loading and preprocessing.
    """

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Uncomment if normalization was used during training:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_path):
        """
        Loads and preprocesses the image.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
