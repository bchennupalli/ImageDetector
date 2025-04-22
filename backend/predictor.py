import torch


class Predictor:
    """
    Handles predictions using the loaded model.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, image_tensor):
        """
        Predicts the class of the given image tensor.
        Returns:
            pred (int): 0 or 1 — predicted class
            prob (float): probability score (between 0 and 1)
        """
        if image_tensor is None:
            return None, None

        # Ensure batch dimension is present
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.sigmoid(output).item()  # Convert logits to probability
            pred = int(prob > 0.5)  # Binary decision

        return pred, prob
