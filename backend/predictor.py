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
        Returns the binary prediction and probability score.
        """
        if image_tensor is None:
            return None, None

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.sigmoid(output)  # Convert logits to probability
            pred = (prob > 0.5).int().item()  # Convert to binary prediction

        return pred, prob.item()
