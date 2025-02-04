import torch
import torch.nn as nn
from torchvision import models


class ModelLoader:
    """
    Loads a pre-trained ResNet18 model for binary classification.
    """

    def __init__(self, model_path, device):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Loads the ResNet18 model with the given weights and modifies the final layer.
        """
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)  # Binary classification

        # Load the trained model weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def get_model(self):
        """Returns the loaded model."""
        return self.model
