import torch
import torch.nn as nn
from torchvision import models


class ModelLoader:
    """
    Loads a pre-trained ResNet50 model for binary classification.
    """

    def __init__(self, model_path, device):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Loads the ResNet50 model with the given weights and modifies the final layer.
        """
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        # Modify the final fully connected layer for binary classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)  # Binary classification (single logit)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def get_model(self):
        return self.model
