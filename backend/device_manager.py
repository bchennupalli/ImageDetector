import torch


class DeviceManager:
    """
    Manages the selection of the best available device.
    """

    @staticmethod
    def get_device():
        """Returns the best available device for computation."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
