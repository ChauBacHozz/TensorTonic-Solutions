import torch

class TransformPipeline:
    def __init__(self, mean: list, std: list):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns a normalized float32 tensor with shape (C, H, W).
        """
        t_img = (image / 255 - self.mean)/self.std
        return t_img.permute(2, 0, 1)
