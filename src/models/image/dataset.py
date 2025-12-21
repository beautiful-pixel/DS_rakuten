import torch
from ..base.dataset import BaseDataset


class ImageDataset(BaseDataset):
    """
    Dataset image PyTorch.

    Args:
        images (np.ndarray | torch.Tensor)
        labels (list[int] | None)
    """

    def __init__(self, images, labels=None):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {"pixel_values": self.images[idx]}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
