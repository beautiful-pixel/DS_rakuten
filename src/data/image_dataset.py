from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RakutenImageDataset(Dataset):
    """
    Dataset that does NOT build a label mapping internally.
    It expects a pre-encoded label column (canonical id) to be present in the dataframe.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        label_col: str = "encoded_label",
        use_image_path_col: bool = True,
        image_path_col: str = "image_path",
    ):
        self.image_dir = Path(image_dir)
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

        if use_image_path_col and image_path_col in self.df.columns:
            # Preferred: load_data() already computed absolute/relative paths
            self.image_paths = self.df[image_path_col].tolist()
        else:
            # Fallback: build from imageid/productid
            self.image_paths = [
                self.image_dir / f"image_{row['imageid']}_product_{row['productid']}.jpg"
                for _, row in self.df.iterrows()
            ]

        # Labels are assumed to be already encoded with CANONICAL_CLASSES
        self.labels = self.df[self.label_col].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label
