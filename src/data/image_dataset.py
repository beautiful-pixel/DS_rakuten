"""
Rakuten Image Dataset Module

Provides PyTorch Dataset class for loading Rakuten product images with pre-encoded labels.
Designed to work with canonical classes and splits for consistent model training and evaluation.

This module assumes labels are already encoded using CANONICAL_CLASSES from label_mapping.py.
"""
from pathlib import Path
from typing import Optional, Callable, Tuple

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RakutenImageDataset(Dataset):
    """
    PyTorch Dataset for Rakuten product images with pre-encoded canonical labels.

    This dataset loads product images from disk and returns them with their corresponding
    labels. Labels must be pre-encoded to canonical class IDs (0-26) before creating the
    dataset. Supports flexible image path construction via explicit image_path column or
    fallback to imageid/productid-based naming.

    The dataset integrates with PyTorch's DataLoader for efficient batched loading with
    optional data augmentation transforms.

    Attributes:
        image_dir: Root directory containing product image files (.jpg)
        df: DataFrame with image metadata and pre-encoded labels
        transform: Optional torchvision transforms to apply to images
        label_col: Name of the column containing encoded labels (0-26)
        image_paths: List of Path objects to image files (one per sample)
        labels: List of integer labels (one per sample, range 0-26)

    Examples:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ... ])
        >>> dataset = RakutenImageDataset(
        ...     dataframe=train_df,
        ...     image_dir="data/images/image_train",
        ...     transform=transform,
        ...     label_col="encoded_label",
        ... )
        >>> image, label = dataset[0]
        >>> print(image.shape, label)  # torch.Size([3, 224, 224]) 5

    Note:
        This dataset does NOT perform label encoding internally. Labels must be encoded
        using the canonical mapping (CANONICAL_CLASSES) before constructing the dataset.
        This ensures consistency across training, validation, and test splits.

    See Also:
        :func:`src.data.label_mapping.encode_labels`: Function to encode labels to canonical IDs
        :class:`torch.utils.data.Dataset`: Base PyTorch Dataset class
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
        """
        Initialize the Rakuten Image Dataset.

        Args:
            dataframe: DataFrame containing image metadata and labels. Must have
                at minimum the label_col (pre-encoded labels 0-26). If use_image_path_col
                is True, must also have image_path_col with absolute/relative paths.
                If use_image_path_col is False, must have 'imageid' and 'productid'
                columns for constructing image filenames.
            image_dir: Path to directory containing product image files. Can be absolute
                or relative. Examples: "data/images/image_train" or "/content/images".
            transform: Optional torchvision.transforms.Compose to apply to each image.
                If None, images are returned as PIL Images. Typically includes Resize,
                ToTensor, Normalize for model input preparation.
            label_col: Name of DataFrame column containing pre-encoded integer labels
                (range 0-26 for 27 canonical classes). Default "encoded_label" matches
                output from encode_labels() in label_mapping.py.
            use_image_path_col: Whether to use explicit image_path column from DataFrame.
                If True (recommended), reads paths directly from image_path_col. If False,
                falls back to constructing paths from imageid/productid. Default True.
            image_path_col: Name of DataFrame column containing image paths. Only used
                if use_image_path_col=True and column exists. Default "image_path".

        Raises:
            KeyError: If required columns (label_col, or imageid/productid when
                use_image_path_col=False) are missing from dataframe
            FileNotFoundError: If image_dir does not exist (raised on first __getitem__ call)

        Note:
            The dataset resets the DataFrame index to ensure integer indexing works correctly.
            Original index is not preserved.
        """
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
        """
        Return the total number of samples in the dataset.

        Returns:
            Number of image-label pairs available in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single image-label pair at the given index.

        Args:
            idx: Index of the sample to load (0 to len(dataset)-1)

        Returns:
            Tuple of (image, label) where:
                - image: Transformed image tensor if transform is set, else PIL Image
                - label: Integer class label (0-26 for canonical 27 classes)

        Raises:
            FileNotFoundError: If image file does not exist at the computed path
            IndexError: If idx is out of range [0, len(dataset))
            PIL.UnidentifiedImageError: If image file is corrupted or not a valid image

        Note:
            Images are loaded in RGB mode (3 channels). Grayscale or RGBA images
            are automatically converted to RGB.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label
