import warnings
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class RakutenImageDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        label_col: str = "prdtypecode",
        verify_images: bool = True,
        remove_missing: bool = True,
    ):
        self.dataframe = dataframe.copy().reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_col = label_col

        # Validate required columns
        required_cols = ["imageid", "productid", label_col]
        missing_cols = [c for c in required_cols if c not in self.dataframe.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        # Pre-build and cache all image paths (lightweight, avoids repeated path construction)
        self.image_paths = self._build_image_paths()

        # Label mapping: try to detect if labels are already 0..n-1
        unique_labels = sorted(self.dataframe[label_col].unique())
        is_already_zero_indexed = (
            pd.api.types.is_integer_dtype(self.dataframe[label_col])
            and unique_labels == list(range(len(unique_labels)))
            and min(unique_labels) == 0
        )

        if is_already_zero_indexed:
            # Labels are already 0, 1, ..., n-1
            self.dataframe["label_idx"] = self.dataframe[label_col].astype(int)
            self.num_classes = len(unique_labels)
            self.label_mapping = None
            print(f"✓ Labels already zero-indexed: {self.num_classes} classes")
        else:
            # Map raw labels (e.g. 10, 2280, ...) to 0..n-1
            self.label_mapping = {
                label: idx for idx, label in enumerate(unique_labels)
            }
            self.dataframe["label_idx"] = self.dataframe[label_col].map(
                self.label_mapping
            )
            self.num_classes = len(self.label_mapping)
            print(f"✓ Created label mapping: {self.num_classes} classes")
            print(
                f"  Example labels: {unique_labels[:5]}"
                f"{' ...' if len(unique_labels) > 5 else ''}"
            )

        # Verify image files (and optionally drop missing ones)
        if verify_images:
            self._verify_images(remove_missing=remove_missing)

        # Pre-build and cache paths & labels (avoids DataFrame access in __getitem__)
        self.image_paths = self._build_image_paths()
        self.labels = self.dataframe["label_idx"].tolist()

        print(
            f"✓ RakutenImageDataset initialized: {len(self)} samples, "
            f"{self.num_classes} classes"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_image_paths(self) -> list:
        """
        Pre-build all image paths at initialization time.
        This avoids repeated path construction in __getitem__.
        Returns a list of Path objects indexed by dataset position.
        """
        print("Pre-building image paths...")
        paths = []
        for row in self.dataframe.itertuples():
            filename = f"image_{row.imageid}_product_{row.productid}.jpg"
            paths.append(self.image_dir / filename)
        print(f"✓ Cached {len(paths)} image paths")
        return paths

    def _construct_image_path(self, imageid: int, productid: int) -> Path:
        """Construct the image file path dynamically (used only for verification)."""
        filename = f"image_{imageid}_product_{productid}.jpg"
        return self.image_dir / filename

    def _verify_images(self, remove_missing: bool = True) -> None:
        """
        Verify that all images exist in the specified directory.

        If `remove_missing` is True, rows with missing images are removed.
        Otherwise a FileNotFoundError is raised.
        """
        print("Verifying image files...")
        missing_indices = []

        for idx, row in self.dataframe.iterrows():
            image_path = self._construct_image_path(row["imageid"], row["productid"])
            if not image_path.exists():
                missing_indices.append(idx)

        if missing_indices:
            n_missing = len(missing_indices)
            n_total = len(self.dataframe)

            if remove_missing:
                self.dataframe = (
                    self.dataframe.drop(missing_indices).reset_index(drop=True)
                )
                warnings.warn(
                    f"Removed {n_missing} / {n_total} samples with missing images.\n"
                    f"Remaining samples: {len(self.dataframe)}"
                )
                print(f"✓ Dataset filtered: {len(self.dataframe)} valid samples")
            else:
                first_row = self.dataframe.iloc[missing_indices[0]]
                first_path = self._construct_image_path(
                    first_row["imageid"], first_row["productid"]
                )
                raise FileNotFoundError(
                    f"Found {n_missing} / {n_total} missing images.\n"
                    f"First missing: {first_path}\n"
                    f"Set remove_missing=True to auto-filter them."
                )
        else:
            print(f"✓ All {len(self.dataframe)} images verified successfully")

    # ------------------------------------------------------------------ #
    # PyTorch Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Use pre-cached path and label (avoids DataFrame access)
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image from disk (on-demand, relies on DataLoader num_workers for parallelism)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found at {image_path}. "
                f"Did you call RakutenImageDataset with verify_images=True?"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        Returns a tensor of shape (num_classes,).
        """
        class_counts = (
            self.dataframe["label_idx"].value_counts().sort_index().values
        )
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        return class_weights

    def get_label_distribution(self) -> pd.DataFrame:
        """
        Get the distribution of raw labels in the dataset.

        Returns a DataFrame with columns: 'label', 'count', 'percentage'.
        """
        distribution = self.dataframe[self.label_col].value_counts().sort_index()
        df_dist = pd.DataFrame(
            {
                "label": distribution.index,
                "count": distribution.values,
                "percentage": (
                    distribution.values / len(self.dataframe) * 100
                ).round(2),
            }
        )
        return df_dist
