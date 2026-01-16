"""
ConvNeXt Canonical Training + Phase 3 Export

Dependencies:
- timm (pip install timm) - Required for ConvNeXt models
- torch, torchvision, numpy, pandas, sklearn, tqdm (standard ML stack)

Local usage:
    python -m src.train.image_convnext --raw-dir data/raw --img-dir data/raw/images/image_train ...

Colab usage:
    python -m src.train.image_convnext --raw-dir /content/drive/.../data_raw --img-dir /content/images/... ...
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score

# FIXED: Colab loader always available, local loader optional
from src.data.data_colab import load_data_colab

try:
    from src.data.data import load_data
    _USE_LOCAL_LOADER = True
except ImportError:
    _USE_LOCAL_LOADER = False

from src.data.split_manager import load_splits, split_signature
from src.data.label_mapping import (
    CANONICAL_CLASSES,
    CANONICAL_CLASSES_FP,
    encode_labels,
    reorder_probs_to_canonical,
)
from src.export.model_exporter import export_predictions, load_predictions

from src.data.image_dataset import RakutenImageDataset
import wandb


@dataclass
class ConvNeXtConfig:
    """
    Configuration for ConvNeXt canonical training pipeline with Phase 3 export.

    This config encapsulates all hyperparameters, paths, and settings needed for
    training ConvNeXt models on Rakuten product classification using canonical splits
    and classes, then exporting predictions in the unified .npz + metadata format.

    Attributes:
        raw_dir: Path to raw CSV directory containing X_train.csv, X_test.csv, etc.
            Used by data loader to build full DataFrame with labels and image IDs.
        img_dir: Path to image directory (e.g., data/images/image_train or
            /content/images for Colab). Must contain .jpg files matching imageid column.
        out_dir: Export output directory for model predictions. Exports are saved as
            {out_dir}/{model_name}/{split_name}.npz with accompanying metadata JSON.
        ckpt_dir: Checkpoint directory for saving model weights during training.
            Best model (by validation F1) is saved as best_model.pth.

        img_size: Input image resolution in pixels (square). 384 is recommended for
            ConvNeXt-Base models pretrained at this resolution for optimal performance.
        batch_size: Training batch size. Reduce if encountering OOM errors. 64 works
            well on 16GB GPU with img_size=384 and AMP enabled.
        num_workers: Number of DataLoader worker processes for parallel data loading.
            Set to 0 for debugging. Recommended: 4-8 for local, 8-12 for Colab.
        num_epochs: Total number of training epochs. 30 recommended for convergence
            with high regularization. Monitor validation curve to avoid underfitting.
        lr: Peak learning rate for AdamW optimizer. 1e-4 is a safe default for
            fine-tuning pretrained ConvNeXt models with cosine annealing schedule.
        weight_decay: L2 weight decay (AdamW). 0.05 is standard for vision transformers
            and ConvNeXt architectures.
        use_amp: Enable Automatic Mixed Precision (torch.cuda.amp) for faster training
            and reduced memory usage. Highly recommended for modern GPUs (Ampere+).

        label_smoothing: Label smoothing factor for CrossEntropyLoss. 0.1 reduces
            overconfidence and improves calibration. Set to 0.0 to disable.
        dropout_rate: Dropout rate for first classification head layer. Set to 0.0
            for ConvNeXt since it relies on drop_path for regularization.
        head_dropout2: Dropout rate for second classification head layer. Set to 0.0
            for ConvNeXt (drop_path is primary regularization mechanism).
        drop_path_rate: Stochastic depth rate applied across ConvNeXt blocks. 0.6 is
            recommended for high regularization to prevent overfitting. This is the
            PRIMARY regularization technique for ConvNeXt architectures.

        mixup_alpha: Mixup augmentation alpha parameter (Beta distribution). 0.8
            recommended for strong augmentation. Set to 0.0 to disable mixup.
        cutmix_alpha: CutMix augmentation alpha parameter. 1.0 recommended. Set to 0.0
            to disable cutmix. Works in conjunction with mixup (randomly selected).
        mixup_prob: Probability of applying mixup/cutmix augmentation to each batch.
            1.0 means always apply. Reduce if augmentation is too aggressive.
        mixup_switch_prob: Probability of choosing mixup vs cutmix when augmentation
            is applied. 0.5 means equal probability. Adjust based on empirical results.

        use_ema: Enable Exponential Moving Average of model weights during training.
            EMA model often has better generalization. Highly recommended for production.
        ema_decay: EMA momentum/decay rate. 0.9999 is standard for vision models.
            Higher values (closer to 1.0) mean slower EMA updates.

        cosine_eta_min: Minimum learning rate for cosine annealing scheduler. 1e-6
            ensures learning rate doesn't decay to zero, maintaining small updates.

        convnext_model_name: timm model identifier for ConvNeXt architecture. Use
            "convnext_base.fb_in22k_ft_in1k_384" for ImageNet-22k pretrained model
            fine-tuned on ImageNet-1k at 384x384 resolution (recommended default).
            See timm docs for other variants (tiny, small, base, large).
        convnext_pretrained: Load pretrained ImageNet weights from timm. Highly
            recommended (True) unless training from scratch for research purposes.

        force_colab_loader: Force use of load_data_colab(raw_dir) instead of local
            load_data() function. Set to True in Colab to handle Google Drive paths.
            Set to False for local development with relative paths.

        device: Training device ("cuda", "cpu", or None for auto-detection). If None,
            automatically selects "cuda" if available, else "cpu".
        model_name: Model identifier for export directory naming. Predictions are
            saved to {out_dir}/{model_name}/. Use descriptive names like
            "convnext_base_384_v2_logits" to distinguish different experiments.

        export_split: Which split to export predictions for after training. "val"
            (validation set) is recommended for model fusion. "test" for final
            submission. Only one split is exported per training run.

    Examples:
        >>> cfg = ConvNeXtConfig(
        ...     raw_dir="data/raw",
        ...     img_dir="data/images/image_train",
        ...     out_dir="artifacts/exports",
        ...     ckpt_dir="checkpoints/convnext",
        ...     model_name="convnext_high_reg_logits",
        ...     num_epochs=30,
        ...     drop_path_rate=0.6,
        ... )
        >>> result = run_convnext_canonical(cfg)

    Note:
        High regularization config (drop_path_rate=0.6, dropout_rate=0.0,
        head_dropout2=0.0) is recommended to prevent overfitting on Rakuten dataset.
        ConvNeXt architecture benefits more from stochastic depth (drop_path) than
        traditional dropout.

    See Also:
        :func:`run_convnext_canonical`: Main training function that uses this config
        :class:`RakutenConvNeXt`: Model architecture that uses these hyperparameters
    """
    # Data / IO
    raw_dir: str                      # Path to raw CSV directory
    img_dir: str                      # Path to image directory
    out_dir: str                      # Export output directory
    ckpt_dir: str                     # Checkpoint directory

    # Training
    img_size: int = 384               # Higher resolution for ConvNeXt
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    use_amp: bool = True

    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.0          # Head dropout (ConvNeXt relies on drop_path)
    head_dropout2: float = 0.0         # Second head dropout (ConvNeXt relies on drop_path)
    drop_path_rate: float = 0.6        # Stochastic depth (primary regularization)

    # Mixup/CutMix
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Scheduler
    cosine_eta_min: float = 1e-6

    # Model
    convnext_model_name: str = "convnext_base.fb_in22k_ft_in1k_384"
    convnext_pretrained: bool = True

    # Data loader
    force_colab_loader: bool = False  # Force Colab loader (ignores local loader)

    # Runtime
    device: Optional[str] = None      # "cuda" or "cpu"
    model_name: str = "convnext"      # Export name

    # Export split
    export_split: str = "val"         # "val" (recommended) or "test"


class IndexedDataset(Dataset):
    """
    Dataset wrapper that selects specific indices while preserving original indices.

    This wrapper enables creating subsets (train/val/test splits) of a base dataset
    while maintaining the original DataFrame row indices. Critical for Phase 3 export
    validation where predictions must be aligned with canonical splits by original
    index rather than sequential position.

    Attributes:
        base: The underlying full dataset (typically RakutenImageDataset with all samples)
        indices: NumPy array of original DataFrame indices to select from base dataset

    Args:
        base_dataset: Full PyTorch Dataset that supports integer indexing. Must return
            (image, label) tuples when indexed.
        indices: Array or list of integer indices specifying which samples from
            base_dataset to include in this subset. Indices refer to positions in the
            original DataFrame before any splitting.

    Returns:
        When accessed via __getitem__, returns (image, label, original_idx) tuple where
        original_idx is the sample's position in the full DataFrame.

    Examples:
        >>> full_dataset = RakutenImageDataset(full_df, img_dir, transform)
        >>> train_indices = np.array([0, 2, 5, 8])  # Select specific samples
        >>> train_dataset = IndexedDataset(full_dataset, train_indices)
        >>> img, label, orig_idx = train_dataset[0]  # orig_idx will be 0
        >>> img, label, orig_idx = train_dataset[1]  # orig_idx will be 2

    Note:
        The returned original_idx (third element of tuple) is essential for Phase 3
        export validation. It allows verification that predictions align with canonical
        splits using split signatures, preventing subtle data leakage bugs.
    """
    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices).astype(int)

    def __len__(self) -> int:
        """Return the number of samples in this indexed subset."""
        return len(self.indices)

    def __getitem__(self, i: int):
        """
        Get item by subset position, returning (image, label, original_index).

        Args:
            i: Position in this subset (0 to len(self)-1)

        Returns:
            Tuple of (image, label, original_idx) where original_idx is the sample's
            position in the original full DataFrame (before splitting)
        """
        real_idx = int(self.indices[i])
        img, label = self.base[real_idx]
        return img, label, real_idx


class RakutenConvNeXt(nn.Module):
    """
    ConvNeXt model for Rakuten product classification with custom regularized head.

    Wraps a timm ConvNeXt backbone (pretrained on ImageNet) with a custom two-layer
    classification head featuring LayerNorm, dropout, and GELU activation. Supports
    stochastic depth (drop_path) in the backbone for regularization.

    The model architecture:
    1. ConvNeXt backbone (from timm) with global average pooling
    2. LayerNorm + Dropout
    3. Linear(feature_dim -> 512) + GELU
    4. Dropout
    5. Linear(512 -> num_classes)

    Attributes:
        backbone: timm ConvNeXt model without classification head (num_classes=0)
        head: Custom classification head (Sequential with LayerNorm, Linear, GELU, Dropout)
        num_classes: Number of output classes (27 for canonical Rakuten)
        model_name: timm model identifier (e.g., "convnext_base.fb_in22k_ft_in1k_384")

    Args:
        model_name: timm model identifier for ConvNeXt variant. Common options:
            - "convnext_base" (88M params, ImageNet-1k)
            - "convnext_base.fb_in22k_ft_in1k_384" (88M params, ImageNet-22k pretrained)
            - "convnext_tiny", "convnext_small", "convnext_large" (other sizes)
        num_classes: Number of output classes for classification. Default 27 for
            canonical Rakuten product taxonomy.
        pretrained: Load ImageNet pretrained weights from timm. Highly recommended
            (True) for transfer learning. Set False only for training from scratch.
        drop_path_rate: Stochastic depth rate applied across ConvNeXt blocks. Higher
            values (0.6) provide stronger regularization. This is the PRIMARY
            regularization for ConvNeXt. Default 0.3.
        dropout_rate: Dropout rate for first head layer (after LayerNorm). Set to 0.0
            for high regularization configs that rely on drop_path. Default 0.5.
        head_dropout2: Dropout rate for second head layer (after first Linear+GELU).
            Set to 0.0 for high regularization configs. Default 0.3.

    Examples:
        >>> # Standard configuration
        >>> model = RakutenConvNeXt(
        ...     model_name="convnext_base.fb_in22k_ft_in1k_384",
        ...     num_classes=27,
        ...     pretrained=True,
        ...     drop_path_rate=0.6,
        ...     dropout_rate=0.0,
        ...     head_dropout2=0.0,
        ... )
        >>> images = torch.randn(4, 3, 384, 384)
        >>> logits = model(images)  # Shape: (4, 27)

    Note:
        High regularization config (drop_path_rate=0.6, dropout_rate=0.0,
        head_dropout2=0.0) is recommended to prevent overfitting on Rakuten dataset.
        ConvNeXt benefits more from stochastic depth than traditional dropout.

    Raises:
        ImportError: If timm library is not installed (required for ConvNeXt models)

    See Also:
        :class:`ConvNeXtConfig`: Configuration dataclass with hyperparameters
        :func:`_build_convnext`: Helper function to construct this model from config
    """

    def __init__(
        self,
        model_name: str = "convnext_base",
        num_classes: int = 27,
        pretrained: bool = True,
        drop_path_rate: float = 0.3,
        dropout_rate: float = 0.5,
        head_dropout2: float = 0.3,
    ):
        super(RakutenConvNeXt, self).__init__()

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for ConvNeXt models.\n"
                "Install with: pip install timm\n"
                "Colab: !pip install timm\n"
                "Windows/Linux: pip install timm"
            )

        # ConvNeXt backbone without classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        feature_dim = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=head_dropout2),
            nn.Linear(512, num_classes),
        )

        self.num_classes = num_classes
        self.model_name = model_name

    def forward(self, x):
        """
        Forward pass through ConvNeXt backbone and classification head.

        Args:
            x: Input image tensor of shape (B, 3, H, W) where B is batch size,
                H and W are height and width (typically 384x384 for ConvNeXt-Base)

        Returns:
            Logits tensor of shape (B, num_classes) with raw class scores (pre-softmax)
        """
        features = self.backbone(x)
        return self.head(features)


def _build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build training and validation image transforms for ConvNeXt models.

    Creates torchvision transform pipelines optimized for ConvNeXt's higher resolution
    inputs (typically 384x384). Training transforms include aggressive augmentation
    (RandomResizedCrop, HorizontalFlip, RandAugment) while validation uses simple
    Resize+CenterCrop for deterministic evaluation.

    Args:
        img_size: Target image size in pixels (square). Recommended: 384 for ConvNeXt-Base
            models pretrained at 384x384 resolution.

    Returns:
        Tuple of (train_transform, val_transform) where each is a torchvision.transforms.Compose:
            - train_transform: Augmentation pipeline for training with RandomResizedCrop,
              RandomHorizontalFlip, RandAugment, ToTensor, and ImageNet normalization
            - val_transform: Deterministic pipeline for validation/test with Resize(1.14x),
              CenterCrop, ToTensor, and ImageNet normalization

    Note:
        Validation transform uses 1.14x oversized Resize before CenterCrop (e.g., 438->384)
        following ConvNeXt paper's evaluation protocol for better accuracy.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Higher base size for center crop (438 -> 384)
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # 438 for 384
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def _build_convnext(
    num_classes: int,
    cfg: ConvNeXtConfig,
) -> RakutenConvNeXt:
    """
    Build ConvNeXt with custom head for canonical classes.

    Args:
        num_classes: Number of output classes (27 for canonical)
        cfg: ConvNeXtConfig with model parameters

    Returns:
        RakutenConvNeXt model

    Raises:
        ImportError: If timm is not installed (pip install timm)
    """
    model = RakutenConvNeXt(
        model_name=cfg.convnext_model_name,
        num_classes=int(num_classes),
        pretrained=cfg.convnext_pretrained,
        drop_path_rate=cfg.drop_path_rate,
        dropout_rate=cfg.dropout_rate,
        head_dropout2=cfg.head_dropout2,
    )
    return model


def _make_loaders(
    df_full: pd.DataFrame,
    y_encoded: np.ndarray,
    splits: Dict[str, np.ndarray],
    img_dir: Path,
    cfg: ConvNeXtConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset]:
    """
    Build train/val/test loaders with canonical split indices and canonical label ids.
    Returns (train_loader, val_loader, test_loader, full_dataset) where full_dataset
    can be reused for export to maintain idx alignment.

    FIXED: Use full_dataset + IndexedDataset to preserve global idx semantics.
    """
    df_full = df_full.copy()
    df_full["encoded_label"] = y_encoded.astype(int)

    train_tf, val_tf = _build_transforms(cfg.img_size)

    # Build FULL dataset for training (will be wrapped with IndexedDataset)
    full_dataset_train = RakutenImageDataset(
        dataframe=df_full.reset_index(drop=True),
        image_dir=str(img_dir),
        transform=train_tf,
        label_col="encoded_label",
    )

    # Build FULL dataset for val/test (deterministic transform)
    full_dataset_val = RakutenImageDataset(
        dataframe=df_full.reset_index(drop=True),
        image_dir=str(img_dir),
        transform=val_tf,
        label_col="encoded_label",
    )

    pin_memory = bool((cfg.device or "").startswith("cuda") or torch.cuda.is_available())

    # FIXED: Use IndexedDataset to wrap full_dataset with split indices
    train_indexed = IndexedDataset(full_dataset_train, splits["train_idx"])
    val_indexed = IndexedDataset(full_dataset_val, splits["val_idx"])
    test_indexed = IndexedDataset(full_dataset_val, splits["test_idx"])

    train_loader = DataLoader(
        train_indexed,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_indexed,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_indexed,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, full_dataset_val


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool,
    scaler: Optional[torch.amp.GradScaler],
    mixup_fn: Optional[Any],
    model_ema: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """
    Train the model for one epoch with mixup/cutmix augmentation and optional EMA.

    Performs a full training pass over the dataset with gradient accumulation,
    mixed precision training, data augmentation (mixup/cutmix), and optional
    exponential moving average (EMA) model weight updates.

    Args:
        model: ConvNeXt model to train (nn.Module in training mode)
        loader: Training DataLoader providing (images, labels) batches
        criterion: Loss function (typically CrossEntropyLoss with label smoothing)
        optimizer: AdamW optimizer for parameter updates
        device: Device string ("cuda" or "cpu") for tensor placement
        use_amp: Enable automatic mixed precision training with torch.amp
        scaler: Gradient scaler for mixed precision (required if use_amp=True)
        mixup_fn: Mixup/CutMix augmentation function (timm.data.Mixup instance).
            If None, no augmentation is applied. Otherwise, applies random mixup
            or cutmix to each batch based on configured probabilities.
        model_ema: Optional EMA model wrapper (timm.utils.ModelEmaV2). If provided,
            EMA weights are updated after each batch with configured decay rate.

    Returns:
        Tuple of (average_loss, accuracy, f1_score) for the epoch:
            - average_loss: Mean cross-entropy loss across all batches
            - accuracy: Classification accuracy on training set
            - f1_score: Weighted F1 score on training set

    Note:
        Training metrics (accuracy, F1) are computed on augmented data with
        mixup/cutmix applied, so they may be lower than validation metrics.
        The primary training objective is loss minimization.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # FIXED: device_type for AMP autocast
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    pbar = tqdm(loader, desc="Train", ncols=100, leave=False)
    for batch in pbar:
        # FIXED: Handle IndexedDataset return (img, label, real_idx)
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply Mixup/CutMix
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            assert scaler is not None
            # FIXED: device_type dynamic
            with torch.autocast(device_type=device_type, enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Update EMA if enabled
        if model_ema is not None:
            model_ema.update(model)

        total_loss += float(loss.item()) * images.size(0)

        # For metrics, we skip when using mixup (soft labels)
        if mixup_fn is None:
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labs)

        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) > 0 else 0.0
    return float(avg_loss), float(acc), float(f1)


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    """
    Evaluate the model for one epoch on validation/test data.

    Performs inference on the validation or test dataset without augmentation,
    computing loss and classification metrics. Model is set to eval mode with
    no gradient computation for efficiency.

    Args:
        model: ConvNeXt model to evaluate (nn.Module in eval mode)
        loader: Validation or test DataLoader providing (images, labels) batches
        criterion: Loss function (typically CrossEntropyLoss) for computing validation loss
        device: Device string ("cuda" or "cpu") for tensor placement
        use_amp: Enable automatic mixed precision for inference. Default False since
            memory savings are less critical during evaluation, but can be enabled
            for consistency with training or to speed up large validation sets.

    Returns:
        Tuple of (average_loss, accuracy, f1_score) for the evaluation:
            - average_loss: Mean cross-entropy loss across all batches
            - accuracy: Classification accuracy on the evaluation set
            - f1_score: Weighted F1 score on the evaluation set (primary metric)

    Note:
        No data augmentation (mixup/cutmix) is applied during evaluation, so
        metrics reflect true model performance on clean data. F1 score is the
        primary metric used for model selection and early stopping.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    pbar = tqdm(loader, desc="Val", ncols=100, leave=False)
    for batch in pbar:
        # FIXED: Handle IndexedDataset return
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type=device_type, enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labs)

        pbar.set_postfix(loss=float(loss.item()))

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) else 0.0
    return float(avg_loss), float(acc), float(f1)


@torch.no_grad()
def _predict_logits_with_real_idx(
    model: nn.Module,
    base_dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict raw logits for specified samples while preserving original DataFrame indices.

    Performs inference on a subset of samples (e.g., validation or test split) specified
    by their original DataFrame row indices. Returns raw model logits WITHOUT softmax,
    which is essential for Phase 4 model fusion and calibration. The returned indices
    match the input indices order exactly, enabling Phase 3 export validation.

    Args:
        model: Trained ConvNeXt model in eval mode (nn.Module)
        base_dataset: Full RakutenImageDataset containing all samples
        indices: NumPy array of original DataFrame row indices for samples to predict.
            Order is preserved in output.
        batch_size: Batch size for inference DataLoader
        num_workers: Number of DataLoader worker processes
        device: Device string ("cuda" or "cpu") for inference

    Returns:
        Tuple of (logits, idx) where:
            - logits: NumPy array of shape (N, 27) with raw model outputs (pre-softmax)
            - idx: NumPy array of shape (N,) with original DataFrame indices, matching
              the order of input indices parameter

    Note:
        - Returns RAW LOGITS (not probabilities) for proper model fusion
        - No softmax applied - downstream code must apply if probabilities needed
        - Uses shuffle=False to preserve index order for export validation
        - Critical for Phase 3 export contract with split signature verification
    """
    model.eval()

    indexed = IndexedDataset(base_dataset, indices)
    loader = DataLoader(
        indexed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    logits_list = []
    idx_list = []

    for images, _, real_idx in tqdm(loader, desc="ExportInference", ncols=100, leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        # Export raw logits (no softmax) for model fusion
        logits_np = logits.detach().cpu().numpy()
        logits_list.append(logits_np)
        idx_list.append(real_idx.detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, len(CANONICAL_CLASSES)), dtype=np.float32)
    idx = np.concatenate(idx_list, axis=0) if idx_list else np.zeros((0,), dtype=int)
    return logits, idx


def run_convnext_canonical(cfg: ConvNeXtConfig) -> Dict[str, Any]:
    """
    Canonical ConvNeXt training pipeline with Phase 3 export contract and validation.

    Implements the complete training workflow for ConvNeXt models on Rakuten product
    classification with canonical splits (cf53f8eb169b3531) and classes (cdfa70b13f7390e6).
    Includes advanced training techniques: Mixup/CutMix augmentation, AdamW optimizer,
    cosine annealing learning rate schedule, automatic mixed precision, and optional
    exponential moving average (EMA) of model weights.

    After training, exports predictions in the Phase 3 unified format (.npz + metadata
    JSON) and validates the export against canonical splits and classes for downstream
    model fusion compatibility.

    Args:
        cfg: ConvNeXtConfig instance containing all training hyperparameters, paths,
            and export settings. See ConvNeXtConfig docstring for detailed field descriptions.

    Returns:
        Dictionary with training results and export validation:
            - export_result: Dict from export_predictions() containing:
                - npz_path: Path to exported predictions (.npz file)
                - meta_json_path: Path to metadata JSON file
                - classes_fp: Classes fingerprint (must be cdfa70b13f7390e6)
                - split_signature: Split signature (must be cf53f8eb169b3531)
                - num_samples: Number of samples exported
            - verify_metadata: Loaded metadata dict from exported files for verification
            - logits_shape: Shape tuple of exported logits (N, 27) or None if not exported
            - probs_shape: Shape tuple of exported probabilities (N, 27) or None if not exported
            - best_val_f1: Best validation F1 score achieved during training (float)
            - history: List of per-epoch metrics dicts with keys:
                - epoch: Epoch number
                - train_loss, train_acc, train_f1: Training metrics
                - val_loss, val_acc, val_f1: Validation metrics
                - lr: Learning rate at end of epoch

    Raises:
        AssertionError: If export validation fails (split signature mismatch,
            classes fingerprint mismatch, or missing expected files)
        FileNotFoundError: If required data files (splits, CSVs, images) not found
        RuntimeError: If CUDA out of memory or other training failures occur

    Examples:
        >>> cfg = ConvNeXtConfig(
        ...     raw_dir="data/raw",
        ...     img_dir="data/images/image_train",
        ...     out_dir="artifacts/exports",
        ...     ckpt_dir="checkpoints/convnext",
        ...     model_name="convnext_v2_logits",
        ... )
        >>> result = run_convnext_canonical(cfg)
        >>> print(f"Best F1: {result['best_val_f1']:.4f}")
        >>> print(f"Export: {result['export_result']['npz_path']}")

    Note:
        - Training is logged to WandB project "rakuten_image" with run name from cfg.model_name
        - Best model is saved to {cfg.ckpt_dir}/best_model.pth based on validation F1
        - Only the specified export_split (val or test) is exported after training
        - Export validation ensures compatibility with model fusion pipelines (Phase 4)

    See Also:
        :class:`ConvNeXtConfig`: Configuration dataclass with all hyperparameters
        :func:`export_predictions`: Export function with detailed contract specification
        :func:`load_predictions`: Load function for verifying exported predictions
    """
    wandb.init(
        project="rakuten_image",
        name=cfg.model_name,
        config=cfg,
        reinit=True,
    )

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    ckpt_dir = Path(cfg.ckpt_dir).expanduser().resolve()
    img_dir = Path(cfg.img_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load full data (NO split generation here)
    # FIXED: Force Colab loader when flag is set
    if cfg.force_colab_loader:
        print("[INFO] Using Colab data loader (forced via force_colab_loader=True)")
        pack = load_data_colab(
            raw_dir=cfg.raw_dir,
            img_root=Path(cfg.img_dir),
            splitted=False,
            verbose=True,
        )
        X, y = pack["X"], pack["y"]
    elif _USE_LOCAL_LOADER:
        print("[INFO] Using local data loader (src.data.data.load_data)")
        pack = load_data(splitted=False)
        X, y = pack["X"], pack["y"]
    else:
        print("[INFO] Using Colab data loader (src.data.data_colab.load_data_colab)")
        pack = load_data_colab(
            raw_dir=cfg.raw_dir,
            img_root=Path(cfg.img_dir),
            splitted=False,
            verbose=True,
        )
        X, y = pack["X"], pack["y"]

    # 2) Canonical splits (single source of truth)
    splits = load_splits(verbose=True)
    sig = split_signature(splits)

    # 3) Canonical label encoding (training IDs = canonical indices)
    y_encoded = encode_labels(y, CANONICAL_CLASSES).astype(int)

    # 4) DataLoaders (FIXED: returns full_dataset for export reuse)
    train_loader, val_loader, _, full_dataset_val = _make_loaders(
        df_full=X,
        y_encoded=y_encoded,
        splits=splits,
        img_dir=img_dir,
        cfg=cfg,
    )

    # 5) Model
    model = _build_convnext(
        num_classes=len(CANONICAL_CLASSES),
        cfg=cfg,
    ).to(device)

    wandb.watch(model, log="all", log_freq=100)

    # 6) EMA (Exponential Moving Average)
    model_ema = None
    if cfg.use_ema:
        try:
            from timm.utils import ModelEmaV2
            model_ema = ModelEmaV2(model, decay=cfg.ema_decay)
            print(f"[INFO] EMA initialized with decay={cfg.ema_decay}")
        except ImportError:
            print("[WARNING] timm.utils.ModelEmaV2 not available, EMA disabled")
            cfg.use_ema = False

    # 7) Mixup/CutMix
    try:
        from timm.data.mixup import Mixup
        from timm.loss import SoftTargetCrossEntropy

        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            switch_prob=cfg.mixup_switch_prob,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=len(CANONICAL_CLASSES),
        )
        criterion_train = SoftTargetCrossEntropy()
    except ImportError:
        raise ImportError(
            "timm.data.mixup and timm.loss are required for ConvNeXt training.\n"
            "Install with: pip install timm\n"
            "Colab: !pip install timm\n"
            "Windows/Linux: pip install timm"
        )

    criterion_val = nn.CrossEntropyLoss()

    # 8) Optimization (AdamW + CosineAnnealingLR)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=cfg.cosine_eta_min,
    )

    # AMP (safe enabling)
    use_amp = bool(cfg.use_amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val_f1 = -1.0
    best_path = ckpt_dir / "best_model.pth"
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "ema_val_acc": [],
        "ema_val_f1": [],
        "lr": [],
    }

    for epoch in range(int(cfg.num_epochs)):
        train_loss, train_acc, train_f1 = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion_train,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            mixup_fn=mixup_fn,
            model_ema=model_ema,
        )

        val_loss, val_acc, val_f1 = _eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion_val,
            device=device,
            use_amp=use_amp,
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": float(optimizer.param_groups[0]["lr"])
        })

        # Evaluate EMA model if enabled
        ema_val_acc, ema_val_f1 = 0.0, 0.0
        if model_ema is not None:
            _, ema_val_acc, ema_val_f1 = _eval_one_epoch(
                model=model_ema.module,
                loader=val_loader,
                criterion=criterion_val,
                device=device,
                use_amp=use_amp,
            )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["ema_val_acc"].append(ema_val_acc)
        history["ema_val_f1"].append(ema_val_f1)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        if model_ema is not None:
            print(f"  EMA: val_acc={ema_val_acc:.4f} val_f1={ema_val_f1:.4f}")

        # Save best model (prefer EMA if better)
        current_f1 = max(val_f1, ema_val_f1)
        if current_f1 > best_val_f1:
            best_val_f1 = float(current_f1)
            use_ema_for_export = (ema_val_f1 > val_f1) and (model_ema is not None)
            torch.save(
                {
                    "model_state_dict": model_ema.module.state_dict() if use_ema_for_export else model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_f1": best_val_f1,
                    "split_signature": sig,
                    "classes_fp": CANONICAL_CLASSES_FP,
                    "is_ema": use_ema_for_export,
                },
                best_path,
            )

    # Load best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # 9) Export raw logits (alignment-safe) on cfg.export_split
    export_idx = splits["val_idx"] if cfg.export_split == "val" else splits["test_idx"]

    # Export raw logits WITHOUT softmax for model fusion and calibration
    logits, seen_idx = _predict_logits_with_real_idx(
        model=model,
        base_dataset=full_dataset_val,
        indices=export_idx,
        batch_size=cfg.batch_size,
        num_workers=0,
        device=device,
    )

    # Hard alignment check
    if not np.array_equal(seen_idx, export_idx):
        raise AssertionError("Index order mismatch during export inference")

    # Explicit no-op reorder for traceability (logits have same class order as probs)
    logits_aligned = reorder_probs_to_canonical(logits, CANONICAL_CLASSES, CANONICAL_CLASSES)

    # FIXED: y_true must be canonical indices (0..26), not original labels
    y_true = y_encoded[seen_idx].astype(int)

    export_result = export_predictions(
        out_dir=out_dir,
        model_name=cfg.model_name,
        split_name=cfg.export_split,
        idx=seen_idx,
        split_signature=sig,
        logits=logits_aligned,  # Export raw logits for model fusion
        probs=None,  # Not exporting probabilities
        classes=CANONICAL_CLASSES,
        y_true=y_true,
        extra_meta={
            "source": "src/train/image_convnext.py",
            "model_architecture": f"timm.{cfg.convnext_model_name}",  # Display only, not for instantiation
            "timm_model_name": cfg.convnext_model_name,  # For model instantiation: timm.create_model(this_value)
            "convnext_pretrained": cfg.convnext_pretrained,
            "img_dir": str(img_dir),
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "use_amp": use_amp,
            "label_smoothing": cfg.label_smoothing,
            "drop_path_rate": cfg.drop_path_rate,
            "dropout_rate": cfg.dropout_rate,
            "mixup_alpha": cfg.mixup_alpha,
            "cutmix_alpha": cfg.cutmix_alpha,
            "use_ema": cfg.use_ema,
            "ema_decay": cfg.ema_decay,
            "classes_fp": CANONICAL_CLASSES_FP,
            "split_signature": sig,
            "export_split": cfg.export_split,
        },
    )

    # 10) Verify export contract (B4-compatible strict checks)
    loaded = load_predictions(
        npz_path=export_result["npz_path"],
        verify_split_signature=sig,
        verify_classes_fp=CANONICAL_CLASSES_FP,
        require_y_true=True,
    )

    wandb.finish()

    return {
        "export_result": export_result,
        "verify_metadata": loaded["metadata"],
        "logits_shape": loaded.get("logits").shape if "logits" in loaded else None,
        "probs_shape": loaded.get("probs").shape if "probs" in loaded else None,
        "best_val_f1": float(best_val_f1),
        "history": history,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConvNeXt Canonical Training + Export")
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw CSV directory")
    parser.add_argument("--img-dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--out-dir", type=str, default="artifacts/exports", help="Export output directory")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/image_convnext", help="Checkpoint directory")

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")

    # Model selection and pretrained control
    parser.add_argument("--model", type=str, default="convnext_base.fb_in22k_ft_in1k_384",
                        help="timm model name (default: convnext_base.fb_in22k_ft_in1k_384)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (default: True)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Do not use pretrained weights")

    # Data loader selection
    parser.add_argument("--force-colab-loader", dest="force_colab_loader", action="store_true",
                        help="Force load_data_colab(raw_dir=...) (recommended in Colab)")
    parser.add_argument("--no-force-colab-loader", dest="force_colab_loader", action="store_false",
                        help="Do not force Colab loader (use local loader if available)")
    parser.set_defaults(force_colab_loader=False)

    parser.add_argument("--export-name", type=str, default="convnext", help="Model name for export")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Export split")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if None)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ConvNeXtConfig(
        raw_dir=args.raw_dir,
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        img_size=384,  # Higher resolution for ConvNeXt
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=True,
        label_smoothing=0.1,
        dropout_rate=0.0,
        head_dropout2=0.0,
        drop_path_rate=0.6,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        use_ema=True,
        ema_decay=0.9999,
        convnext_model_name=args.model,
        convnext_pretrained=args.pretrained,
        force_colab_loader=args.force_colab_loader,
        device=args.device,
        model_name=args.export_name,
        export_split=args.split,
    )

    print("="*80)
    print("ConvNeXt Canonical Training Configuration")
    print("="*80)
    print(f"Raw dir: {cfg.raw_dir}")
    print(f"Image dir: {cfg.img_dir}")
    print(f"Export dir: {cfg.out_dir}")
    print(f"Checkpoint dir: {cfg.ckpt_dir}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Model name: {cfg.model_name}")
    print(f"Export split: {cfg.export_split}")
    print(f"EMA: {cfg.use_ema}")
    print("="*80)

    result = run_convnext_canonical(cfg)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best val F1: {result['best_val_f1']:.4f}")
    print(f"Logits shape: {result.get('logits_shape', 'N/A')}")
    print(f"Probs shape: {result.get('probs_shape', 'N/A')}")
    print("\nExport Result:")
    for k, v in result["export_result"].items():
        print(f"  {k}: {v}")
    print("="*80)
