"""
Utility functions for reproducibility, checkpointing, and early stopping.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, random, and PyTorch.

    Args:
        seed (int): Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Make cudnn deterministic (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✓ Random seed set to {seed} for reproducibility")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    filepath: str,
    scaler: torch.cuda.amp.GradScaler = None
):
    """
    Save model checkpoint with training state.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        val_loss: Validation loss
        val_acc: Validation accuracy
        filepath: Path to save checkpoint
        scaler: AMP gradient scaler (optional)
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }

    # Save scaler state if using mixed precision
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath} (Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
    device: str = 'cuda'
):
    """
    Load model checkpoint and optionally restore training state.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optimizer to restore state (optional)
        scaler: AMP gradient scaler to restore state (optional)
        device: Device to load model to

    Returns:
        dict: Checkpoint information (epoch, val_loss, val_acc)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scaler state if provided
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    val_acc = checkpoint.get('val_acc', 0.0)

    print(f"✓ Checkpoint loaded: {filepath} (Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")

    return {
        'epoch': epoch,
        'val_loss': val_loss,
        'val_acc': val_acc
    }


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Example:
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        for epoch in range(epochs):
            val_loss = validate(...)
            early_stopping(val_loss)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = float('inf')
        else:  # mode == 'max'
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = float('-inf')

    def __call__(self, score: float, epoch: int = None):
        """
        Check if training should stop based on the current score.

        Args:
            score: Current validation metric (loss or accuracy)
            epoch: Current epoch number (optional, for logging)
        """
        if self.is_better(score, self.best_score):
            # Improvement detected
            self.best_score = score
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠ Early stopping triggered after {self.counter} epochs without improvement")
                print(f"  Best score: {self.best_score:.4f} at epoch {self.best_epoch}")

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
        self.best_epoch = 0
