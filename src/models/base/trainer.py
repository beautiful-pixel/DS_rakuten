import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import compute_classification_metrics


class Trainer:
    """
    Trainer PyTorch générique avec :
    - GPU / CPU
    - AMP (mixed precision)
    - Metrics
    - TensorBoard
    - Checkpointing
    - Early stopping
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion=None,
        device=None,
        use_amp=True,
        log_dir="runs/experiment",
        checkpoint_path=None,
        early_stopping=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.checkpoint_path = checkpoint_path
        self.early_stopping = early_stopping
        self.best_val_score = None

        self.model.to(self.device)

    def _step(self, batch, train=True):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with autocast(enabled=self.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else self.criterion(
                outputs, batch["labels"]
            )

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        return loss.item(), preds.cpu(), batch["labels"].cpu()

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        losses, y_true, y_pred = [], [], []

        for batch in tqdm(dataloader, desc="Training"):
            loss, preds, labels = self._step(batch, train=True)
            losses.append(loss)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        self.writer.add_scalar("train/loss", sum(losses) / len(losses), epoch)
        self.writer.add_scalar("train/accuracy", metrics["accuracy"], epoch)
        self.writer.add_scalar("train/f1", metrics["f1_macro"], epoch)

        return metrics

    @torch.no_grad()
    def eval_epoch(self, dataloader, epoch):
        self.model.eval()
        losses, y_true, y_pred = [], [], []

        for batch in tqdm(dataloader, desc="Validation"):
            loss, preds, labels = self._step(batch, train=False)
            losses.append(loss)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        val_loss = sum(losses) / len(losses)

        self.writer.add_scalar("val/loss", val_loss, epoch)
        self.writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
        self.writer.add_scalar("val/f1", metrics["f1_macro"], epoch)

        # Checkpoint
        if self.checkpoint_path:
            score = metrics["f1_macro"]
            if self.best_val_score is None or score > self.best_val_score:
                self.best_val_score = score
                torch.save(self.model.state_dict(), self.checkpoint_path)

        return metrics
