import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from .metrics import compute_classification_metrics


import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from .metrics import compute_classification_metrics


class Trainer:
    """
    Trainer PyTorch générique pour modèles de classification
    (compatible avec l'API Hugging Face).

    Cette classe fournit une boucle d'entraînement complète incluant :
    - Support CPU / GPU
    - Entraînement en précision mixte (AMP)
    - Gradient clipping
    - Calcul de métriques de classification
    - Logging TensorBoard
    - Logging optionnel avec Weights & Biases (W&B)
    - Gestion des learning rate schedulers (par step ou par epoch)
    - Sauvegarde des meilleurs checkpoints sur le score de validation

    Le modèle passé au Trainer doit respecter les conventions Hugging Face :
    - le `forward` accepte l'argument `labels`
    - la sortie contient les attributs `loss` et `logits`
    """

    def __init__(
        self,
        model,
        optimizer,
        device=None,
        max_grad_norm=1.0,
        log_dir="runs/experiment",
        checkpoint_dir=None,
        scheduler=None,
        scheduler_type="epoch"
    ):
        """
        Initialise le Trainer.

        Args:
            model (torch.nn.Module):
                Modèle PyTorch compatible Hugging Face.
            optimizer (torch.optim.Optimizer):
                Optimiseur utilisé pour l'entraînement.
            device (str, optional):
                Device de calcul (`"cuda"` ou `"cpu"`).
                Si None, sélection automatique.
            max_grad_norm (float, optional):
                Valeur maximale pour le gradient clipping.
            log_dir (str, optional):
                Répertoire des logs TensorBoard.
            checkpoint_dir (str or None, optional):
                Répertoire de sauvegarde des checkpoints.
            scheduler (torch.optim.lr_scheduler._LRScheduler or None, optional):
                Scheduler de learning rate.
            scheduler_type (str, optional):
                Fréquence d'appel du scheduler :
                - `"step"` : à chaque batch
                - `"epoch"` : à la fin de chaque epoch
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.use_amp = self.device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = max_grad_norm

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.best_val_score = None

        self.model.to(self.device)

    def _step(self, batch, train=True):
        """
        Effectue un step d'entraînement ou d'évaluation sur un batch.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch contenant les entrées du modèle et les labels.
            train (bool, optional):
                Indique si le step est en mode entraînement.

        Returns:
            tuple:
                - loss (float): Valeur de la loss.
                - preds (torch.Tensor): Prédictions du modèle.
                - labels (torch.Tensor): Labels réels.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with autocast(self.device, enabled=self.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler and self.scheduler_type == "step":
                self.scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch["labels"]

        return loss.item(), preds.cpu(), labels.cpu()

    def train_epoch(self, dataloader, epoch):
        """
        Entraîne le modèle sur une epoch complète.

        Args:
            dataloader (torch.utils.data.DataLoader):
                Dataloader d'entraînement.
            epoch (int):
                Index de l'epoch.

        Returns:
            tuple:
                - metrics (dict): Métriques de classification.
                - train_loss (float): Loss moyenne.
        """
        self.model.train()
        losses, y_true, y_pred = [], [], []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

        running_loss = 0.0
        for i, batch in enumerate(progress_bar):
            loss, preds, labels = self._step(batch, train=True)
            losses.append(loss)
            running_loss += loss

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

            progress_bar.set_postfix(
                {"train_loss": f"{running_loss / (i + 1):.4f}"}
            )

        metrics = compute_classification_metrics(y_true, y_pred)
        train_loss = running_loss / len(losses)

        if self.writer:
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/accuracy", metrics["accuracy"], epoch)
            self.writer.add_scalar("train/f1", metrics["f1_weighted"], epoch)

        return metrics, train_loss

    @torch.no_grad()
    def eval_epoch(self, dataloader, epoch):
        """
        Évalue le modèle sur le jeu de validation.

        Args:
            dataloader (torch.utils.data.DataLoader):
                Dataloader de validation.
            epoch (int):
                Index de l'epoch.

        Returns:
            tuple:
                - metrics (dict): Métriques de classification.
                - val_loss (float): Loss moyenne.
        """
        self.model.eval()
        losses, y_true, y_pred = [], [], []

        for batch in tqdm(dataloader, desc="Validation"):
            loss, preds, labels = self._step(batch, train=False)
            losses.append(loss)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        val_loss = sum(losses) / len(losses)

        if self.writer:
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
            self.writer.add_scalar("val/f1", metrics["f1_weighted"], epoch)

        if self.scheduler and self.scheduler_type == "epoch":
            self.scheduler.step(val_loss)

        if self.checkpoint_dir:
            score = metrics["f1_weighted"]
            if self.best_val_score is None or score > self.best_val_score:
                self.best_val_score = score
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, "model.pt"),
                )
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(self.checkpoint_dir, "optimizer.pt"),
                )

        return metrics, val_loss
