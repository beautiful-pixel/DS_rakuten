import torch
from transformers import PreTrainedTokenizer
from ..base.dataset import BaseDataset


class TextDataset(BaseDataset):
    """
    Dataset texte pour modèles Transformers.

    Args:
        texts (list[str]): Textes d'entrée.
        labels (list[int] | None): Labels.
        tokenizer (PreTrainedTokenizer): Tokenizer HF.
        max_length (int): Longueur max.
    """

    def __init__(self, texts, labels, tokenizer: PreTrainedTokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
