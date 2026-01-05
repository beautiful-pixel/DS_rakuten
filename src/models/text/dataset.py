import torch
from transformers import PreTrainedTokenizer
from ..base.dataset import BaseDataset


# Si petite RAM ne pas garder l'encoding entier mémoire mais compute à chaque fois
# => opération dans d''encodage dans __getitem__
class TransformersDataset(BaseDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, X, y, extra_features=None):
        self.encodings = tokenizer(
            list(X),
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"   # retourne directement un pytorch tensor
        )
        self.labels = torch.tensor(y, dtype=torch.long)
        self.extra_features = (
            torch.tensor(extra_features, dtype=torch.float)
            if extra_features is not None
            else None
        )


    def __getitem__(self, idx):
        # on parcourt les clés du dictionnaire self.encoding
        # et on récupère les input_ids, token_type_ids et attention_mask de l'index
        item = {k: v[idx] for k, v in self.encodings.items()}
        # car le model attent le paramètre labels pour l'entrainement
        item["labels"] = self.labels[idx]
        if self.extra_features is not None:
            item["extra_features"] = self.extra_features[idx]
        return item

    def __len__(self):
        return len(self.labels)
