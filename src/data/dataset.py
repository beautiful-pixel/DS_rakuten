from torch.utils.data import Dataset
import torch, torchvision


class TextDataset(Dataset):

    def __init__(self, tokenizer, X, y):
        self.encodings = tokenizer(
            list(X['text']),
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"   # retourne directement un pytorch tensor
        )
        self.extra_features = torch.tensor(
            X.drop('text', axis=1).to_numpy(),
            dtype=torch.float32
        )
        self.labels = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        # on parcourt les clés du dictionnaire self.encoding
        # et on récupère les input_ids, token_type_ids et attention_mask de l'index
        item = {k: v[idx] for k, v in self.encodings.items()}
        # car le model attent le paramètre labels pour l'entrainement
        item["labels"] = self.labels[idx]
        item["extra_features"] = self.extra_features[idx]
        return item

    def __len__(self):
        return len(self.labels)
    

class ImageDataset(Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x = torchvision.io.read_image(self.X[idx])
        x = x.float() / 255
        # par default interpolation bilinear
        x = torchvision.transforms.functional.resize(x, (128, 128))
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

class TextDataset(Dataset):

    def __init__(self, tokenizer, X, y):
        self.encodings = tokenizer(
            list(X['text']),
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"   # retourne directement un pytorch tensor
        )
        self.extra_features = torch.tensor(
            X.drop('text', axis=1).to_numpy(),
            dtype=torch.float32
        )
        self.labels = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        # on parcourt les clés du dictionnaire self.encoding
        # et on récupère les input_ids, token_type_ids et attention_mask de l'index
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["extra_features"] = self.extra_features[idx]
        return item

    def __len__(self):
        return len(self.labels)
    

class CombinedDataset(Dataset):

    def __init__(self, tokenizer, X, y, transform=None):
        self.impath = X['image_path']
        self.encodings = tokenizer(
            list(X['text']),
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"   # retourne directement un pytorch tensor
        )
        self.extra_features = torch.tensor(
            X.drop(['text', 'image_path'], axis=1).to_numpy(),
            dtype=torch.float32
        )
        self.labels = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __getitem__(self, idx):
        # on parcourt les clés du dictionnaire self.encoding
        # et on récupère les input_ids, token_type_ids et attention_mask de l'index
        item = {k: v[idx] for k, v in self.encodings.items()}
        image = torchvision.io.read_image(self.impath[idx])
        image = image.float() / 255
        image = torchvision.transforms.functional.resize(image, (128, 128))
        if self.transform:
            image = self.transform(image)
        item['image'] = image
        item["labels"] = self.labels[idx]
        item["extra_features"] = self.extra_features[idx]
        return item

    def __len__(self):
        return len(self.X)
