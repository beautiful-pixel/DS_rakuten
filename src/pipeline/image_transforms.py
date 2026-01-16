from torchvision import transforms


def build_swin_transform(img_size: int = 224):
    """
    Transform EXACTEMENT équivalente à celle utilisée à l'export Swin.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_convnext_transform(img_size: int = 384):
    """
    Transform EXACTEMENT équivalente à celle utilisée à l'export ConvNeXt.
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
