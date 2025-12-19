"""
Image transformations and augmentations for training and validation.
"""

from torchvision import transforms


def get_train_transforms(img_size: int = 224, augment: bool = True):
    """
    Get training transformations with optional data augmentation.

    Args:
        img_size: Target image size (default: 224 for ImageNet models)
        augment: Whether to apply data augmentation (default: True)

    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    if augment:
        # Training with augmentation for better generalization
        transform_list = [
            transforms.Resize((img_size + 32, img_size + 32)),  # Slightly larger for random crop
            transforms.RandomCrop(img_size),  # Random crop to img_size
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
            transforms.RandomRotation(degrees=15),  # Random rotation ±15 degrees
            transforms.ColorJitter(
                brightness=0.2,  # Brightness variation
                contrast=0.2,    # Contrast variation
                saturation=0.2,  # Saturation variation
                hue=0.1          # Hue variation
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Random translation ±10%
                scale=(0.9, 1.1),      # Random scaling 90%-110%
            ),
            transforms.ToTensor(),  # Convert PIL Image to tensor [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            ),
        ]
    else:
        # Training without augmentation (minimal preprocessing)
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]

    return transforms.Compose(transform_list)


def get_val_transforms(img_size: int = 224):
    """
    Get validation/test transformations (no augmentation).

    Args:
        img_size: Target image size (default: 224 for ImageNet models)

    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    transform_list = [
        transforms.Resize((img_size, img_size)),  # Resize to target size
        transforms.ToTensor(),  # Convert PIL Image to tensor [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean (RGB)
            std=[0.229, 0.224, 0.225]    # ImageNet std (RGB)
        ),
    ]

    return transforms.Compose(transform_list)


# Example usage and verification
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Create a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))

    print("Testing transforms...")

    # Test training transforms
    train_transform = get_train_transforms(img_size=224, augment=True)
    train_tensor = train_transform(dummy_img)
    print(f"✓ Train transform output shape: {train_tensor.shape}")
    print(f"  Value range: [{train_tensor.min():.2f}, {train_tensor.max():.2f}]")

    # Test validation transforms
    val_transform = get_val_transforms(img_size=224)
    val_tensor = val_transform(dummy_img)
    print(f"✓ Val transform output shape: {val_tensor.shape}")
    print(f"  Value range: [{val_tensor.min():.2f}, {val_tensor.max():.2f}]")

    # Test TTA transforms
    tta_transforms = get_tta_transforms(img_size=224)
    print(f"✓ TTA transforms: {len(tta_transforms)} variants")
    for i, tta_transform in enumerate(tta_transforms):
        tta_tensor = tta_transform(dummy_img)
        print(f"  TTA {i+1} shape: {tta_tensor.shape}")
