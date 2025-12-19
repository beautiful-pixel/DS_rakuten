"""
Neural network architectures for Rakuten product classification.

Includes:
1. ResNet50Classifier - Transfer learning with custom head
2. ViTClassifier - Vision Transformer with custom head
3. FusionClassifier - Ensemble of ResNet50 + ViT
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
try:
    # Try importing from torchvision first (available in torchvision >= 0.13)
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    TORCHVISION_VIT = True
except ImportError:
    # Fallback to timm if torchvision doesn't have ViT
    TORCHVISION_VIT = False
    try:
        import timm
        print("⚠ Using timm for Vision Transformer (torchvision ViT not available)")
    except ImportError:
        print("⚠ Warning: Neither torchvision ViT nor timm available. ViTClassifier may not work.")


class ResNet50Classifier(nn.Module):
    """
    ResNet50-based image classifier with custom head.

    Architecture:
    - Backbone: ResNet50 pretrained on ImageNet (frozen initially)
    - Custom Head: Linear(2048 -> 512) -> ReLU -> BatchNorm -> Dropout(0.3) -> Linear(512 -> num_classes)

    Args:
        num_classes (int): Number of output classes
        freeze_backbone (bool): Whether to freeze backbone initially (default: True)
        dropout_rate (float): Dropout probability in custom head (default: 0.3)

    Example:
        >>> model = ResNet50Classifier(num_classes=27)
        >>> model.unfreeze_backbone()  # Unfreeze for fine-tuning
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True, dropout_rate: float = 0.3):
        super(ResNet50Classifier, self).__init__()

        # Load pretrained ResNet50 backbone
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze backbone parameters if specified
        if freeze_backbone:
            self._freeze_backbone()

        # Get the input features for the custom head (2048 for ResNet50)
        in_features = self.backbone.fc.in_features

        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()

        # Custom classification head (as per strict requirements)
        self.custom_head = nn.Sequential(
            nn.Linear(in_features, 512),      # 2048 -> 512
            nn.ReLU(),                         # Activation
            nn.BatchNorm1d(512),              # Batch normalization
            nn.Dropout(p=dropout_rate),       # Dropout for regularization
            nn.Linear(512, num_classes)       # 512 -> num_classes
        )

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        print(f"✓ ResNet50Classifier initialized: {num_classes} classes, backbone {'frozen' if freeze_backbone else 'trainable'}")

    def _freeze_backbone(self):
        """Freeze all backbone parameters (transfer learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("✓ ResNet50 backbone unfrozen for fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch_size, 2048)

        # Pass through custom head
        logits = self.custom_head(features)  # (batch_size, num_classes)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (before final classification layer).

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Feature embeddings of shape (batch_size, 2048)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) based classifier with custom head.

    Uses ViT-B/16 pretrained on ImageNet21k and fine-tuned on ImageNet1k.

    Args:
        num_classes (int): Number of output classes
        freeze_backbone (bool): Whether to freeze backbone initially (default: True)
        dropout_rate (float): Dropout probability in custom head (default: 0.3)
        use_timm (bool): Force use of timm library instead of torchvision (default: False)

    Example:
        >>> model = ViTClassifier(num_classes=27)
        >>> model.unfreeze_backbone()
    """

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3,
        use_timm: bool = False
    ):
        super(ViTClassifier, self).__init__()

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # Load ViT backbone (try torchvision first, fallback to timm)
        if TORCHVISION_VIT and not use_timm:
            # Use torchvision ViT
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = self.backbone.heads.head.in_features  # 768 for ViT-B/16
            self.backbone.heads.head = nn.Identity()  # Remove original head
            print("✓ Using torchvision ViT-B/16")
        else:
            # Use timm ViT
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
            in_features = self.backbone.head.in_features  # 768 for ViT-B/16
            self.backbone.head = nn.Identity()  # Remove original head
            print("✓ Using timm ViT-B/16")

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        # Custom classification head (similar to ResNet50)
        self.custom_head = nn.Sequential(
            nn.Linear(in_features, 512),      # 768 -> 512
            nn.ReLU(),                         # Activation
            nn.BatchNorm1d(512),              # Batch normalization
            nn.Dropout(p=dropout_rate),       # Dropout for regularization
            nn.Linear(512, num_classes)       # 512 -> num_classes
        )

        print(f"✓ ViTClassifier initialized: {num_classes} classes, backbone {'frozen' if freeze_backbone else 'trainable'}")

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("✓ ViT backbone unfrozen for fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch_size, 768)

        # Pass through custom head
        logits = self.custom_head(features)  # (batch_size, num_classes)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        with torch.no_grad():
            features = self.backbone(x)
        return features


class FusionClassifier(nn.Module):
    """
    Advanced fusion model combining ResNet50 and ViT backbones.

    Architecture:
    1. Image -> ResNet50 backbone -> features_A (2048-dim)
    2. Image -> ViT backbone -> features_B (768-dim)
    3. Concatenate [features_A, features_B] -> fused_features (2816-dim)
    4. fused_features -> Custom MLP head -> logits

    This architecture leverages both CNN-based (ResNet) and attention-based (ViT)
    feature extraction for improved classification performance.

    Args:
        num_classes (int): Number of output classes
        freeze_backbones (bool): Whether to freeze both backbones initially (default: True)
        dropout_rate (float): Dropout probability in fusion head (default: 0.3)
        use_timm_vit (bool): Use timm for ViT instead of torchvision (default: False)

    Example:
        >>> model = FusionClassifier(num_classes=27)
        >>> model.unfreeze_backbones()  # Unfreeze for end-to-end fine-tuning
    """

    def __init__(
        self,
        num_classes: int,
        freeze_backbones: bool = True,
        dropout_rate: float = 0.3,
        use_timm_vit: bool = False
    ):
        super(FusionClassifier, self).__init__()

        self.num_classes = num_classes

        # ResNet50 backbone
        resnet_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet_features = resnet_backbone.fc.in_features  # 2048
        resnet_backbone.fc = nn.Identity()
        self.resnet_backbone = resnet_backbone

        # ViT backbone
        if TORCHVISION_VIT and not use_timm_vit:
            vit_backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            vit_features = vit_backbone.heads.head.in_features  # 768
            vit_backbone.heads.head = nn.Identity()
            self.vit_backbone = vit_backbone
            print("✓ Using torchvision ViT-B/16 for fusion")
        else:
            vit_backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
            vit_features = vit_backbone.head.in_features  # 768
            vit_backbone.head = nn.Identity()
            self.vit_backbone = vit_backbone
            print("✓ Using timm ViT-B/16 for fusion")

        # Freeze backbones if specified
        if freeze_backbones:
            self._freeze_backbones()

        # Fusion head: concatenated features -> classification
        fused_features = resnet_features + vit_features  # 2048 + 768 = 2816

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_features, 1024),   # 2816 -> 1024
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),              # 1024 -> 512
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)        # 512 -> num_classes
        )

        self.freeze_backbones = freeze_backbones

        print(f"✓ FusionClassifier initialized: {num_classes} classes")
        print(f"  ResNet features: {resnet_features}, ViT features: {vit_features}, Fused: {fused_features}")
        print(f"  Backbones {'frozen' if freeze_backbones else 'trainable'}")

    def _freeze_backbones(self):
        """Freeze both ResNet and ViT backbones."""
        for param in self.resnet_backbone.parameters():
            param.requires_grad = False
        for param in self.vit_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze both backbones for end-to-end fine-tuning."""
        for param in self.resnet_backbone.parameters():
            param.requires_grad = True
        for param in self.vit_backbone.parameters():
            param.requires_grad = True
        self.freeze_backbones = False
        print("✓ Both backbones unfrozen for fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features from both backbones
        resnet_features = self.resnet_backbone(x)  # (batch_size, 2048)
        vit_features = self.vit_backbone(x)        # (batch_size, 768)

        # Concatenate features
        fused_features = torch.cat([resnet_features, vit_features], dim=1)  # (batch_size, 2816)

        # Pass through fusion head
        logits = self.fusion_head(fused_features)  # (batch_size, num_classes)

        return logits

    def get_fused_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract fused feature embeddings (before final classification).

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Fused features of shape (batch_size, 2816)
        """
        with torch.no_grad():
            resnet_features = self.resnet_backbone(x)
            vit_features = self.vit_backbone(x)
            fused_features = torch.cat([resnet_features, vit_features], dim=1)
        return fused_features


# Model factory function
def create_model(model_name: str, num_classes: int, **kwargs):
    """
    Factory function to create models by name.

    Args:
        model_name: Name of the model ('resnet50', 'vit', 'fusion')
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor

    Returns:
        nn.Module: Instantiated model

    Example:
        >>> model = create_model('resnet50', num_classes=27, freeze_backbone=True)
    """
    model_name = model_name.lower()

    if model_name == 'resnet50':
        return ResNet50Classifier(num_classes=num_classes, **kwargs)
    elif model_name == 'vit':
        return ViTClassifier(num_classes=num_classes, **kwargs)
    elif model_name == 'fusion':
        return FusionClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from ['resnet50', 'vit', 'fusion']")


# Testing and model summary
if __name__ == "__main__":
    """Test model architectures."""
    print("=" * 60)
    print("Testing Model Architectures")
    print("=" * 60)

    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    num_classes = 27

    # Test ResNet50Classifier
    print("\n1. Testing ResNet50Classifier...")
    resnet_model = ResNet50Classifier(num_classes=num_classes)
    resnet_output = resnet_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {resnet_output.shape}")
    assert resnet_output.shape == (batch_size, num_classes), "ResNet50 output shape mismatch!"
    print("   ✓ ResNet50Classifier test passed")

    # Test ViTClassifier
    print("\n2. Testing ViTClassifier...")
    vit_model = ViTClassifier(num_classes=num_classes)
    vit_output = vit_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {vit_output.shape}")
    assert vit_output.shape == (batch_size, num_classes), "ViT output shape mismatch!"
    print("   ✓ ViTClassifier test passed")

    # Test FusionClassifier
    print("\n3. Testing FusionClassifier...")
    fusion_model = FusionClassifier(num_classes=num_classes)
    fusion_output = fusion_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {fusion_output.shape}")
    assert fusion_output.shape == (batch_size, num_classes), "Fusion output shape mismatch!"
    print("   ✓ FusionClassifier test passed")

    # Parameter count
    print("\n" + "=" * 60)
    print("Model Parameter Counts:")
    print("=" * 60)
    for name, model in [('ResNet50', resnet_model), ('ViT', vit_model), ('Fusion', fusion_model)]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:12} - Total: {total_params:,} | Trainable: {trainable_params:,}")

    print("\n✓ All model tests passed successfully!")

class LightweightFusionClassifier(nn.Module):
    """
    Memory-efficient fusion model for RTX 3060 Ti (8GB VRAM).

    It reduces feature dimensions from both backbones before fusion:
    - ResNet: 2048 -> 512
    - ViT: 768 -> 256
    - Fused: 768 -> classification head

    This significantly reduces the number of parameters in the fusion head
    compared to the standard FusionClassifier.
    """

    def __init__(
        self,
        num_classes: int,
        freeze_backbones: bool = True,
        dropout_rate: float = 0.3,
        use_timm_vit: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ResNet50 backbone
        resnet_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet_features = resnet_backbone.fc.in_features  # 2048
        resnet_backbone.fc = nn.Identity()
        self.resnet_backbone = resnet_backbone

        # ViT backbone
        if TORCHVISION_VIT and not use_timm_vit:
            vit_backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            vit_features = vit_backbone.heads.head.in_features  # 768
            vit_backbone.heads.head = nn.Identity()
            self.vit_backbone = vit_backbone
            print("✓ Using torchvision ViT-B/16 for lightweight fusion")
        else:
            vit_backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
            vit_features = vit_backbone.head.in_features  # 768
            vit_backbone.head = nn.Identity()
            self.vit_backbone = vit_backbone
            print("✓ Using timm ViT-B/16 for lightweight fusion")

        # Freeze backbones if specified
        if freeze_backbones:
            self._freeze_backbones()

        # Projection layers to reduce dimensionality
        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),
        )

        fused_features = 512 + 256  # 768

        # Lightweight fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        self.freeze_backbones = freeze_backbones

        print(f"✓ LightweightFusionClassifier initialized: {num_classes} classes")
        print(
            f"  ResNet proj: {resnet_features} -> 512, "
            f"ViT proj: {vit_features} -> 256, fused: {fused_features}"
        )

    def _freeze_backbones(self):
        """Freeze both backbones."""
        for p in self.resnet_backbone.parameters():
            p.requires_grad = False
        for p in self.vit_backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze both backbones for fine-tuning."""
        for p in self.resnet_backbone.parameters():
            p.requires_grad = True
        for p in self.vit_backbone.parameters():
            p.requires_grad = True
        self.freeze_backbones = False
        print("✓ Both backbones unfrozen for lightweight fusion fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with lightweight fusion."""
        resnet_features = self.resnet_backbone(x)  # (B, 2048)
        vit_features = self.vit_backbone(x)        # (B, 768)

        resnet_proj = self.resnet_proj(resnet_features)  # (B, 512)
        vit_proj = self.vit_proj(vit_features)          # (B, 256)

        fused = torch.cat([resnet_proj, vit_proj], dim=1)  # (B, 768)
        logits = self.fusion_head(fused)                   # (B, num_classes)
        return logits
