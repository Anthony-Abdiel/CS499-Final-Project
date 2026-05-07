import torch.nn as nn
from torchvision import models


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None

        self.model = models.vit_b_16(weights=weights)

        # Replace ImageNet classifier head with BloodMNIST classifier head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)