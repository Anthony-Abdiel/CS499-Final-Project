import torch
import torch.nn as nn
from torchvision import models


class ViTCounting(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None

        self.model = models.vit_b_16(weights=weights)

        in_features = self.model.heads.head.in_features

        # Replace classification head with regression head
        self.model.heads.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.model(x)

        # Cell counts cannot be negative.
        x = torch.nn.functional.softplus(x)

        return x