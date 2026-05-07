import torch
import torch.nn as nn
from torchvision import models


class ResNet18Counting(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnet18(weights=weights)

        in_features = self.model.fc.in_features

        # Replace the classification head with a regression head.
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.model(x)

        # Cell counts should be nonnegative.
        x = torch.nn.functional.softplus(x)

        return x