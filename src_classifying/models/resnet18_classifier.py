import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnet18(weights=weights)

        # Replace the original ImageNet classifier.
        # Original ResNet-18 outputs 1000 ImageNet classes.
        # BloodMNIST has 8 classes.
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)