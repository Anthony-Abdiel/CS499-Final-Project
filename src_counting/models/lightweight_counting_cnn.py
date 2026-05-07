import torch
import torch.nn as nn


class LightweightCountingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [batch, 1, 128, 128]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 16, 64, 64]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 128, 8, 8]
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)

        # Cell counts cannot be negative.
        # Softplus keeps the output positive but smoother than ReLU.
        x = torch.nn.functional.softplus(x)

        return x