import os

os.environ["TORCH_HOME"] = "/scratch/aan266/torch_cache"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)


import random
import math

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src_counting.datasets.bbbc005_dataset import BBBC005Dataset
from src_counting.models.resnet18_counting import ResNet18Counting


# -----------------------------
# Configuration
# -----------------------------

IMAGE_DIR = "./data/bbbc005/BBBC005_v1_images"

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224

# Start simple:
# w2 = nuclei stain
# F1 = most in-focus images
STAIN = 2
BLUR_LEVELS = [1]

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

CHECKPOINT_DIR = "./src_counting/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "bbbc005_resnet18_counting_f1_w2_best.pt"
)


# -----------------------------
# Setup
# -----------------------------

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# Dataset and DataLoaders
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

dataset = BBBC005Dataset(
    image_dir=IMAGE_DIR,
    transform=transform,
    stain=STAIN,
    blur_levels=BLUR_LEVELS,
)

total_size = len(dataset)
train_size = int(TRAIN_SPLIT * total_size)
val_size = int(VAL_SPLIT * total_size)
test_size = total_size - train_size - val_size

print(f"Total samples: {total_size}")
print(f"Train samples: {train_size}")
print(f"Val samples: {val_size}")
print(f"Test samples: {test_size}")

generator = torch.Generator().manual_seed(RANDOM_SEED)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Model, Loss, Optimizer
# -----------------------------

model = ResNet18Counting(pretrained=True).to(device)

# MSELoss is common for regression training.
criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Metric Helpers
# -----------------------------

def compute_regression_metrics(predictions, targets):
    """
    predictions and targets should be tensors with shape [N, 1].
    Returns MAE and RMSE.
    """

    errors = predictions - targets

    mae = errors.abs().mean().item()
    mse = (errors ** 2).mean().item()
    rmse = math.sqrt(mse)

    return mae, rmse


def run_training_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    loss_total = 0.0
    all_predictions = []
    all_targets = []

    for images, counts in dataloader:
        images = images.to(device)
        counts = counts.to(device)

        predictions = model(images)
        loss = criterion(predictions, counts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item() * images.size(0)

        all_predictions.append(predictions.detach().cpu())
        all_targets.append(counts.detach().cpu())

    average_loss = loss_total / len(dataloader.dataset)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mae, rmse = compute_regression_metrics(
        all_predictions,
        all_targets
    )

    return average_loss, mae, rmse


def run_evaluation(model, dataloader, criterion, device):
    model.eval()

    loss_total = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, counts in dataloader:
            images = images.to(device)
            counts = counts.to(device)

            predictions = model(images)
            loss = criterion(predictions, counts)

            loss_total += loss.item() * images.size(0)

            all_predictions.append(predictions.detach().cpu())
            all_targets.append(counts.detach().cpu())

    average_loss = loss_total / len(dataloader.dataset)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mae, rmse = compute_regression_metrics(
        all_predictions,
        all_targets
    )

    return average_loss, mae, rmse


# -----------------------------
# Training Loop
# -----------------------------

best_val_mae = float("inf")
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    train_loss, train_mae, train_rmse = run_training_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    val_loss, val_mae, val_rmse = run_evaluation(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device
    )

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Train RMSE: {train_rmse:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}"
    )

    # For counting, lower validation MAE is better.
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_epoch = epoch + 1

        checkpoint = {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "stain": STAIN,
            "blur_levels": BLUR_LEVELS,
            "model_name": "Resnet18Counting",
            "dataset": "BBBC005",
            "task": "cell_count_regression",
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

        print(
            f"Saved new best model at epoch {best_epoch} "
            f"with Val MAE: {best_val_mae:.4f}"
        )


# -----------------------------
# Load Best Checkpoint
# -----------------------------

print(
    f"Loading best model from epoch {best_epoch} "
    f"with Val MAE: {best_val_mae:.4f}"
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])


# -----------------------------
# Final Test Evaluation
# -----------------------------

test_loss, test_mae, test_rmse = run_evaluation(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    device=device
)

print(
    f"Final Test Results using best validation checkpoint | "
    f"Best Epoch: {checkpoint['epoch']} | "
    f"Best Val MAE: {checkpoint['val_mae']:.4f} | "
    f"Test Loss: {test_loss:.4f} | "
    f"Test MAE: {test_mae:.4f} | "
    f"Test RMSE: {test_rmse:.4f}"
)