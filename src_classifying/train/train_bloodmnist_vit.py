import os

os.environ["TORCH_HOME"] = "/scratch/aan266/torch_cache"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from src_classifying.datasets.bloodmnist_dataset import BloodMNISTDataset
from src_classifying.models.vit_classifier import ViTClassifier


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "./data/bloodmnist/bloodmnist.npz"

BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001

CHECKPOINT_DIR = "./src/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "bloodmnist_vit_normal_best.pt"
)


# -----------------------------
# Setup
# -----------------------------

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# Dataset and DataLoaders
# -----------------------------

# Resize BloodMNIST images from 28x28 -> 224x224 for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224))
])

train_dataset = BloodMNISTDataset(DATA_PATH, split="train", transform=transform)
val_dataset = BloodMNISTDataset(DATA_PATH, split="val", transform=transform)
test_dataset = BloodMNISTDataset(DATA_PATH, split="test", transform=transform)

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

model = ViTClassifier(num_classes=8, pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------
# Helper Functions
# -----------------------------

def run_training_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    correct = 0
    total = 0
    loss_total = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        loss_total += loss.item() * labels.size(0)

    accuracy = correct / total
    average_loss = loss_total / total

    return average_loss, accuracy


def run_evaluation(model, dataloader, criterion, device):
    model.eval()

    correct = 0
    total = 0
    loss_total = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item() * labels.size(0)

    accuracy = correct / total
    average_loss = loss_total / total

    return average_loss, accuracy


# -----------------------------
# Training Loop
# -----------------------------

best_val_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = run_training_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    val_loss, val_acc = run_evaluation(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device
    )

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1

        checkpoint = {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model_name": "ViT-B/16",
            "dataset": "BloodMNIST",
            "condition": "normal",
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

        print(
            f"Saved new best model at epoch {best_epoch} "
            f"with Val Acc: {best_val_acc:.4f}"
        )


# -----------------------------
# Load Best Checkpoint
# -----------------------------

print(
    f"Loading best model from epoch {best_epoch} "
    f"with Val Acc: {best_val_acc:.4f}"
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])


# -----------------------------
# Final Test Evaluation
# -----------------------------

test_loss, test_acc = run_evaluation(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    device=device
)

print(
    f"Final Test Results using best validation checkpoint | "
    f"Best Epoch: {checkpoint['epoch']} | "
    f"Best Val Acc: {checkpoint['val_acc']:.4f} | "
    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
)