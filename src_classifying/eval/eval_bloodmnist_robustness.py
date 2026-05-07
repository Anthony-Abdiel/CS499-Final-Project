import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from src_classifying.datasets.bloodmnist_dataset import BloodMNISTDataset
from src_classifying.models.lightweight_cnn import LightweightCNN
from src_classifying.models.resnet18_classifier import ResNet18Classifier
from src_classifying.models.vit_classifier import ViTClassifier


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "./data/bloodmnist/bloodmnist.npz"

BATCH_SIZE_CNN = 32
BATCH_SIZE_RESNET = 32
BATCH_SIZE_VIT = 16

CHECKPOINTS = {
    "Lightweight CNN": "./src/checkpoints/bloodmnist_lightweight_cnn_normal_best.pt",
    "ResNet-18": "./src/checkpoints/bloodmnist_resnet18_normal_best.pt",
    "ViT-B/16": "./src/checkpoints/bloodmnist_vit_normal_best.pt",
}

# Corruption severity levels
BLUR_SIGMAS = [0.5, 1.0, 2.0]
NOISE_STDS = [0.05, 0.10, 0.20]


# -----------------------------
# Setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()


# -----------------------------
# Noise Transform
# -----------------------------

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0.0, 1.0)


# -----------------------------
# Evaluation Helper
# -----------------------------

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


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# -----------------------------
# Corruption Configs
# -----------------------------

def build_condition_configs():
    """
    Returns a list of condition dictionaries.

    Each dictionary describes one test condition:
    - normal
    - blur with a specific sigma
    - noise with a specific std
    """

    condition_configs = [
        {
            "condition": "normal",
            "corruption_type": "none",
            "severity": 0.0,
            "label": "normal",
        }
    ]

    for sigma in BLUR_SIGMAS:
        condition_configs.append(
            {
                "condition": "blur",
                "corruption_type": "gaussian_blur",
                "severity": sigma,
                "label": f"blur_sigma_{sigma}",
            }
        )

    for std in NOISE_STDS:
        condition_configs.append(
            {
                "condition": "noise",
                "corruption_type": "gaussian_noise",
                "severity": std,
                "label": f"noise_std_{std}",
            }
        )

    return condition_configs


# -----------------------------
# Dataset Builders
# -----------------------------

def get_cnn_test_loader(condition_config):
    condition = condition_config["condition"]
    severity = condition_config["severity"]

    if condition == "normal":
        transform = None

    elif condition == "blur":
        transform = transforms.Compose([
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=severity
            )
        ])

    elif condition == "noise":
        transform = transforms.Compose([
            AddGaussianNoise(mean=0.0, std=severity)
        ])

    else:
        raise ValueError(f"Unknown condition: {condition}")

    dataset = BloodMNISTDataset(
        DATA_PATH,
        split="test",
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE_CNN,
        shuffle=False
    )


def get_resnet_vit_test_loader(condition_config, batch_size):
    condition = condition_config["condition"]
    severity = condition_config["severity"]

    if condition == "normal":
        transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])

    elif condition == "blur":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(
                kernel_size=7,
                sigma=severity
            )
        ])

    elif condition == "noise":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            AddGaussianNoise(mean=0.0, std=severity)
        ])

    else:
        raise ValueError(f"Unknown condition: {condition}")

    dataset = BloodMNISTDataset(
        DATA_PATH,
        split="test",
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )


# -----------------------------
# Model Evaluation Wrapper
# -----------------------------

def evaluate_model_on_conditions(
    model_name,
    model,
    checkpoint_path,
    condition_configs,
    loader_builder,
):
    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    print(
        f"\nLoaded {model_name} checkpoint | "
        f"Best Epoch: {checkpoint['epoch']} | "
        f"Best Val Acc: {checkpoint['val_acc']:.4f}"
    )

    model_results = []

    for condition_config in condition_configs:
        test_loader = loader_builder(condition_config)

        test_loss, test_acc = run_evaluation(
            model,
            test_loader,
            criterion,
            device
        )

        result = {
            "model_name": model_name,
            "condition": condition_config["condition"],
            "corruption_type": condition_config["corruption_type"],
            "severity": condition_config["severity"],
            "label": condition_config["label"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_epoch": checkpoint["epoch"],
            "best_val_acc": checkpoint["val_acc"],
        }

        model_results.append(result)

        print(
            f"{model_name} | "
            f"Condition: {condition_config['label']} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    return model_results


# -----------------------------
# Main Evaluation
# -----------------------------

condition_configs = build_condition_configs()
results = []

# Lightweight CNN
cnn_model = LightweightCNN().to(device)

results.extend(
    evaluate_model_on_conditions(
        model_name="Lightweight CNN",
        model=cnn_model,
        checkpoint_path=CHECKPOINTS["Lightweight CNN"],
        condition_configs=condition_configs,
        loader_builder=get_cnn_test_loader,
    )
)

# ResNet-18
resnet_model = ResNet18Classifier(num_classes=8, pretrained=False).to(device)

results.extend(
    evaluate_model_on_conditions(
        model_name="ResNet-18",
        model=resnet_model,
        checkpoint_path=CHECKPOINTS["ResNet-18"],
        condition_configs=condition_configs,
        loader_builder=lambda condition_config: get_resnet_vit_test_loader(
            condition_config,
            batch_size=BATCH_SIZE_RESNET
        ),
    )
)

# ViT-B/16
vit_model = ViTClassifier(num_classes=8, pretrained=False).to(device)

results.extend(
    evaluate_model_on_conditions(
        model_name="ViT-B/16",
        model=vit_model,
        checkpoint_path=CHECKPOINTS["ViT-B/16"],
        condition_configs=condition_configs,
        loader_builder=lambda condition_config: get_resnet_vit_test_loader(
            condition_config,
            batch_size=BATCH_SIZE_VIT
        ),
    )
)


# -----------------------------
# Summary Table
# -----------------------------

print("\nSummary")
print(
    "Model,Condition,Corruption Type,Severity,Test Loss,Test Accuracy,"
    "Best Epoch,Best Val Accuracy"
)

for result in results:
    print(
        f"{result['model_name']},"
        f"{result['condition']},"
        f"{result['corruption_type']},"
        f"{result['severity']},"
        f"{result['test_loss']:.4f},"
        f"{result['test_acc']:.4f},"
        f"{result['best_epoch']},"
        f"{result['best_val_acc']:.4f}"
    )