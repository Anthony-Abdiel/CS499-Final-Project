import math
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image

from src_counting.datasets.bbbc005_dataset import BBBC005Dataset
from src_counting.models.lightweight_counting_cnn import LightweightCountingCNN
from src_counting.models.resnet18_counting import ResNet18Counting
from src_counting.models.vit_counting import ViTCounting


# -----------------------------
# Configuration
# -----------------------------

IMAGE_DIR = "./data/bbbc005/BBBC005_v1_images"

STAIN = 2

# Actual BBBC005 blur levels found in the dataset for w2.
BLUR_LEVELS_TO_TEST = [
    1, 4, 7, 10,
    14, 17, 20, 23,
    26, 29, 32, 35,
    39, 42, 45, 48,
]

# Artificial Gaussian noise levels to test on clean F1 images.
NOISE_STDS_TO_TEST = [
    0.0,
    0.05,
    0.10,
    0.20,
]

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

BATCH_SIZE_CNN = 32
BATCH_SIZE_RESNET = 32
BATCH_SIZE_VIT = 16

CHECKPOINTS = {
    "Lightweight CNN": "./src_counting/checkpoints/bbbc005_lightweight_counting_cnn_f1_w2_best.pt",
    "ResNet-18": "./src_counting/checkpoints/bbbc005_resnet18_counting_f1_w2_best.pt",
    "ViT-B/16": "./src_counting/checkpoints/bbbc005_vit_counting_f1_w2_best.pt",
}


# -----------------------------
# Setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

criterion = nn.MSELoss()


# -----------------------------
# Noise Transform
# -----------------------------

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.std == 0.0:
            return tensor

        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = tensor + noise

        return torch.clamp(noisy_tensor, 0.0, 1.0)


# -----------------------------
# Metric Helpers
# -----------------------------

def compute_regression_metrics(predictions, targets):
    errors = predictions - targets

    mae = errors.abs().mean().item()
    mse = (errors ** 2).mean().item()
    rmse = math.sqrt(mse)

    return mae, rmse


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


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# -----------------------------
# Robustness Dataset
# -----------------------------

class BBBC005MatchedDataset(Dataset):
    """
    Dataset that evaluates the same held-out logical test samples
    at a chosen BBBC005 blur/focus level.

    target_keys contains tuples:
        (cell_count, sample, stain)

    This lets us take the original F1/w2 test split and find
    matching examples at F4, F7, F10, ..., F48.
    """

    def __init__(
        self,
        image_dir,
        target_keys,
        blur_level,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.target_keys = set(target_keys)
        self.blur_level = blur_level
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        self.samples = []

        for path in sorted(self.image_dir.glob("*.TIF")):
            metadata = BBBC005Dataset.parse_bbbc005_metadata(path.name)

            if metadata["blur"] != blur_level:
                continue

            key = (
                metadata["cell_count"],
                metadata["sample"],
                metadata["stain"],
            )

            if key not in self.target_keys:
                continue

            self.samples.append(
                {
                    "path": path,
                    "cell_count": metadata["cell_count"],
                    "blur": metadata["blur"],
                    "stain": metadata["stain"],
                    "sample": metadata["sample"],
                    "well": metadata["well"],
                }
            )

        if len(self.samples) == 0:
            raise ValueError(
                f"No matched samples found for blur level F{blur_level}"
            )

        if len(self.samples) != len(self.target_keys):
            print(
                f"Warning: found {len(self.samples)} matched samples "
                f"for {len(self.target_keys)} target keys at F{blur_level}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["path"]).convert("L")

        if self.transform is not None:
            image = self.transform(image)
        else:
            raise ValueError("A transform is required.")

        count = torch.tensor(
            [sample["cell_count"]],
            dtype=torch.float32
        )

        return image, count


# -----------------------------
# Build Same Test Split Keys
# -----------------------------

def get_f1_test_keys():
    """
    Recreates the original F1/w2 split using the same seed as training,
    then extracts logical test keys.

    This is important because we want to evaluate the same logical samples
    across all blur/noise conditions.
    """

    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    base_dataset = BBBC005Dataset(
        image_dir=IMAGE_DIR,
        transform=base_transform,
        stain=STAIN,
        blur_levels=[1],
    )

    total_size = len(base_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    _, _, test_dataset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    test_keys = []

    for index in test_dataset.indices:
        sample = base_dataset.samples[index]

        key = (
            sample["cell_count"],
            sample["sample"],
            sample["stain"],
        )

        test_keys.append(key)

    print(f"F1/w2 total samples: {total_size}")
    print(f"Matched logical test samples: {len(test_keys)}")

    return test_keys


# -----------------------------
# Transform Builders
# -----------------------------

def get_cnn_transform(noise_std=0.0):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=noise_std),
    ])


def get_resnet_vit_transform(noise_std=0.0):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=noise_std),
    ])


def get_matched_loader(test_keys, blur_level, transform, batch_size):
    dataset = BBBC005MatchedDataset(
        image_dir=IMAGE_DIR,
        target_keys=test_keys,
        blur_level=blur_level,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )


# -----------------------------
# Model Evaluation Wrappers
# -----------------------------

def evaluate_model_across_blur_levels(
    model_name,
    model,
    checkpoint_path,
    test_keys,
    transform_builder,
    batch_size,
):
    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    print(
        f"\nLoaded {model_name} checkpoint | "
        f"Best Epoch: {checkpoint['epoch']} | "
        f"Best Val MAE: {checkpoint['val_mae']:.4f}"
    )

    model_results = []

    for blur_level in BLUR_LEVELS_TO_TEST:
        test_loader = get_matched_loader(
            test_keys=test_keys,
            blur_level=blur_level,
            transform=transform_builder(noise_std=0.0),
            batch_size=batch_size,
        )

        test_loss, test_mae, test_rmse = run_evaluation(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device
        )

        result = {
            "model_name": model_name,
            "experiment": "built_in_blur",
            "blur_level": blur_level,
            "noise_std": 0.0,
            "test_loss": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "best_epoch": checkpoint["epoch"],
            "best_val_mae": checkpoint["val_mae"],
        }

        model_results.append(result)

        print(
            f"{model_name} | "
            f"Experiment: built_in_blur | "
            f"Blur Level: F{blur_level} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )

    return model_results


def evaluate_model_across_noise_levels(
    model_name,
    model,
    test_keys,
    transform_builder,
    batch_size,
):
    """
    Evaluates artificial Gaussian noise on clean F1/w2 matched test images.
    The checkpoint is already loaded by evaluate_model_across_blur_levels.
    """

    model_results = []

    for noise_std in NOISE_STDS_TO_TEST:
        test_loader = get_matched_loader(
            test_keys=test_keys,
            blur_level=1,
            transform=transform_builder(noise_std=noise_std),
            batch_size=batch_size,
        )

        test_loss, test_mae, test_rmse = run_evaluation(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device
        )

        result = {
            "model_name": model_name,
            "experiment": "gaussian_noise_on_F1",
            "blur_level": 1,
            "noise_std": noise_std,
            "test_loss": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "best_epoch": None,
            "best_val_mae": None,
        }

        model_results.append(result)

        print(
            f"{model_name} | "
            f"Experiment: gaussian_noise_on_F1 | "
            f"Noise Std: {noise_std} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )

    return model_results


def evaluate_model(
    model_name,
    model,
    checkpoint_path,
    test_keys,
    transform_builder,
    batch_size,
):
    blur_results = evaluate_model_across_blur_levels(
        model_name=model_name,
        model=model,
        checkpoint_path=checkpoint_path,
        test_keys=test_keys,
        transform_builder=transform_builder,
        batch_size=batch_size,
    )

    noise_results = evaluate_model_across_noise_levels(
        model_name=model_name,
        model=model,
        test_keys=test_keys,
        transform_builder=transform_builder,
        batch_size=batch_size,
    )

    # Add checkpoint metadata to noise results after checkpoint has been loaded.
    checkpoint_epoch = blur_results[0]["best_epoch"]
    checkpoint_val_mae = blur_results[0]["best_val_mae"]

    for result in noise_results:
        result["best_epoch"] = checkpoint_epoch
        result["best_val_mae"] = checkpoint_val_mae

    return blur_results + noise_results


# -----------------------------
# Main Evaluation
# -----------------------------

test_keys = get_f1_test_keys()

results = []

# Lightweight CNN
cnn_model = LightweightCountingCNN().to(device)

results.extend(
    evaluate_model(
        model_name="Lightweight CNN",
        model=cnn_model,
        checkpoint_path=CHECKPOINTS["Lightweight CNN"],
        test_keys=test_keys,
        transform_builder=get_cnn_transform,
        batch_size=BATCH_SIZE_CNN,
    )
)

# ResNet-18
resnet_model = ResNet18Counting(pretrained=False).to(device)

results.extend(
    evaluate_model(
        model_name="ResNet-18",
        model=resnet_model,
        checkpoint_path=CHECKPOINTS["ResNet-18"],
        test_keys=test_keys,
        transform_builder=get_resnet_vit_transform,
        batch_size=BATCH_SIZE_RESNET,
    )
)

# ViT-B/16
vit_model = ViTCounting(pretrained=False).to(device)

results.extend(
    evaluate_model(
        model_name="ViT-B/16",
        model=vit_model,
        checkpoint_path=CHECKPOINTS["ViT-B/16"],
        test_keys=test_keys,
        transform_builder=get_resnet_vit_transform,
        batch_size=BATCH_SIZE_VIT,
    )
)


# -----------------------------
# Summary Table
# -----------------------------

print("\nSummary")
print(
    "Model,Experiment,Blur Level,Noise Std,Test Loss,Test MAE,Test RMSE,"
    "Best Epoch,Best Val MAE"
)

for result in results:
    print(
        f"{result['model_name']},"
        f"{result['experiment']},"
        f"F{result['blur_level']},"
        f"{result['noise_std']},"
        f"{result['test_loss']:.4f},"
        f"{result['test_mae']:.4f},"
        f"{result['test_rmse']:.4f},"
        f"{result['best_epoch']},"
        f"{result['best_val_mae']:.4f}"
    )