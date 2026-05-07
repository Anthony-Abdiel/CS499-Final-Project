from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from src_counting.datasets.bbbc005_dataset import BBBC005Dataset


IMAGE_DIR = Path("./data/bbbc005/BBBC005_v1_images")

TARGET_CELL_COUNT = 53
TARGET_SAMPLE = 10
TARGET_STAIN = 2

BLUR_LEVELS_TO_SHOW = [1, 10, 20, 35, 48]
NOISE_STDS_TO_SHOW = [0.0, 0.05, 0.10, 0.20]

OUTPUT_PATH = "/scratch/aan266/project/bbbc005_corruption_visualization.png"


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


def build_image_index():
    image_index = {}

    for path in sorted(IMAGE_DIR.glob("*.TIF")):
        metadata = BBBC005Dataset.parse_bbbc005_metadata(path.name)

        key = (
            metadata["blur"],
            metadata["cell_count"],
            metadata["sample"],
            metadata["stain"],
        )

        image_index[key] = path

    return image_index


def find_matching_image(image_index, blur_level, cell_count, sample, stain):
    key = (
        blur_level,
        cell_count,
        sample,
        stain,
    )

    if key not in image_index:
        raise ValueError(
            f"No image found for "
            f"count={cell_count}, sample={sample}, stain={stain}, blur=F{blur_level}"
        )

    return image_index[key]


def load_grayscale_tensor(path):
    image = Image.open(path).convert("L")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    return transform(image)


def tensor_to_pil_image(tensor):
    tensor = tensor.squeeze(0).clamp(0.0, 1.0)
    tensor = tensor * 255.0
    tensor = tensor.byte()

    image = Image.fromarray(tensor.numpy(), mode="L")
    return image.convert("RGB")


def add_label(image, label):
    labeled_height = image.height + 30
    labeled = Image.new("RGB", (image.width, labeled_height), "white")
    labeled.paste(image, (0, 30))

    draw = ImageDraw.Draw(labeled)
    draw.text((8, 8), label, fill="black")

    return labeled


def make_row(images, labels, row_label, cell_width, cell_height):
    labeled_images = []

    for image, label in zip(images, labels):
        image = image.resize((cell_width, cell_height))
        labeled_images.append(add_label(image, label))

    row_label_width = 130
    row_height = cell_height + 30

    row = Image.new(
        "RGB",
        (row_label_width + cell_width * len(labeled_images), row_height),
        "white"
    )

    draw = ImageDraw.Draw(row)
    draw.text((8, row_height // 2 - 8), row_label, fill="black")

    x = row_label_width

    for image in labeled_images:
        row.paste(image, (x, 0))
        x += cell_width

    return row


print("Building image index...")
image_index = build_image_index()
print(f"Indexed {len(image_index)} images.")


# -----------------------------
# Built-in blur row
# -----------------------------

blur_images = []
blur_labels = []

for blur_level in BLUR_LEVELS_TO_SHOW:
    path = find_matching_image(
        image_index=image_index,
        blur_level=blur_level,
        cell_count=TARGET_CELL_COUNT,
        sample=TARGET_SAMPLE,
        stain=TARGET_STAIN,
    )

    tensor = load_grayscale_tensor(path)
    image = tensor_to_pil_image(tensor)

    blur_images.append(image)
    blur_labels.append(f"F{blur_level}")


# -----------------------------
# Noise row
# -----------------------------

clean_f1_path = find_matching_image(
    image_index=image_index,
    blur_level=1,
    cell_count=TARGET_CELL_COUNT,
    sample=TARGET_SAMPLE,
    stain=TARGET_STAIN,
)

clean_tensor = load_grayscale_tensor(clean_f1_path)

noise_images = []
noise_labels = []

for noise_std in NOISE_STDS_TO_SHOW:
    noise_transform = AddGaussianNoise(mean=0.0, std=noise_std)
    noisy_tensor = noise_transform(clean_tensor.clone())

    image = tensor_to_pil_image(noisy_tensor)

    noise_images.append(image)

    if noise_std == 0.0:
        noise_labels.append("clean")
    else:
        noise_labels.append(f"noise {noise_std}")


# Pad shorter row so both rows align visually
while len(noise_images) < len(blur_images):
    blank = Image.new("RGB", (256, 256), "white")
    noise_images.append(blank)
    noise_labels.append("")


# -----------------------------
# Combine rows
# -----------------------------

cell_width = 256
cell_height = 256

blur_row = make_row(
    images=blur_images,
    labels=blur_labels,
    row_label="Built-in blur",
    cell_width=cell_width,
    cell_height=cell_height,
)

noise_row = make_row(
    images=noise_images,
    labels=noise_labels,
    row_label="Gaussian noise",
    cell_width=cell_width,
    cell_height=cell_height,
)

title_height = 50

combined = Image.new(
    "RGB",
    (
        max(blur_row.width, noise_row.width),
        title_height + blur_row.height + noise_row.height
    ),
    "white"
)

draw = ImageDraw.Draw(combined)
draw.text(
    (8, 15),
    f"BBBC005 Visualization | count={TARGET_CELL_COUNT}, sample={TARGET_SAMPLE}, stain=w{TARGET_STAIN}",
    fill="black"
)

combined.paste(blur_row, (0, title_height))
combined.paste(noise_row, (0, title_height + blur_row.height))

combined.save(OUTPUT_PATH)
print(f"Saved visualization to: {OUTPUT_PATH}")