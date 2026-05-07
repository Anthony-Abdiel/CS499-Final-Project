from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = Path("./data/bloodmnist/bloodmnist.npz")

OUTPUT_PATH = Path("./src_classifying/visualizations/bloodmnist_corruption_visualization.png")

# Pick which split/image to visualize
SPLIT = "test"
IMAGE_INDEX = 0

# Corruption levels to visualize
BLUR_SIGMAS = [0.5, 1.0, 2.0]
NOISE_STDS = [0.05, 0.10, 0.20]

# Make tiny 28x28 images easier to see
DISPLAY_SIZE = 160


# -----------------------------
# Helpers
# -----------------------------

def load_bloodmnist_image(data_path, split, index):
    data = np.load(data_path)

    image_key = f"{split}_images"
    label_key = f"{split}_labels"

    if image_key not in data:
        raise KeyError(
            f"Could not find key '{image_key}' in {data_path}. "
            f"Available keys: {list(data.keys())}"
        )

    image = data[image_key][index]

    label = None
    if label_key in data:
        label_array = data[label_key][index]
        label = int(np.squeeze(label_array))

    # BloodMNIST images are usually uint8 RGB, shape [28, 28, 3].
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image).convert("RGB")

    return pil_image, label


def add_gaussian_noise(image, std):
    """
    Adds Gaussian noise to a PIL image.

    std is in normalized [0, 1] units.
    Example:
        std=0.10 means noise standard deviation is about 10% of pixel range.
    """

    image_array = np.asarray(image).astype(np.float32) / 255.0

    noise = np.random.normal(
        loc=0.0,
        scale=std,
        size=image_array.shape
    ).astype(np.float32)

    noisy_array = image_array + noise
    noisy_array = np.clip(noisy_array, 0.0, 1.0)

    noisy_array = (noisy_array * 255.0).astype(np.uint8)

    return Image.fromarray(noisy_array).convert("RGB")


def resize_for_display(image):
    return image.resize(
        (DISPLAY_SIZE, DISPLAY_SIZE),
        resample=Image.Resampling.NEAREST
    )


def add_label(image, label):
    label_height = 32

    labeled = Image.new(
        "RGB",
        (image.width, image.height + label_height),
        "white"
    )

    labeled.paste(image, (0, label_height))

    draw = ImageDraw.Draw(labeled)
    draw.text((8, 10), label, fill="black")

    return labeled


def make_row(images, labels, row_label):
    labeled_images = []

    for image, label in zip(images, labels):
        display_image = resize_for_display(image)
        labeled_images.append(add_label(display_image, label))

    row_label_width = 120
    row_height = labeled_images[0].height
    row_width = row_label_width + sum(image.width for image in labeled_images)

    row = Image.new(
        "RGB",
        (row_width, row_height),
        "white"
    )

    draw = ImageDraw.Draw(row)
    draw.text((8, row_height // 2 - 8), row_label, fill="black")

    x = row_label_width

    for image in labeled_images:
        row.paste(image, (x, 0))
        x += image.width

    return row


# -----------------------------
# Main
# -----------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find BloodMNIST file: {DATA_PATH}")

    clean_image, label = load_bloodmnist_image(
        data_path=DATA_PATH,
        split=SPLIT,
        index=IMAGE_INDEX
    )

    # -----------------------------
    # Blur row
    # -----------------------------

    blur_images = [clean_image]
    blur_labels = ["clean"]

    for sigma in BLUR_SIGMAS:
        blurred = clean_image.filter(
            ImageFilter.GaussianBlur(radius=sigma)
        )

        blur_images.append(blurred)
        blur_labels.append(f"blur {sigma}")

    # -----------------------------
    # Noise row
    # -----------------------------

    noise_images = [clean_image]
    noise_labels = ["clean"]

    for std in NOISE_STDS:
        noisy = add_gaussian_noise(clean_image, std=std)

        noise_images.append(noisy)
        noise_labels.append(f"noise {std}")

    # -----------------------------
    # Combine rows
    # -----------------------------

    blur_row = make_row(
        images=blur_images,
        labels=blur_labels,
        row_label="Gaussian blur"
    )

    noise_row = make_row(
        images=noise_images,
        labels=noise_labels,
        row_label="Gaussian noise"
    )

    title_height = 50

    combined_width = max(blur_row.width, noise_row.width)
    combined_height = title_height + blur_row.height + noise_row.height

    combined = Image.new(
        "RGB",
        (combined_width, combined_height),
        "white"
    )

    draw = ImageDraw.Draw(combined)

    title = f"BloodMNIST Corruption Visualization | split={SPLIT}, index={IMAGE_INDEX}"

    if label is not None:
        title += f", label={label}"

    draw.text((8, 18), title, fill="black")

    combined.paste(blur_row, (0, title_height))
    combined.paste(noise_row, (0, title_height + blur_row.height))

    combined.save(OUTPUT_PATH)

    print(f"Saved visualization to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()