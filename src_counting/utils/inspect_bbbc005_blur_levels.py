from pathlib import Path
from src_counting.datasets.bbbc005_dataset import BBBC005Dataset

IMAGE_DIR = Path("./data/bbbc005/BBBC005_v1_images")

blur_levels = set()

for path in IMAGE_DIR.glob("*.TIF"):
    metadata = BBBC005Dataset.parse_bbbc005_metadata(path.name)

    if metadata["stain"] == 2:
        blur_levels.add(metadata["blur"])

print("Available blur levels for stain w2:")
print(sorted(blur_levels))