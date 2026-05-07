import os
from pathlib import Path

DATA_DIR = Path("./data/bbbc005")

print(f"Data directory: {DATA_DIR.resolve()}")
print(f"Exists: {DATA_DIR.exists()}")

for root, dirs, files in os.walk(DATA_DIR):
    root_path = Path(root)
    image_files = [
        f for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]

    if image_files:
        print("\nFolder:", root_path)
        print("Number of image files:", len(image_files))
        print("First 10 image files:")
        for name in image_files[:10]:
            print("  ", name)