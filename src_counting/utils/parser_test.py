from pathlib import Path
import re

image_dir = Path("./data/bbbc005/BBBC005_v1_images")


def parse_bbbc005_metadata(filename):
    pattern = re.compile(
        r"SIMCEPImages_([A-P]\d{1,2})_C(\d+)_F(\d+)_s(\d+)_w(\d+)\.TIF",
        re.IGNORECASE
    )

    match = pattern.match(filename)

    if match is None:
        raise ValueError(f"Could not parse metadata from filename: {filename}")

    well = match.group(1)
    cell_count = int(match.group(2))
    blur = int(match.group(3))
    sample = int(match.group(4))
    stain = int(match.group(5))

    return {
        "well": well,
        "cell_count": cell_count,
        "blur": blur,
        "sample": sample,
        "stain": stain,
    }


for path in list(image_dir.glob("*.TIF"))[:20]:
    metadata = parse_bbbc005_metadata(path.name)
    print(path.name, "->", metadata)