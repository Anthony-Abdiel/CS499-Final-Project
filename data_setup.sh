#!/bin/bash
set -e

# -----------------------------
# Create data directories
# -----------------------------
mkdir -p data/bloodmnist
mkdir -p data/bbbc005

# -----------------------------
# Download BloodMNIST
# -----------------------------
python -m pip install medmnist

python - <<'PY'
import os
from medmnist import BloodMNIST

os.makedirs("data/bloodmnist", exist_ok=True)
BloodMNIST(split="train", root="data/bloodmnist", download=True)

print("BloodMNIST download complete.")
PY

# Make sure bloodmnist.npz is directly under data/bloodmnist
FOUND_BLOODMNIST=$(find data/bloodmnist -name "bloodmnist.npz" | head -n 1)

if [ -n "$FOUND_BLOODMNIST" ] && [ "$FOUND_BLOODMNIST" != "data/bloodmnist/bloodmnist.npz" ]; then
    cp "$FOUND_BLOODMNIST" data/bloodmnist/bloodmnist.npz
fi

# -----------------------------
# Download BBBC005
# -----------------------------
cd data/bbbc005

if [ ! -f "BBBC005_v1_images.zip" ]; then
    wget https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip
fi

if [ ! -f "BBBC005_v1_ground_truth.zip" ]; then
    wget https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip
fi

if [ ! -d "images" ]; then
    unzip -q BBBC005_v1_images.zip

    if [ -d "BBBC005_v1_images" ]; then
        mv BBBC005_v1_images images
    fi
fi

if [ ! -d "ground_truth" ]; then
    unzip -q BBBC005_v1_ground_truth.zip

    if [ -d "BBBC005_v1_ground_truth" ]; then
        mv BBBC005_v1_ground_truth ground_truth
    fi
fi

cd ../..

# -----------------------------
# Verify setup
# -----------------------------
echo ""
echo "Checking dataset setup..."

test -f data/bloodmnist/bloodmnist.npz && echo "BloodMNIST found" || echo "BloodMNIST missing"
test -d data/bbbc005/images && echo "BBBC005 images folder found" || echo "BBBC005 images folder missing"
test -d data/bbbc005/ground_truth && echo "BBBC005 ground truth folder found" || echo "BBBC005 ground truth folder missing"

echo ""
echo "Dataset setup complete."