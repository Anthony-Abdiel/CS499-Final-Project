# CS499 Final Project: Robust Cell Image Classification and Counting

This project investigates the performance and robustness of different deep learning models on biomedical image analysis tasks. The project focuses on two related problems:

1. **Cell image classification** using the BloodMNIST dataset.
2. **Cell counting/regression** using the BBBC005 synthetic microscopy dataset.

The goal is to compare how different model architectures perform under normal image conditions and degraded image conditions such as blur and noise. This allows the project to evaluate not only raw model performance, but also how well each architecture handles lower-quality image data that may appear in real-world biomedical imaging scenarios.

## Project Overview

Biomedical image analysis often requires models to classify cell types, detect abnormalities, or count biological structures in microscopy images. In real-world settings, image quality can vary due to focus issues, sensor noise, lighting conditions, and acquisition differences.

This project compares three model families:

- **Lightweight CNN**
  - A smaller custom convolutional neural network.
  - Designed to be simple, fast, and less computationally expensive.

- **ResNet-18**
  - A residual convolutional neural network.
  - Uses skip connections to support deeper feature extraction.
  - Commonly used as a strong convolutional baseline.

- **Vision Transformer**
  - A transformer-based image model.
  - Splits images into patches and learns relationships between patches using self-attention.
  - Used to compare transformer-based image learning against CNN-based approaches.

The models are evaluated on both clean and corrupted images to compare their robustness.

## Datasets

### BloodMNIST

BloodMNIST is part of the MedMNIST collection. It contains small RGB blood cell images labeled into multiple cell classes. In this project, BloodMNIST is used for the classification task.

The classification models are trained to predict the correct blood cell class from each image.

### BBBC005

BBBC005 is a synthetic microscopy dataset from the Broad Bioimage Benchmark Collection. It contains simulated fluorescence microscopy images with known object counts.

In this project, BBBC005 is used for the counting/regression task, where models estimate the number of cells or objects in an image.

## Project Structure

The project structure is as shown below:

```text
.
└── project/
    ├── data/
    ├── jobs/
    ├── src_classifying/
    │   ├── datasets/
    │   ├── eval/
    │   ├── models/
    │   ├── train/
    │   ├── transforms/
    │   ├── utils/
    │   └── visualizations/
    └── src_counting/
        ├── datasets/
        ├── eval/
        ├── models/
        ├── train/
        ├── transforms/
        ├── utils/
        └── visualizations/
```

The exact structure may vary slightly depending on the current version of the project.

---

# Setup Instructions

## 1. Clone the Repository

Clone the repository onto your local machine or HPC account.

Using SSH:

```bash
git clone git@github.com:Anthony-Abdiel/CS499-Final-Project.git
cd CS499-Final-Project
```

Using HTTPS:

```bash
git clone https://github.com/Anthony-Abdiel/CS499-Final-Project.git
cd CS499-Final-Project
```

## 2. Create or Activate a Python Environment

This project should be run inside a Python environment with PyTorch, torchvision, NumPy, matplotlib, pandas, scikit-learn, tqdm, and Pillow installed.

### Option A: Create a New Conda Environment

```bash
conda create -n cells-gpu python=3.10
conda activate cells-gpu
```

Install PyTorch and torchvision:

```bash
pip install torch torchvision torchaudio
```

Then install the remaining dependencies:

```bash
pip install numpy matplotlib pandas scikit-learn tqdm pillow
```

If a `requirements.txt` file is provided, install dependencies using:

```bash
pip install -r requirements.txt
```

### Option B: Use an Existing HPC Conda Environment

If the project is being run on the HPC and an environment already exists:

```bash
module load anaconda3
conda activate cells-gpu
```

Verify that PyTorch is installed:

```bash
python -c "import torch; print(torch.__version__)"
```

If using a GPU node, verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The output should be:

```text
True
```

when the code is running on a GPU-enabled node with the correct PyTorch installation.

## 3. Set Up the Data Directory

Datasets are not stored directly in this GitHub repository because they are too large for normal Git tracking.

Place datasets inside the `data/` directory.

Expected layout:

```text
data/
├── bloodmnist/
│   └── bloodmnist.npz
│
└── bbbc005/
    └── ...
```

The BloodMNIST dataset should be placed in:

```text
data/bloodmnist/bloodmnist.npz
```

The BBBC005 dataset should be placed in:

```text
data/bbbc005/
```

## 4. Keep Large Files Out of Git

Datasets, checkpoints, generated results, and log files should not be committed to GitHub.

A recommended `.gitignore` is:

```gitignore
# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# Environments
.env
.venv/
venv/
conda_envs/
cells-gpu/

# Data
data/
datasets/
*.npz
*.npy
*.csv
*.zip
*.tar
*.tar.gz

# Model outputs/checkpoints
checkpoints/
runs/
outputs/
results/
*.pth
*.pt
*.ckpt
*.onnx

# SLURM/logs
jobs/logs/
*.out
*.err
slurm-*.out

# OS/editor
.DS_Store
.vscode/
.idea/
```

Before committing, check the files Git is tracking:

```bash
git status
```

To check for large tracked files:

```bash
git ls-files | xargs -d '\n' du -h 2>/dev/null | sort -h | tail -30
```

---

# Execution Instructions

## Running Classification Experiments

Classification experiments are run on BloodMNIST.

Example commands:

```bash
python -m src.train.train_bloodmnist_cnn
```

```bash
python -m src.train.train_bloodmnist_resnet18
```

```bash
python -m src.train.train_bloodmnist_vit
```

These scripts train the selected model, evaluate validation performance, and save the best checkpoint.

## Running Counting Experiments

Counting experiments are run on BBBC005.

Example commands:

```bash
python -m src.train.train_bbbc005_cnn
```

```bash
python -m src.train.train_bbbc005_resnet18
```

```bash
python -m src.train.train_bbbc005_vit
```

These scripts train the selected model to estimate the number of objects in each image.

## Evaluating Robustness

After training, models can be evaluated under different image corruption conditions, including normal images, blurred images, and noisy images.

Example corruption conditions include:

```text
normal
blur_sigma_0.5
blur_sigma_1.0
blur_sigma_2.0
noise_std_0.05
noise_std_0.10
noise_std_0.20
```

The purpose of these tests is to determine how well each model maintains performance when image quality is reduced.

## Running on the HPC with SLURM

Training jobs can be submitted to the HPC using SLURM job scripts.

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=bloodmnist_resnet18
#SBATCH --output=jobs/logs/bloodmnist_resnet18_%j.out
#SBATCH --error=jobs/logs/bloodmnist_resnet18_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd /scratch/aan266/project

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cells-gpu

python -m src.train.train_bloodmnist_resnet18
```

Submit the job with:

```bash
sbatch jobs/bloodmnist_resnet18.slurm
```

Check job status with:

```bash
squeue -u aan266
```

View output logs with:

```bash
cat jobs/logs/bloodmnist_resnet18_<job_id>.out
```

View error logs with:

```bash
cat jobs/logs/bloodmnist_resnet18_<job_id>.err
```

## Expected Outputs

Training scripts generally produce:

- Training loss
- Training accuracy or error
- Validation loss
- Validation accuracy or error
- Best model checkpoint
- Final test results

Classification experiments report metrics such as:

- Test accuracy
- Test loss
- Robustness under blur
- Robustness under noise

Counting experiments report metrics such as:

- Mean absolute error
- Test loss
- Robustness under blur
- Robustness under noise

---

# Results

The project compares model performance across normal and corrupted image conditions.

The main comparison areas are:

- Classification accuracy on BloodMNIST
- Counting error on BBBC005
- Robustness to blur
- Robustness to noise
- Practicality of model size and complexity

## Expected Findings

This project investigates whether:

- Vision Transformers can perform well despite limited biomedical image data.
- ResNet-18 provides strong classification and counting performance but may be sensitive to image corruption.
- A lightweight CNN can remain competitive while requiring less complexity and computational overhead.
- Different architectures respond differently to blur and noise.

---

# Acknowledgements

This project uses publicly available biomedical imaging datasets and open-source machine learning tools.

Datasets used in this project include:

- **BloodMNIST**, from the MedMNIST collection.
- **BBBC005**, from the Broad Bioimage Benchmark Collection.

Software and libraries used in this project include:

- Python
- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib
- scikit-learn
- tqdm
- Pillow

This project was completed as part of a CS499 final project focused on biomedical image analysis, model comparison, and robustness evaluation.

## Author

Anthony Narvaez  
CS499 Final Project
