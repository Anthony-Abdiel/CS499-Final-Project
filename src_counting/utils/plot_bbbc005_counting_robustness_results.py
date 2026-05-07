import os
import csv
from io import StringIO

import matplotlib.pyplot as plt


# -----------------------------
# Input log text
# -----------------------------

LOG_TEXT = r"""
Using device: cuda
F1/w2 total samples: 600
Matched logical test samples: 90

Loaded Lightweight CNN checkpoint | Best Epoch: 30 | Best Val MAE: 1.8795
Lightweight CNN | Experiment: built_in_blur | Blur Level: F1 | Test Loss: 7.2170 | Test MAE: 2.1407 | Test RMSE: 2.6864
Lightweight CNN | Experiment: built_in_blur | Blur Level: F4 | Test Loss: 7.2525 | Test MAE: 2.1436 | Test RMSE: 2.6930
Lightweight CNN | Experiment: built_in_blur | Blur Level: F7 | Test Loss: 7.3423 | Test MAE: 2.1598 | Test RMSE: 2.7097
Lightweight CNN | Experiment: built_in_blur | Blur Level: F10 | Test Loss: 7.4229 | Test MAE: 2.1694 | Test RMSE: 2.7245
Lightweight CNN | Experiment: built_in_blur | Blur Level: F14 | Test Loss: 7.4828 | Test MAE: 2.1758 | Test RMSE: 2.7355
Lightweight CNN | Experiment: built_in_blur | Blur Level: F17 | Test Loss: 7.4852 | Test MAE: 2.1770 | Test RMSE: 2.7359
Lightweight CNN | Experiment: built_in_blur | Blur Level: F20 | Test Loss: 7.4862 | Test MAE: 2.1718 | Test RMSE: 2.7361
Lightweight CNN | Experiment: built_in_blur | Blur Level: F23 | Test Loss: 7.4820 | Test MAE: 2.1755 | Test RMSE: 2.7353
Lightweight CNN | Experiment: built_in_blur | Blur Level: F26 | Test Loss: 7.4592 | Test MAE: 2.1656 | Test RMSE: 2.7312
Lightweight CNN | Experiment: built_in_blur | Blur Level: F29 | Test Loss: 7.4756 | Test MAE: 2.1689 | Test RMSE: 2.7341
Lightweight CNN | Experiment: built_in_blur | Blur Level: F32 | Test Loss: 7.5103 | Test MAE: 2.1590 | Test RMSE: 2.7405
Lightweight CNN | Experiment: built_in_blur | Blur Level: F35 | Test Loss: 7.6940 | Test MAE: 2.1737 | Test RMSE: 2.7738
Lightweight CNN | Experiment: built_in_blur | Blur Level: F39 | Test Loss: 8.1227 | Test MAE: 2.2300 | Test RMSE: 2.8500
Lightweight CNN | Experiment: built_in_blur | Blur Level: F42 | Test Loss: 8.7681 | Test MAE: 2.3068 | Test RMSE: 2.9611
Lightweight CNN | Experiment: built_in_blur | Blur Level: F45 | Test Loss: 9.1807 | Test MAE: 2.3635 | Test RMSE: 3.0300
Lightweight CNN | Experiment: built_in_blur | Blur Level: F48 | Test Loss: 10.4072 | Test MAE: 2.5435 | Test RMSE: 3.2260
Lightweight CNN | Experiment: gaussian_noise_on_F1 | Noise Std: 0.0 | Test Loss: 7.2170 | Test MAE: 2.1407 | Test RMSE: 2.6864
Lightweight CNN | Experiment: gaussian_noise_on_F1 | Noise Std: 0.05 | Test Loss: 7.3516 | Test MAE: 2.1649 | Test RMSE: 2.7114
Lightweight CNN | Experiment: gaussian_noise_on_F1 | Noise Std: 0.1 | Test Loss: 9.6964 | Test MAE: 2.4592 | Test RMSE: 3.1139
Lightweight CNN | Experiment: gaussian_noise_on_F1 | Noise Std: 0.2 | Test Loss: 106.2313 | Test MAE: 9.2778 | Test RMSE: 10.3069

Loaded ResNet-18 checkpoint | Best Epoch: 22 | Best Val MAE: 1.8744
ResNet-18 | Experiment: built_in_blur | Blur Level: F1 | Test Loss: 7.0143 | Test MAE: 2.0575 | Test RMSE: 2.6485
ResNet-18 | Experiment: built_in_blur | Blur Level: F4 | Test Loss: 6.9672 | Test MAE: 2.0357 | Test RMSE: 2.6395
ResNet-18 | Experiment: built_in_blur | Blur Level: F7 | Test Loss: 7.3139 | Test MAE: 2.0306 | Test RMSE: 2.7044
ResNet-18 | Experiment: built_in_blur | Blur Level: F10 | Test Loss: 7.8947 | Test MAE: 2.0572 | Test RMSE: 2.8097
ResNet-18 | Experiment: built_in_blur | Blur Level: F14 | Test Loss: 9.2270 | Test MAE: 2.2049 | Test RMSE: 3.0376
ResNet-18 | Experiment: built_in_blur | Blur Level: F17 | Test Loss: 9.9162 | Test MAE: 2.3038 | Test RMSE: 3.1490
ResNet-18 | Experiment: built_in_blur | Blur Level: F20 | Test Loss: 11.7216 | Test MAE: 2.5639 | Test RMSE: 3.4237
ResNet-18 | Experiment: built_in_blur | Blur Level: F23 | Test Loss: 14.1730 | Test MAE: 2.8650 | Test RMSE: 3.7647
ResNet-18 | Experiment: built_in_blur | Blur Level: F26 | Test Loss: 16.9513 | Test MAE: 3.1771 | Test RMSE: 4.1172
ResNet-18 | Experiment: built_in_blur | Blur Level: F29 | Test Loss: 18.1342 | Test MAE: 3.2869 | Test RMSE: 4.2584
ResNet-18 | Experiment: built_in_blur | Blur Level: F32 | Test Loss: 21.1102 | Test MAE: 3.5581 | Test RMSE: 4.5946
ResNet-18 | Experiment: built_in_blur | Blur Level: F35 | Test Loss: 23.5683 | Test MAE: 3.7618 | Test RMSE: 4.8547
ResNet-18 | Experiment: built_in_blur | Blur Level: F39 | Test Loss: 26.3597 | Test MAE: 3.9709 | Test RMSE: 5.1342
ResNet-18 | Experiment: built_in_blur | Blur Level: F42 | Test Loss: 28.6371 | Test MAE: 4.1390 | Test RMSE: 5.3514
ResNet-18 | Experiment: built_in_blur | Blur Level: F45 | Test Loss: 29.4662 | Test MAE: 4.1966 | Test RMSE: 5.4283
ResNet-18 | Experiment: built_in_blur | Blur Level: F48 | Test Loss: 31.6161 | Test MAE: 4.3823 | Test RMSE: 5.6228
ResNet-18 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.0 | Test Loss: 7.0143 | Test MAE: 2.0575 | Test RMSE: 2.6485
ResNet-18 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.05 | Test Loss: 793.0739 | Test MAE: 22.0387 | Test RMSE: 28.1616
ResNet-18 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.1 | Test Loss: 1538.7288 | Test MAE: 32.2234 | Test RMSE: 39.2266
ResNet-18 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.2 | Test Loss: 1115.2612 | Test MAE: 28.2606 | Test RMSE: 33.3955

Loaded ViT-B/16 checkpoint | Best Epoch: 53 | Best Val MAE: 1.5867
ViT-B/16 | Experiment: built_in_blur | Blur Level: F1 | Test Loss: 6.7638 | Test MAE: 1.9093 | Test RMSE: 2.6007
ViT-B/16 | Experiment: built_in_blur | Blur Level: F4 | Test Loss: 6.8519 | Test MAE: 1.9214 | Test RMSE: 2.6176
ViT-B/16 | Experiment: built_in_blur | Blur Level: F7 | Test Loss: 7.0064 | Test MAE: 1.9351 | Test RMSE: 2.6470
ViT-B/16 | Experiment: built_in_blur | Blur Level: F10 | Test Loss: 7.2016 | Test MAE: 1.9616 | Test RMSE: 2.6836
ViT-B/16 | Experiment: built_in_blur | Blur Level: F14 | Test Loss: 7.5422 | Test MAE: 2.0112 | Test RMSE: 2.7463
ViT-B/16 | Experiment: built_in_blur | Blur Level: F17 | Test Loss: 7.6028 | Test MAE: 2.0164 | Test RMSE: 2.7573
ViT-B/16 | Experiment: built_in_blur | Blur Level: F20 | Test Loss: 8.0071 | Test MAE: 2.0868 | Test RMSE: 2.8297
ViT-B/16 | Experiment: built_in_blur | Blur Level: F23 | Test Loss: 8.2201 | Test MAE: 2.1285 | Test RMSE: 2.8671
ViT-B/16 | Experiment: built_in_blur | Blur Level: F26 | Test Loss: 8.7214 | Test MAE: 2.2298 | Test RMSE: 2.9532
ViT-B/16 | Experiment: built_in_blur | Blur Level: F29 | Test Loss: 8.6767 | Test MAE: 2.2229 | Test RMSE: 2.9456
ViT-B/16 | Experiment: built_in_blur | Blur Level: F32 | Test Loss: 9.2344 | Test MAE: 2.3525 | Test RMSE: 3.0388
ViT-B/16 | Experiment: built_in_blur | Blur Level: F35 | Test Loss: 9.4118 | Test MAE: 2.4149 | Test RMSE: 3.0679
ViT-B/16 | Experiment: built_in_blur | Blur Level: F39 | Test Loss: 9.8938 | Test MAE: 2.5436 | Test RMSE: 3.1454
ViT-B/16 | Experiment: built_in_blur | Blur Level: F42 | Test Loss: 10.5930 | Test MAE: 2.6924 | Test RMSE: 3.2547
ViT-B/16 | Experiment: built_in_blur | Blur Level: F45 | Test Loss: 10.5863 | Test MAE: 2.6998 | Test RMSE: 3.2537
ViT-B/16 | Experiment: built_in_blur | Blur Level: F48 | Test Loss: 11.5943 | Test MAE: 2.8606 | Test RMSE: 3.4050
ViT-B/16 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.0 | Test Loss: 6.7638 | Test MAE: 1.9093 | Test RMSE: 2.6007
ViT-B/16 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.05 | Test Loss: 6.4324 | Test MAE: 1.8591 | Test RMSE: 2.5362
ViT-B/16 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.1 | Test Loss: 7.2738 | Test MAE: 2.0138 | Test RMSE: 2.6970
ViT-B/16 | Experiment: gaussian_noise_on_F1 | Noise Std: 0.2 | Test Loss: 27.7155 | Test MAE: 4.4678 | Test RMSE: 5.2646

Summary
Model,Experiment,Blur Level,Noise Std,Test Loss,Test MAE,Test RMSE,Best Epoch,Best Val MAE
Lightweight CNN,built_in_blur,F1,0.0,7.2170,2.1407,2.6864,30,1.8795
Lightweight CNN,built_in_blur,F4,0.0,7.2525,2.1436,2.6930,30,1.8795
Lightweight CNN,built_in_blur,F7,0.0,7.3423,2.1598,2.7097,30,1.8795
Lightweight CNN,built_in_blur,F10,0.0,7.4229,2.1694,2.7245,30,1.8795
Lightweight CNN,built_in_blur,F14,0.0,7.4828,2.1758,2.7355,30,1.8795
Lightweight CNN,built_in_blur,F17,0.0,7.4852,2.1770,2.7359,30,1.8795
Lightweight CNN,built_in_blur,F20,0.0,7.4862,2.1718,2.7361,30,1.8795
Lightweight CNN,built_in_blur,F23,0.0,7.4820,2.1755,2.7353,30,1.8795
Lightweight CNN,built_in_blur,F26,0.0,7.4592,2.1656,2.7312,30,1.8795
Lightweight CNN,built_in_blur,F29,0.0,7.4756,2.1689,2.7341,30,1.8795
Lightweight CNN,built_in_blur,F32,0.0,7.5103,2.1590,2.7405,30,1.8795
Lightweight CNN,built_in_blur,F35,0.0,7.6940,2.1737,2.7738,30,1.8795
Lightweight CNN,built_in_blur,F39,0.0,8.1227,2.2300,2.8500,30,1.8795
Lightweight CNN,built_in_blur,F42,0.0,8.7681,2.3068,2.9611,30,1.8795
Lightweight CNN,built_in_blur,F45,0.0,9.1807,2.3635,3.0300,30,1.8795
Lightweight CNN,built_in_blur,F48,0.0,10.4072,2.5435,3.2260,30,1.8795
Lightweight CNN,gaussian_noise_on_F1,F1,0.0,7.2170,2.1407,2.6864,30,1.8795
Lightweight CNN,gaussian_noise_on_F1,F1,0.05,7.3516,2.1649,2.7114,30,1.8795
Lightweight CNN,gaussian_noise_on_F1,F1,0.1,9.6964,2.4592,3.1139,30,1.8795
Lightweight CNN,gaussian_noise_on_F1,F1,0.2,106.2313,9.2778,10.3069,30,1.8795
ResNet-18,built_in_blur,F1,0.0,7.0143,2.0575,2.6485,22,1.8744
ResNet-18,built_in_blur,F4,0.0,6.9672,2.0357,2.6395,22,1.8744
ResNet-18,built_in_blur,F7,0.0,7.3139,2.0306,2.7044,22,1.8744
ResNet-18,built_in_blur,F10,0.0,7.8947,2.0572,2.8097,22,1.8744
ResNet-18,built_in_blur,F14,0.0,9.2270,2.2049,3.0376,22,1.8744
ResNet-18,built_in_blur,F17,0.0,9.9162,2.3038,3.1490,22,1.8744
ResNet-18,built_in_blur,F20,0.0,11.7216,2.5639,3.4237,22,1.8744
ResNet-18,built_in_blur,F23,0.0,14.1730,2.8650,3.7647,22,1.8744
ResNet-18,built_in_blur,F26,0.0,16.9513,3.1771,4.1172,22,1.8744
ResNet-18,built_in_blur,F29,0.0,18.1342,3.2869,4.2584,22,1.8744
ResNet-18,built_in_blur,F32,0.0,21.1102,3.5581,4.5946,22,1.8744
ResNet-18,built_in_blur,F35,0.0,23.5683,3.7618,4.8547,22,1.8744
ResNet-18,built_in_blur,F39,0.0,26.3597,3.9709,5.1342,22,1.8744
ResNet-18,built_in_blur,F42,0.0,28.6371,4.1390,5.3514,22,1.8744
ResNet-18,built_in_blur,F45,0.0,29.4662,4.1966,5.4283,22,1.8744
ResNet-18,built_in_blur,F48,0.0,31.6161,4.3823,5.6228,22,1.8744
ResNet-18,gaussian_noise_on_F1,F1,0.0,7.0143,2.0575,2.6485,22,1.8744
ResNet-18,gaussian_noise_on_F1,F1,0.05,793.0739,22.0387,28.1616,22,1.8744
ResNet-18,gaussian_noise_on_F1,F1,0.1,1538.7288,32.2234,39.2266,22,1.8744
ResNet-18,gaussian_noise_on_F1,F1,0.2,1115.2612,28.2606,33.3955,22,1.8744
ViT-B/16,built_in_blur,F1,0.0,6.7638,1.9093,2.6007,53,1.5867
ViT-B/16,built_in_blur,F4,0.0,6.8519,1.9214,2.6176,53,1.5867
ViT-B/16,built_in_blur,F7,0.0,7.0064,1.9351,2.6470,53,1.5867
ViT-B/16,built_in_blur,F10,0.0,7.2016,1.9616,2.6836,53,1.5867
ViT-B/16,built_in_blur,F14,0.0,7.5422,2.0112,2.7463,53,1.5867
ViT-B/16,built_in_blur,F17,0.0,7.6028,2.0164,2.7573,53,1.5867
ViT-B/16,built_in_blur,F20,0.0,8.0071,2.0868,2.8297,53,1.5867
ViT-B/16,built_in_blur,F23,0.0,8.2201,2.1285,2.8671,53,1.5867
ViT-B/16,built_in_blur,F26,0.0,8.7214,2.2298,2.9532,53,1.5867
ViT-B/16,built_in_blur,F29,0.0,8.6767,2.2229,2.9456,53,1.5867
ViT-B/16,built_in_blur,F32,0.0,9.2344,2.3525,3.0388,53,1.5867
ViT-B/16,built_in_blur,F35,0.0,9.4118,2.4149,3.0679,53,1.5867
ViT-B/16,built_in_blur,F39,0.0,9.8938,2.5436,3.1454,53,1.5867
ViT-B/16,built_in_blur,F42,0.0,10.5930,2.6924,3.2547,53,1.5867
ViT-B/16,built_in_blur,F45,0.0,10.5863,2.6998,3.2537,53,1.5867
ViT-B/16,built_in_blur,F48,0.0,11.5943,2.8606,3.4050,53,1.5867
ViT-B/16,gaussian_noise_on_F1,F1,0.0,6.7638,1.9093,2.6007,53,1.5867
ViT-B/16,gaussian_noise_on_F1,F1,0.05,6.4324,1.8591,2.5362,53,1.5867
ViT-B/16,gaussian_noise_on_F1,F1,0.1,7.2738,2.0138,2.6970,53,1.5867
ViT-B/16,gaussian_noise_on_F1,F1,0.2,27.7155,4.4678,5.2646,53,1.5867

"""


# -----------------------------
# Configuration
# -----------------------------

OUTPUT_DIR = "src_counting/visualizations/results_plots"


# -----------------------------
# Parsing
# -----------------------------

def extract_summary_csv(log_text):
    marker = "Summary"

    if marker not in log_text:
        raise ValueError("Could not find 'Summary' section in the log text.")

    summary_part = log_text.split(marker, 1)[1].strip()

    lines = []

    for line in summary_part.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    if not lines:
        raise ValueError("Summary section was empty.")

    return "\n".join(lines)


def parse_blur_level(blur_level_text):
    """
    Converts:
        F1  -> 1
        F48 -> 48
    """

    blur_level_text = blur_level_text.strip()

    if not blur_level_text.startswith("F"):
        raise ValueError(f"Unexpected blur level format: {blur_level_text}")

    return int(blur_level_text[1:])


def parse_summary_rows(log_text):
    csv_text = extract_summary_csv(log_text)

    reader = csv.DictReader(StringIO(csv_text))
    rows = []

    for row in reader:
        parsed = {
            "model": row["Model"],
            "experiment": row["Experiment"],
            "blur_level_text": row["Blur Level"],
            "blur_level": parse_blur_level(row["Blur Level"]),
            "noise_std": float(row["Noise Std"]),
            "test_loss": float(row["Test Loss"]),
            "test_mae": float(row["Test MAE"]),
            "test_rmse": float(row["Test RMSE"]),
            "best_epoch": int(row["Best Epoch"]),
            "best_val_mae": float(row["Best Val MAE"]),
        }

        rows.append(parsed)

    if not rows:
        raise ValueError("No data rows were parsed from the summary table.")

    return rows


# -----------------------------
# Plot helpers
# -----------------------------

def plot_clean_mae_bar(rows, output_path):
    """
    Uses the clean F1/no-noise result from gaussian_noise_on_F1.
    This avoids duplicating the built-in blur F1 rows.
    """

    clean_rows = [
        row for row in rows
        if row["experiment"] == "gaussian_noise_on_F1"
        and row["noise_std"] == 0.0
    ]

    clean_rows.sort(key=lambda row: row["test_mae"])

    model_names = [row["model"] for row in clean_rows]
    maes = [row["test_mae"] for row in clean_rows]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, maes)

    plt.title("BBBC005 Clean Counting Test MAE")
    plt.xlabel("Model")
    plt.ylabel("Test MAE (cells)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mae_vs_blur(rows, output_path):
    blur_rows = [
        row for row in rows
        if row["experiment"] == "built_in_blur"
    ]

    models = sorted(set(row["model"] for row in blur_rows))

    plt.figure(figsize=(10, 6))

    for model in models:
        model_rows = [
            row for row in blur_rows
            if row["model"] == model
        ]

        model_rows.sort(key=lambda row: row["blur_level"])

        blur_levels = [row["blur_level"] for row in model_rows]
        maes = [row["test_mae"] for row in model_rows]

        plt.plot(
            blur_levels,
            maes,
            marker="o",
            linewidth=2,
            markersize=5,
            label=model,
        )

    plt.title("BBBC005 Counting Robustness: Test MAE vs Built-in Blur Level")
    plt.xlabel("BBBC005 Blur Level")
    plt.ylabel("Test MAE (cells)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mae_vs_noise(rows, output_path):
    noise_rows = [
        row for row in rows
        if row["experiment"] == "gaussian_noise_on_F1"
    ]

    models = sorted(set(row["model"] for row in noise_rows))

    plt.figure(figsize=(10, 6))

    for model in models:
        model_rows = [
            row for row in noise_rows
            if row["model"] == model
        ]

        model_rows.sort(key=lambda row: row["noise_std"])

        noise_stds = [row["noise_std"] for row in model_rows]
        maes = [row["test_mae"] for row in model_rows]

        plt.plot(
            noise_stds,
            maes,
            marker="o",
            linewidth=2,
            markersize=5,
            label=model,
        )

    plt.title("BBBC005 Counting Robustness: Test MAE vs Gaussian Noise")
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Test MAE (cells)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rmse_vs_blur(rows, output_path):
    blur_rows = [
        row for row in rows
        if row["experiment"] == "built_in_blur"
    ]

    models = sorted(set(row["model"] for row in blur_rows))

    plt.figure(figsize=(10, 6))

    for model in models:
        model_rows = [
            row for row in blur_rows
            if row["model"] == model
        ]

        model_rows.sort(key=lambda row: row["blur_level"])

        blur_levels = [row["blur_level"] for row in model_rows]
        rmses = [row["test_rmse"] for row in model_rows]

        plt.plot(
            blur_levels,
            rmses,
            marker="o",
            linewidth=2,
            markersize=5,
            label=model,
        )

    plt.title("BBBC005 Counting Robustness: Test RMSE vs Built-in Blur Level")
    plt.xlabel("BBBC005 Blur Level")
    plt.ylabel("Test RMSE (cells)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rmse_vs_noise(rows, output_path):
    noise_rows = [
        row for row in rows
        if row["experiment"] == "gaussian_noise_on_F1"
    ]

    models = sorted(set(row["model"] for row in noise_rows))

    plt.figure(figsize=(10, 6))

    for model in models:
        model_rows = [
            row for row in noise_rows
            if row["model"] == model
        ]

        model_rows.sort(key=lambda row: row["noise_std"])

        noise_stds = [row["noise_std"] for row in model_rows]
        rmses = [row["test_rmse"] for row in model_rows]

        plt.plot(
            noise_stds,
            rmses,
            marker="o",
            linewidth=2,
            markersize=5,
            label=model,
        )

    plt.title("BBBC005 Counting Robustness: Test RMSE vs Gaussian Noise")
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Test RMSE (cells)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = parse_summary_rows(LOG_TEXT)

    clean_mae_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_clean_test_mae_bar.png"
    )

    blur_mae_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_blur_test_mae.png"
    )

    noise_mae_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_noise_test_mae.png"
    )

    blur_rmse_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_blur_test_rmse.png"
    )

    noise_rmse_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_noise_test_rmse.png"
    )

    plot_clean_mae_bar(rows, clean_mae_path)
    plot_mae_vs_blur(rows, blur_mae_path)
    plot_mae_vs_noise(rows, noise_mae_path)
    plot_rmse_vs_blur(rows, blur_rmse_path)
    plot_rmse_vs_noise(rows, noise_rmse_path)

    print(f"Saved clean MAE bar chart to: {clean_mae_path}")
    print(f"Saved blur MAE plot to: {blur_mae_path}")
    print(f"Saved noise MAE plot to: {noise_mae_path}")
    print(f"Saved blur RMSE plot to: {blur_rmse_path}")
    print(f"Saved noise RMSE plot to: {noise_rmse_path}")


if __name__ == "__main__":
    main()