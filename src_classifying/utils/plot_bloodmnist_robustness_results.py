import os
import re
import csv
from io import StringIO

import matplotlib.pyplot as plt


# -----------------------------
# Input log text
# -----------------------------

LOG_TEXT = r"""
Using device: cuda

Loaded Lightweight CNN checkpoint | Best Epoch: 33 | Best Val Acc: 0.9340
Lightweight CNN | Condition: normal | Test Loss: 0.2729 | Test Acc: 0.9281
Lightweight CNN | Condition: blur_sigma_0.5 | Test Loss: 0.3851 | Test Acc: 0.8992
Lightweight CNN | Condition: blur_sigma_1.0 | Test Loss: 0.9720 | Test Acc: 0.7928
Lightweight CNN | Condition: blur_sigma_2.0 | Test Loss: 1.1866 | Test Acc: 0.7638
Lightweight CNN | Condition: noise_std_0.05 | Test Loss: 0.6143 | Test Acc: 0.8702
Lightweight CNN | Condition: noise_std_0.1 | Test Loss: 2.6816 | Test Acc: 0.7021
Lightweight CNN | Condition: noise_std_0.2 | Test Loss: 7.7120 | Test Acc: 0.5583

Loaded ResNet-18 checkpoint | Best Epoch: 19 | Best Val Acc: 0.9597
ResNet-18 | Condition: normal | Test Loss: 0.2166 | Test Acc: 0.9547
ResNet-18 | Condition: blur_sigma_0.5 | Test Loss: 0.2129 | Test Acc: 0.9562
ResNet-18 | Condition: blur_sigma_1.0 | Test Loss: 0.2046 | Test Acc: 0.9567
ResNet-18 | Condition: blur_sigma_2.0 | Test Loss: 0.2036 | Test Acc: 0.9547
ResNet-18 | Condition: noise_std_0.05 | Test Loss: 0.3287 | Test Acc: 0.9322
ResNet-18 | Condition: noise_std_0.1 | Test Loss: 0.9124 | Test Acc: 0.8351
ResNet-18 | Condition: noise_std_0.2 | Test Loss: 5.8610 | Test Acc: 0.3911

Loaded ViT-B/16 checkpoint | Best Epoch: 24 | Best Val Acc: 0.9690
ViT-B/16 | Condition: normal | Test Loss: 0.1517 | Test Acc: 0.9567
ViT-B/16 | Condition: blur_sigma_0.5 | Test Loss: 0.1520 | Test Acc: 0.9564
ViT-B/16 | Condition: blur_sigma_1.0 | Test Loss: 0.1531 | Test Acc: 0.9562
ViT-B/16 | Condition: blur_sigma_2.0 | Test Loss: 0.1561 | Test Acc: 0.9538
ViT-B/16 | Condition: noise_std_0.05 | Test Loss: 0.1645 | Test Acc: 0.9532
ViT-B/16 | Condition: noise_std_0.1 | Test Loss: 0.1803 | Test Acc: 0.9509
ViT-B/16 | Condition: noise_std_0.2 | Test Loss: 0.2961 | Test Acc: 0.9214

Summary
Model,Condition,Corruption Type,Severity,Test Loss,Test Accuracy,Best Epoch,Best Val Accuracy
Lightweight CNN,normal,none,0.0,0.2729,0.9281,33,0.9340
Lightweight CNN,blur,gaussian_blur,0.5,0.3851,0.8992,33,0.9340
Lightweight CNN,blur,gaussian_blur,1.0,0.9720,0.7928,33,0.9340
Lightweight CNN,blur,gaussian_blur,2.0,1.1866,0.7638,33,0.9340
Lightweight CNN,noise,gaussian_noise,0.05,0.6143,0.8702,33,0.9340
Lightweight CNN,noise,gaussian_noise,0.1,2.6816,0.7021,33,0.9340
Lightweight CNN,noise,gaussian_noise,0.2,7.7120,0.5583,33,0.9340
ResNet-18,normal,none,0.0,0.2166,0.9547,19,0.9597
ResNet-18,blur,gaussian_blur,0.5,0.2129,0.9562,19,0.9597
ResNet-18,blur,gaussian_blur,1.0,0.2046,0.9567,19,0.9597
ResNet-18,blur,gaussian_blur,2.0,0.2036,0.9547,19,0.9597
ResNet-18,noise,gaussian_noise,0.05,0.3287,0.9322,19,0.9597
ResNet-18,noise,gaussian_noise,0.1,0.9124,0.8351,19,0.9597
ResNet-18,noise,gaussian_noise,0.2,5.8610,0.3911,19,0.9597
ViT-B/16,normal,none,0.0,0.1517,0.9567,24,0.9690
ViT-B/16,blur,gaussian_blur,0.5,0.1520,0.9564,24,0.9690
ViT-B/16,blur,gaussian_blur,1.0,0.1531,0.9562,24,0.9690
ViT-B/16,blur,gaussian_blur,2.0,0.1561,0.9538,24,0.9690
ViT-B/16,noise,gaussian_noise,0.05,0.1645,0.9532,24,0.9690
ViT-B/16,noise,gaussian_noise,0.1,0.1803,0.9509,24,0.9690
ViT-B/16,noise,gaussian_noise,0.2,0.2961,0.9214,24,0.9690
"""


# -----------------------------
# Configuration
# -----------------------------

OUTPUT_DIR = "/scratch/aan266/project/src_classifying/visualizations/results_plots"


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


def parse_summary_rows(log_text):
    csv_text = extract_summary_csv(log_text)

    reader = csv.DictReader(StringIO(csv_text))
    rows = []

    for row in reader:
        parsed = {
            "model": row["Model"],
            "condition": row["Condition"],
            "corruption_type": row["Corruption Type"],
            "severity": float(row["Severity"]),
            "test_loss": float(row["Test Loss"]),
            "test_accuracy": float(row["Test Accuracy"]),
            "best_epoch": int(row["Best Epoch"]),
            "best_val_accuracy": float(row["Best Val Accuracy"]),
        }
        rows.append(parsed)

    if not rows:
        raise ValueError("No data rows were parsed from the summary table.")

    return rows


# -----------------------------
# Plot helpers
# -----------------------------

def plot_clean_accuracy_bar(rows, output_path):
    clean_rows = [row for row in rows if row["condition"] == "normal"]

    clean_rows.sort(key=lambda row: row["test_accuracy"], reverse=True)

    model_names = [row["model"] for row in clean_rows]
    accuracies = [row["test_accuracy"] for row in clean_rows]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, accuracies)

    plt.title("BloodMNIST Clean Test Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_accuracy_vs_severity(rows, condition_name, output_path, title, x_label):
    filtered_rows = [row for row in rows if row["condition"] in ("normal", condition_name)]

    models = sorted(set(row["model"] for row in filtered_rows))

    plt.figure(figsize=(10, 6))

    for model in models:
        model_rows = [row for row in filtered_rows if row["model"] == model]

        model_rows.sort(key=lambda row: row["severity"])

        severities = [row["severity"] for row in model_rows]
        accuracies = [row["test_accuracy"] for row in model_rows]

        plt.plot(
            severities,
            accuracies,
            marker="o",
            linewidth=2,
            markersize=5,
            label=model,
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Test Accuracy")
    plt.ylim(0.0, 1.0)
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

    clean_bar_path = os.path.join(
        OUTPUT_DIR,
        "bloodmnist_clean_test_accuracy_bar.png"
    )

    blur_plot_path = os.path.join(
        OUTPUT_DIR,
        "bloodmnist_blur_test_accuracy.png"
    )

    noise_plot_path = os.path.join(
        OUTPUT_DIR,
        "bloodmnist_noise_test_accuracy.png"
    )

    plot_clean_accuracy_bar(rows, clean_bar_path)

    plot_accuracy_vs_severity(
        rows=rows,
        condition_name="blur",
        output_path=blur_plot_path,
        title="BloodMNIST Robustness: Test Accuracy vs Blur Severity",
        x_label="Blur Sigma",
    )

    plot_accuracy_vs_severity(
        rows=rows,
        condition_name="noise",
        output_path=noise_plot_path,
        title="BloodMNIST Robustness: Test Accuracy vs Noise Severity",
        x_label="Noise Standard Deviation",
    )

    print(f"Saved clean accuracy bar chart to: {clean_bar_path}")
    print(f"Saved blur robustness plot to: {blur_plot_path}")
    print(f"Saved noise robustness plot to: {noise_plot_path}")


if __name__ == "__main__":
    main()