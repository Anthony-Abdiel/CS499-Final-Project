import os
import re
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

LOG_FILES = {
    "Lightweight CNN": "src_counting/utils/training_logs/counting_cnn.txt",
    "ResNet-18": "src_counting/utils/training_logs/counting_resnet18.txt",
    "ViT-B/16": "src_counting/utils/training_logs/counting_vit.txt",
}

OUTPUT_DIR = "/scratch/aan266/project/src_counting/visualizations/training_plots"

SHOW_BEST_POINTS = True
SHOW_BEST_LABELS = False


# -----------------------------
# Log Parser
# -----------------------------

def parse_counting_training_log(log_text):
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
        r"Train Loss:\s+([0-9.]+)\s+\|\s+"
        r"Train MAE:\s+([0-9.]+)\s+\|\s+"
        r"Train RMSE:\s+([0-9.]+)\s+\|\s+"
        r"Val Loss:\s+([0-9.]+)\s+\|\s+"
        r"Val MAE:\s+([0-9.]+)\s+\|\s+"
        r"Val RMSE:\s+([0-9.]+)"
    )

    epochs = []
    train_losses = []
    train_maes = []
    train_rmses = []
    val_losses = []
    val_maes = []
    val_rmses = []

    for match in pattern.finditer(log_text):
        epochs.append(int(match.group(1)))
        train_losses.append(float(match.group(3)))
        train_maes.append(float(match.group(4)))
        train_rmses.append(float(match.group(5)))
        val_losses.append(float(match.group(6)))
        val_maes.append(float(match.group(7)))
        val_rmses.append(float(match.group(8)))

    if len(epochs) == 0:
        raise ValueError("No epoch lines were parsed. Check the log format.")

    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "train_maes": train_maes,
        "train_rmses": train_rmses,
        "val_losses": val_losses,
        "val_maes": val_maes,
        "val_rmses": val_rmses,
    }


def load_logs(log_files):
    data_by_model = {}

    for model_name, path in log_files.items():
        with open(path, "r", encoding="utf-8") as file:
            log_text = file.read()

        data_by_model[model_name] = parse_counting_training_log(log_text)

        print(
            f"Loaded {model_name}: "
            f"{len(data_by_model[model_name]['epochs'])} epochs"
        )

    return data_by_model


# -----------------------------
# Plot Helpers
# -----------------------------

def plot_val_mae(data_by_model, output_path):
    plt.figure(figsize=(10, 6))

    for model_name, metrics in data_by_model.items():
        plt.plot(
            metrics["epochs"],
            metrics["val_maes"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

        best_mae = min(metrics["val_maes"])
        best_idx = metrics["val_maes"].index(best_mae)
        best_epoch = metrics["epochs"][best_idx]

        if SHOW_BEST_POINTS:
            plt.scatter([best_epoch], [best_mae], s=70)

        if SHOW_BEST_LABELS:
            plt.annotate(
                f"{model_name} best: {best_mae:.4f} @ {best_epoch}",
                (best_epoch, best_mae),
                textcoords="offset points",
                xytext=(5, 5),
            )

    plt.title("BBBC005 Counting: Validation MAE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_val_rmse(data_by_model, output_path):
    plt.figure(figsize=(10, 6))

    for model_name, metrics in data_by_model.items():
        plt.plot(
            metrics["epochs"],
            metrics["val_rmses"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

        best_rmse = min(metrics["val_rmses"])
        best_idx = metrics["val_rmses"].index(best_rmse)
        best_epoch = metrics["epochs"][best_idx]

        if SHOW_BEST_POINTS:
            plt.scatter([best_epoch], [best_rmse], s=70)

        if SHOW_BEST_LABELS:
            plt.annotate(
                f"{model_name} best: {best_rmse:.4f} @ {best_epoch}",
                (best_epoch, best_rmse),
                textcoords="offset points",
                xytext=(5, 5),
            )

    plt.title("BBBC005 Counting: Validation RMSE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_val_loss(data_by_model, output_path):
    plt.figure(figsize=(10, 6))

    for model_name, metrics in data_by_model.items():
        plt.plot(
            metrics["epochs"],
            metrics["val_losses"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

        best_loss = min(metrics["val_losses"])
        best_idx = metrics["val_losses"].index(best_loss)
        best_epoch = metrics["epochs"][best_idx]

        if SHOW_BEST_POINTS:
            plt.scatter([best_epoch], [best_loss], s=70)

        if SHOW_BEST_LABELS:
            plt.annotate(
                f"{model_name} best: {best_loss:.4f} @ {best_epoch}",
                (best_epoch, best_loss),
                textcoords="offset points",
                xytext=(5, 5),
            )

    plt.title("BBBC005 Counting: Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss / MSE")
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

    data_by_model = load_logs(LOG_FILES)

    val_mae_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_val_mae_vs_epoch.png"
    )

    val_rmse_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_val_rmse_vs_epoch.png"
    )

    val_loss_path = os.path.join(
        OUTPUT_DIR,
        "bbbc005_val_loss_vs_epoch.png"
    )

    plot_val_mae(data_by_model, val_mae_path)
    plot_val_rmse(data_by_model, val_rmse_path)
    plot_val_loss(data_by_model, val_loss_path)

    print(f"Saved validation MAE plot to: {val_mae_path}")
    print(f"Saved validation RMSE plot to: {val_rmse_path}")
    print(f"Saved validation loss plot to: {val_loss_path}")


if __name__ == "__main__":
    main()