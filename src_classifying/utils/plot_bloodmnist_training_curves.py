import re
import os
import matplotlib.pyplot as plt


CNN_LOG = r"""
Using device: cpu
Epoch 1/40 | Train Loss: 1.1417 | Train Acc: 0.5679 | Val Loss: 0.8625 | Val Acc: 0.6641
Saved new best model at epoch 1 with Val Acc: 0.6641
Epoch 2/40 | Train Loss: 0.7128 | Train Acc: 0.7338 | Val Loss: 0.6332 | Val Acc: 0.7494
Saved new best model at epoch 2 with Val Acc: 0.7494
Epoch 3/40 | Train Loss: 0.6142 | Train Acc: 0.7737 | Val Loss: 0.5139 | Val Acc: 0.8067
Saved new best model at epoch 3 with Val Acc: 0.8067
Epoch 4/40 | Train Loss: 0.5286 | Train Acc: 0.8069 | Val Loss: 0.5081 | Val Acc: 0.7961
Epoch 5/40 | Train Loss: 0.4625 | Train Acc: 0.8329 | Val Loss: 0.4484 | Val Acc: 0.8312
Saved new best model at epoch 5 with Val Acc: 0.8312
Epoch 6/40 | Train Loss: 0.4098 | Train Acc: 0.8515 | Val Loss: 0.4298 | Val Acc: 0.8423
Saved new best model at epoch 6 with Val Acc: 0.8423
Epoch 7/40 | Train Loss: 0.3674 | Train Acc: 0.8691 | Val Loss: 0.3361 | Val Acc: 0.8744
Saved new best model at epoch 7 with Val Acc: 0.8744
Epoch 8/40 | Train Loss: 0.3428 | Train Acc: 0.8782 | Val Loss: 0.3749 | Val Acc: 0.8604
Epoch 9/40 | Train Loss: 0.3127 | Train Acc: 0.8869 | Val Loss: 0.3583 | Val Acc: 0.8732
Epoch 10/40 | Train Loss: 0.2916 | Train Acc: 0.8934 | Val Loss: 0.3110 | Val Acc: 0.8890
Saved new best model at epoch 10 with Val Acc: 0.8890
Epoch 11/40 | Train Loss: 0.2746 | Train Acc: 0.8987 | Val Loss: 0.3104 | Val Acc: 0.8966
Saved new best model at epoch 11 with Val Acc: 0.8966
Epoch 12/40 | Train Loss: 0.2481 | Train Acc: 0.9086 | Val Loss: 0.2851 | Val Acc: 0.8978
Saved new best model at epoch 12 with Val Acc: 0.8978
Epoch 13/40 | Train Loss: 0.2353 | Train Acc: 0.9150 | Val Loss: 0.2812 | Val Acc: 0.8989
Saved new best model at epoch 13 with Val Acc: 0.8989
Epoch 14/40 | Train Loss: 0.2203 | Train Acc: 0.9187 | Val Loss: 0.2586 | Val Acc: 0.9112
Saved new best model at epoch 14 with Val Acc: 0.9112
Epoch 15/40 | Train Loss: 0.2192 | Train Acc: 0.9231 | Val Loss: 0.2596 | Val Acc: 0.9106
Epoch 16/40 | Train Loss: 0.1981 | Train Acc: 0.9289 | Val Loss: 0.2544 | Val Acc: 0.9071
Epoch 17/40 | Train Loss: 0.1826 | Train Acc: 0.9339 | Val Loss: 0.2460 | Val Acc: 0.9194
Saved new best model at epoch 17 with Val Acc: 0.9194
Epoch 18/40 | Train Loss: 0.1716 | Train Acc: 0.9381 | Val Loss: 0.2415 | Val Acc: 0.9159
Epoch 19/40 | Train Loss: 0.1584 | Train Acc: 0.9439 | Val Loss: 0.2389 | Val Acc: 0.9165
Epoch 20/40 | Train Loss: 0.1649 | Train Acc: 0.9397 | Val Loss: 0.2396 | Val Acc: 0.9176
Epoch 21/40 | Train Loss: 0.1416 | Train Acc: 0.9495 | Val Loss: 0.2511 | Val Acc: 0.9176
Epoch 22/40 | Train Loss: 0.1283 | Train Acc: 0.9528 | Val Loss: 0.2241 | Val Acc: 0.9276
Saved new best model at epoch 22 with Val Acc: 0.9276
Epoch 23/40 | Train Loss: 0.1139 | Train Acc: 0.9590 | Val Loss: 0.2745 | Val Acc: 0.9153
Epoch 24/40 | Train Loss: 0.1117 | Train Acc: 0.9602 | Val Loss: 0.2337 | Val Acc: 0.9322
Saved new best model at epoch 24 with Val Acc: 0.9322
Epoch 25/40 | Train Loss: 0.1070 | Train Acc: 0.9608 | Val Loss: 0.2513 | Val Acc: 0.9206
Epoch 26/40 | Train Loss: 0.0986 | Train Acc: 0.9652 | Val Loss: 0.3208 | Val Acc: 0.9089
Epoch 27/40 | Train Loss: 0.0872 | Train Acc: 0.9688 | Val Loss: 0.2733 | Val Acc: 0.9159
Epoch 28/40 | Train Loss: 0.0922 | Train Acc: 0.9655 | Val Loss: 0.2379 | Val Acc: 0.9322
Epoch 29/40 | Train Loss: 0.0822 | Train Acc: 0.9712 | Val Loss: 0.2546 | Val Acc: 0.9282
Epoch 30/40 | Train Loss: 0.0620 | Train Acc: 0.9794 | Val Loss: 0.2498 | Val Acc: 0.9287
Epoch 31/40 | Train Loss: 0.0619 | Train Acc: 0.9783 | Val Loss: 0.2874 | Val Acc: 0.9246
Epoch 32/40 | Train Loss: 0.0579 | Train Acc: 0.9783 | Val Loss: 0.2813 | Val Acc: 0.9176
Epoch 33/40 | Train Loss: 0.0625 | Train Acc: 0.9772 | Val Loss: 0.2396 | Val Acc: 0.9340
Saved new best model at epoch 33 with Val Acc: 0.9340
Epoch 34/40 | Train Loss: 0.0517 | Train Acc: 0.9818 | Val Loss: 0.2701 | Val Acc: 0.9317
Epoch 35/40 | Train Loss: 0.0454 | Train Acc: 0.9838 | Val Loss: 0.3112 | Val Acc: 0.9276
Epoch 36/40 | Train Loss: 0.0440 | Train Acc: 0.9839 | Val Loss: 0.3021 | Val Acc: 0.9252
Epoch 37/40 | Train Loss: 0.0492 | Train Acc: 0.9828 | Val Loss: 0.2790 | Val Acc: 0.9276
Epoch 38/40 | Train Loss: 0.0296 | Train Acc: 0.9900 | Val Loss: 0.3180 | Val Acc: 0.9299
Epoch 39/40 | Train Loss: 0.0375 | Train Acc: 0.9864 | Val Loss: 0.3472 | Val Acc: 0.9246
Epoch 40/40 | Train Loss: 0.0448 | Train Acc: 0.9843 | Val Loss: 0.3898 | Val Acc: 0.9071
Loading best model from epoch 33 with Val Acc: 0.9340
Final Test Results using best validation checkpoint | Best Epoch: 33 | Best Val Acc: 0.9340 | Test Loss: 0.2729 | Test Acc: 0.9281
"""

RESNET_LOG = r"""
Using device: cuda
Epoch 1/30 | Train Loss: 0.3826 | Train Acc: 0.8669 | Val Loss: 0.5244 | Val Acc: 0.8137
Saved new best model at epoch 1 with Val Acc: 0.8137
Epoch 2/30 | Train Loss: 0.2391 | Train Acc: 0.9135 | Val Loss: 0.2702 | Val Acc: 0.9036
Saved new best model at epoch 2 with Val Acc: 0.9036
Epoch 3/30 | Train Loss: 0.1998 | Train Acc: 0.9300 | Val Loss: 0.3137 | Val Acc: 0.8908
Epoch 4/30 | Train Loss: 0.1674 | Train Acc: 0.9416 | Val Loss: 0.3167 | Val Acc: 0.8896
Epoch 5/30 | Train Loss: 0.1417 | Train Acc: 0.9509 | Val Loss: 0.2281 | Val Acc: 0.9223
Saved new best model at epoch 5 with Val Acc: 0.9223
Epoch 6/30 | Train Loss: 0.1257 | Train Acc: 0.9564 | Val Loss: 0.3318 | Val Acc: 0.8972
Epoch 7/30 | Train Loss: 0.1057 | Train Acc: 0.9622 | Val Loss: 0.2016 | Val Acc: 0.9299
Saved new best model at epoch 7 with Val Acc: 0.9299
Epoch 8/30 | Train Loss: 0.0838 | Train Acc: 0.9686 | Val Loss: 0.2055 | Val Acc: 0.9369
Saved new best model at epoch 8 with Val Acc: 0.9369
Epoch 9/30 | Train Loss: 0.0716 | Train Acc: 0.9732 | Val Loss: 0.1494 | Val Acc: 0.9433
Saved new best model at epoch 9 with Val Acc: 0.9433
Epoch 10/30 | Train Loss: 0.0647 | Train Acc: 0.9772 | Val Loss: 0.2246 | Val Acc: 0.9340
Epoch 11/30 | Train Loss: 0.0575 | Train Acc: 0.9790 | Val Loss: 0.1523 | Val Acc: 0.9533
Saved new best model at epoch 11 with Val Acc: 0.9533
Epoch 12/30 | Train Loss: 0.0641 | Train Acc: 0.9773 | Val Loss: 0.4681 | Val Acc: 0.8867
Epoch 13/30 | Train Loss: 0.0491 | Train Acc: 0.9835 | Val Loss: 0.2046 | Val Acc: 0.9422
Epoch 14/30 | Train Loss: 0.0414 | Train Acc: 0.9859 | Val Loss: 0.1939 | Val Acc: 0.9451
Epoch 15/30 | Train Loss: 0.0476 | Train Acc: 0.9846 | Val Loss: 0.2217 | Val Acc: 0.9352
Epoch 16/30 | Train Loss: 0.0304 | Train Acc: 0.9891 | Val Loss: 0.2185 | Val Acc: 0.9474
Epoch 17/30 | Train Loss: 0.0291 | Train Acc: 0.9905 | Val Loss: 0.1973 | Val Acc: 0.9509
Epoch 18/30 | Train Loss: 0.0370 | Train Acc: 0.9876 | Val Loss: 0.2531 | Val Acc: 0.9346
Epoch 19/30 | Train Loss: 0.0212 | Train Acc: 0.9933 | Val Loss: 0.1658 | Val Acc: 0.9597
Saved new best model at epoch 19 with Val Acc: 0.9597
Epoch 20/30 | Train Loss: 0.0367 | Train Acc: 0.9893 | Val Loss: 0.2674 | Val Acc: 0.9287
Epoch 21/30 | Train Loss: 0.0408 | Train Acc: 0.9866 | Val Loss: 0.2088 | Val Acc: 0.9527
Epoch 22/30 | Train Loss: 0.0215 | Train Acc: 0.9925 | Val Loss: 0.4651 | Val Acc: 0.9106
Epoch 23/30 | Train Loss: 0.0236 | Train Acc: 0.9912 | Val Loss: 0.2241 | Val Acc: 0.9457
Epoch 24/30 | Train Loss: 0.0119 | Train Acc: 0.9967 | Val Loss: 0.4007 | Val Acc: 0.9118
Epoch 25/30 | Train Loss: 0.0345 | Train Acc: 0.9890 | Val Loss: 0.2893 | Val Acc: 0.9352
Epoch 26/30 | Train Loss: 0.0212 | Train Acc: 0.9930 | Val Loss: 0.1887 | Val Acc: 0.9527
Epoch 27/30 | Train Loss: 0.0147 | Train Acc: 0.9951 | Val Loss: 0.1896 | Val Acc: 0.9480
Epoch 28/30 | Train Loss: 0.0333 | Train Acc: 0.9893 | Val Loss: 0.1963 | Val Acc: 0.9498
Epoch 29/30 | Train Loss: 0.0153 | Train Acc: 0.9952 | Val Loss: 0.2275 | Val Acc: 0.9480
Epoch 30/30 | Train Loss: 0.0221 | Train Acc: 0.9925 | Val Loss: 0.2551 | Val Acc: 0.9416
Loading best model from epoch 19 with Val Acc: 0.9597
Final Test Results using best validation checkpoint | Best Epoch: 19 | Best Val Acc: 0.9597 | Test Loss: 0.2167 | Test Acc: 0.9547
"""

VIT_LOG = r"""
Using device: cuda
Epoch 1/25 | Train Loss: 0.8009 | Train Acc: 0.6943 | Val Loss: 0.2913 | Val Acc: 0.9036
Saved new best model at epoch 1 with Val Acc: 0.9036
Epoch 2/25 | Train Loss: 0.2842 | Train Acc: 0.8987 | Val Loss: 0.4171 | Val Acc: 0.8668
Epoch 3/25 | Train Loss: 0.2112 | Train Acc: 0.9237 | Val Loss: 0.1480 | Val Acc: 0.9463
Saved new best model at epoch 3 with Val Acc: 0.9463
Epoch 4/25 | Train Loss: 0.1737 | Train Acc: 0.9392 | Val Loss: 0.1294 | Val Acc: 0.9527
Saved new best model at epoch 4 with Val Acc: 0.9527
Epoch 5/25 | Train Loss: 0.1492 | Train Acc: 0.9459 | Val Loss: 0.1653 | Val Acc: 0.9428
Epoch 6/25 | Train Loss: 0.1351 | Train Acc: 0.9513 | Val Loss: 0.2165 | Val Acc: 0.9293
Epoch 7/25 | Train Loss: 0.1184 | Train Acc: 0.9566 | Val Loss: 0.1970 | Val Acc: 0.9328
Epoch 8/25 | Train Loss: 0.1109 | Train Acc: 0.9602 | Val Loss: 0.1417 | Val Acc: 0.9533
Saved new best model at epoch 8 with Val Acc: 0.9533
Epoch 9/25 | Train Loss: 0.0985 | Train Acc: 0.9644 | Val Loss: 0.1161 | Val Acc: 0.9527
Epoch 10/25 | Train Loss: 0.0852 | Train Acc: 0.9706 | Val Loss: 0.0993 | Val Acc: 0.9679
Saved new best model at epoch 10 with Val Acc: 0.9679
Epoch 11/25 | Train Loss: 0.0794 | Train Acc: 0.9706 | Val Loss: 0.1264 | Val Acc: 0.9591
Epoch 12/25 | Train Loss: 0.0721 | Train Acc: 0.9745 | Val Loss: 0.1318 | Val Acc: 0.9544
Epoch 13/25 | Train Loss: 0.0629 | Train Acc: 0.9789 | Val Loss: 0.1694 | Val Acc: 0.9550
Epoch 14/25 | Train Loss: 0.0582 | Train Acc: 0.9784 | Val Loss: 0.1375 | Val Acc: 0.9556
Epoch 15/25 | Train Loss: 0.0534 | Train Acc: 0.9813 | Val Loss: 0.1411 | Val Acc: 0.9579
Epoch 16/25 | Train Loss: 0.0537 | Train Acc: 0.9808 | Val Loss: 0.1566 | Val Acc: 0.9521
Epoch 17/25 | Train Loss: 0.0465 | Train Acc: 0.9837 | Val Loss: 0.1470 | Val Acc: 0.9498
Epoch 18/25 | Train Loss: 0.0381 | Train Acc: 0.9857 | Val Loss: 0.1456 | Val Acc: 0.9603
Epoch 19/25 | Train Loss: 0.0516 | Train Acc: 0.9808 | Val Loss: 0.1242 | Val Acc: 0.9644
Epoch 20/25 | Train Loss: 0.0343 | Train Acc: 0.9884 | Val Loss: 0.1462 | Val Acc: 0.9626
Epoch 21/25 | Train Loss: 0.0331 | Train Acc: 0.9879 | Val Loss: 0.1876 | Val Acc: 0.9480
Epoch 22/25 | Train Loss: 0.0324 | Train Acc: 0.9892 | Val Loss: 0.1508 | Val Acc: 0.9556
Epoch 23/25 | Train Loss: 0.0394 | Train Acc: 0.9862 | Val Loss: 0.2140 | Val Acc: 0.9393
Epoch 24/25 | Train Loss: 0.0323 | Train Acc: 0.9887 | Val Loss: 0.1210 | Val Acc: 0.9690
Saved new best model at epoch 24 with Val Acc: 0.9690
Epoch 25/25 | Train Loss: 0.0216 | Train Acc: 0.9932 | Val Loss: 0.1339 | Val Acc: 0.9632
Loading best model from epoch 24 with Val Acc: 0.9690
Final Test Results using best validation checkpoint | Best Epoch: 24 | Best Val Acc: 0.9690 | Test Loss: 0.1517 | Test Acc: 0.9567
"""


def parse_training_log(log_text):
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
        r"Train Loss:\s+([0-9.]+)\s+\|\s+"
        r"Train Acc:\s+([0-9.]+)\s+\|\s+"
        r"Val Loss:\s+([0-9.]+)\s+\|\s+"
        r"Val Acc:\s+([0-9.]+)"
    )

    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for match in pattern.finditer(log_text):
        epochs.append(int(match.group(1)))
        train_losses.append(float(match.group(3)))
        train_accs.append(float(match.group(4)))
        val_losses.append(float(match.group(5)))
        val_accs.append(float(match.group(6)))

    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
    }


def plot_val_accuracy(data_by_model, output_path):
    plt.figure(figsize=(10, 6))

    for model_name, metrics in data_by_model.items():
        plt.plot(
            metrics["epochs"],
            metrics["val_accs"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

        best_acc = max(metrics["val_accs"])
        best_idx = metrics["val_accs"].index(best_acc)
        best_epoch = metrics["epochs"][best_idx]

        plt.scatter([best_epoch], [best_acc], s=60)
        

    plt.title("Validation Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
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

        plt.scatter([best_epoch], [best_loss], s=60)

    plt.title("Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    output_dir = "/scratch/aan266/project/src_classifying/visualizations/training_plots"
    os.makedirs(output_dir, exist_ok=True)

    data_by_model = {
        "Lightweight CNN": parse_training_log(CNN_LOG),
        "ResNet-18": parse_training_log(RESNET_LOG),
        "ViT-B/16": parse_training_log(VIT_LOG),
    }

    val_acc_path = os.path.join(output_dir, "bloodmnist_val_accuracy_vs_epoch.png")
    val_loss_path = os.path.join(output_dir, "bloodmnist_val_loss_vs_epoch.png")

    plot_val_accuracy(data_by_model, val_acc_path)
    plot_val_loss(data_by_model, val_loss_path)

    print(f"Saved validation accuracy plot to: {val_acc_path}")
    print(f"Saved validation loss plot to: {val_loss_path}")


if __name__ == "__main__":
    main()