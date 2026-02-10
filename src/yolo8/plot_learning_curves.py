import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results_yolo/train_v14/results.csv"
OUT_PATH = "docs/results/learning_curves_final.png"

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Nu găsesc: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    epoch = "epoch"

    # Coloane LOSS
    train_box = "train/box_loss"
    val_box   = "val/box_loss"

    # Coloane mAP (la tine au sufixul (B))
    map50   = "metrics/mAP50(B)"
    map5095 = "metrics/mAP50-95(B)"

    # Asigură folder output
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # FIGURA 1: Loss curves
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df[epoch], df[train_box], label="Train box loss")
    plt.plot(df[epoch], df[val_box], label="Val box loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Box Loss")
    plt.legend()

    # FIGURA 2: mAP curves
    plt.subplot(1, 2, 2)
    plt.plot(df[epoch], df[map50], label="mAP@50")
    plt.plot(df[epoch], df[map5095], label="mAP@50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP Evolution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.show()

    print(f"   learning_curves_final.png salvat în: {OUT_PATH}")
    print(f"   Folosite coloane: {map50} și {map5095}")

if __name__ == "__main__":
    main()
