import os
import json
import matplotlib.pyplot as plt

V13_JSON = "results/yolo_test_metrics_v13.json"
V14_JSON = "results/yolo_test_metrics_v14.json"
OUT_PATH = "docs/results/metrics_evolution.png"

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["metrics"]

def main():
    if not os.path.exists(V13_JSON):
        raise FileNotFoundError(f"Nu găsesc: {V13_JSON}")
    if not os.path.exists(V14_JSON):
        raise FileNotFoundError(f"Nu găsesc: {V14_JSON}")

    m13 = load_metrics(V13_JSON)
    m14 = load_metrics(V14_JSON)

    labels = ["v13", "v14"]
    map5095 = [m13["mAP50_95"], m14["mAP50_95"]]
    map50   = [m13["mAP50"],    m14["mAP50"]]
    prec    = [m13["precision_mean"], m14["precision_mean"]]
    rec     = [m13["recall_mean"],    m14["recall_mean"]]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(labels, map5095, marker="o", label="mAP50-95")
    plt.plot(labels, map50,   marker="o", label="mAP50")
    plt.plot(labels, prec,    marker="o", label="Precision (mean)")
    plt.plot(labels, rec,     marker="o", label="Recall (mean)")
    plt.ylim(0, 1)
    plt.xlabel("Experiment")
    plt.ylabel("Score")
    plt.title("Metrics evolution (v13 -> v14)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.show()

    print(f" metrics_evolution.png salvat în: {OUT_PATH}")

if __name__ == "__main__":
    main()
