from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def f1_from_pr(p, r):
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)

def load_metrics_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    m = d["metrics"]
    p = float(m["precision_mean"])
    r = float(m["recall_mean"])
    return {
        "tag": path.stem.replace("yolo_test_metrics_", ""),
        "mAP50_95": float(m["mAP50_95"]),
        "precision_mean": p,
        "recall_mean": r,
        "f1_mean": float(f1_from_pr(p, r)),
    }

def find_first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    # 1) Input JSON-uri (v13/v14)
    j_v13 = PROJECT_ROOT / "results" / "yolo_test_metrics_v13.json"
    j_v14 = PROJECT_ROOT / "results" / "yolo_test_metrics_v14.json"

    # 2) Run final (learning curves) - numele tău e train_v14
    run_final_csv = PROJECT_ROOT / "results_yolo" / "train_v14" / "results.csv"

    # Output
    out_dir = PROJECT_ROOT / "docs" / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # A) Grafice comparative (din JSON)
    # -------------------------
    rows = []
    for p in [j_v13, j_v14]:
        if not p.exists():
            raise FileNotFoundError(f"Lipsește JSON: {p}")
        rows.append(load_metrics_json(p))

    dfm = pd.DataFrame(rows).sort_values("tag")
    # accuracy_comparison.png (mAP50_95)
    plt.figure()
    plt.bar(dfm["tag"], dfm["mAP50_95"])
    plt.title("YOLO - mAP50-95 pe test set (comparativ experimente)")
    plt.ylabel("mAP50-95")
    plt.xlabel("Experiment")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_comparison.png", dpi=200)
    plt.close()

    # f1_comparison.png (F1_mean)
    plt.figure()
    plt.bar(dfm["tag"], dfm["f1_mean"])
    plt.title("YOLO - F1_mean (din Precision/Recall) pe test set")
    plt.ylabel("F1_mean")
    plt.xlabel("Experiment")
    plt.tight_layout()
    plt.savefig(out_dir / "f1_comparison.png", dpi=200)
    plt.close()

    # -------------------------
    # B) Learning curves pentru modelul final (din results.csv)
    # -------------------------
    if not run_final_csv.exists():
        raise FileNotFoundError(f"Lipsește results.csv pentru run final: {run_final_csv}")

    df = pd.read_csv(run_final_csv)

    # Coloane posibile la Ultralytics (diferă ușor între versiuni)
    col_epoch = find_first_col(df, ["epoch", "Epoch"])
    col_map = find_first_col(df, [
        "metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(Box)",
        "metrics/mAP50-95(M)", "metrics/mAP50-95(B)"
    ])

    # Loss total (sumă box+cls+dfl) dacă există; altfel ia ce găsește
    train_box = find_first_col(df, ["train/box_loss"])
    train_cls = find_first_col(df, ["train/cls_loss"])
    train_dfl = find_first_col(df, ["train/dfl_loss"])

    val_box = find_first_col(df, ["val/box_loss"])
    val_cls = find_first_col(df, ["val/cls_loss"])
    val_dfl = find_first_col(df, ["val/dfl_loss"])

    if col_epoch is None:
        # fallback: index
        df["epoch_fallback"] = range(len(df))
        col_epoch = "epoch_fallback"

    # Construim loss total dacă avem componente
    if train_box and train_cls and train_dfl:
        df["train/total_loss"] = df[train_box] + df[train_cls] + df[train_dfl]
        train_total = "train/total_loss"
    else:
        train_total = train_box or train_cls or train_dfl  # orice există

    if val_box and val_cls and val_dfl:
        df["val/total_loss"] = df[val_box] + df[val_cls] + df[val_dfl]
        val_total = "val/total_loss"
    else:
        val_total = val_box or val_cls or val_dfl

    plt.figure()
    if train_total:
        plt.plot(df[col_epoch], df[train_total], label="train_loss")
    if val_total:
        plt.plot(df[col_epoch], df[val_total], label="val_loss")
    if col_map:
        plt.plot(df[col_epoch], df[col_map], label="mAP50-95(test/val)")
    plt.title("Learning curves - run final (train_v14)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves_best.png", dpi=200)
    plt.close()

    print(f"[OK] Grafice generate în: {out_dir}")

if __name__ == "__main__":
    main()
