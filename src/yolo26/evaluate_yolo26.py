import json
from pathlib import Path
from ultralytics import YOLO

def main():
    weights = r"results_yolo\train_yolo26_v5_finetune_yolo26s/weights/best.pt"
    data_yaml = r"data/yolo26/data.yaml"

    model = YOLO(weights)

    r = model.val(
        data=data_yaml,
        imgsz=1024,
        device=0,
        plots=True,   # salvează grafice/CM în folderul de run
        save_json=True
    )

    out = {
        "model": weights,
        "data": data_yaml,
        "imgsz": 1024,
        "metrics": {
            "mAP50_95": float(r.box.map),
            "mAP50": float(r.box.map50),
            "mAP75": float(r.box.map75),
            "precision_mean": float(r.box.mp),
            "recall_mean": float(r.box.mr),
        }
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/yolo26s_best_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nSaved -> results/yolo26s_best_metrics.json")
    print(json.dumps(out["metrics"], indent=2))

if __name__ == "__main__":
    main()
