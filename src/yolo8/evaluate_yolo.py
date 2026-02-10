from pathlib import Path
import json
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    # Ajustează aici dacă ai altă locație
    DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"

    # Rule-uri de evaluat (best.pt din fiecare run sau direct weights/best.pt din run)
    RUNS = {
        "v13": PROJECT_ROOT / "results_yolo" / "train_v13" / "weights" / "best.pt",
        "v14": PROJECT_ROOT / "results_yolo" / "train_v14" / "weights" / "best.pt",
    }

    OUT_DIR = PROJECT_ROOT / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    imgsz = 1024
    device = 0  # GPU; pune "cpu" dacă vrei CPU

    for tag, model_path in RUNS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Nu găsesc modelul: {model_path}")

        model = YOLO(str(model_path))

        # Evaluare pe split-ul test (Ultralytics folosește data.yaml + split=test)
        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=imgsz,
            split="test",
            device=device,
            verbose=False
        )

        # Extragem valori importante într-un JSON simplu
        # metrics.box.map -> mAP50-95
        # metrics.box.map50 -> mAP50
        # metrics.box.map75 -> mAP75
        # metrics.box.mp -> mean precision
        # metrics.box.mr -> mean recall
        out = {
            "task": "detect",
            "imgsz": imgsz,
            "model": str(model_path.relative_to(PROJECT_ROOT)),
            "data": str(DATA_YAML.relative_to(PROJECT_ROOT)),
            "metrics": {
                "mAP50_95": float(metrics.box.map),
                "mAP50": float(metrics.box.map50),
                "mAP75": float(metrics.box.map75),
                "precision_mean": float(metrics.box.mp),
                "recall_mean": float(metrics.box.mr),
            }
        }

        out_path = OUT_DIR / f"yolo_test_metrics_{tag}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
