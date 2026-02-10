from ultralytics import YOLO

def main():
    model = YOLO(
        "runs/detect/results_yolo/train_yolo26_v3_yolo26s/weights/best.pt"
    )

    model.train(
        data="data/yolo26/data.yaml",
        imgsz=1024,
        epochs=50,
        batch=8,
        device=0,
        workers=0,
        project="results_yolo",
        name="train_yolo26_v4_finetune_yolo26s",
        lr0=1e-4,        # learning rate mic → fine-tuning real
        patience=20,     # early stopping
        plots=True
    )

if __name__ == "__main__":
    main()
