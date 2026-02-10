from ultralytics import YOLO

def main():
    model = YOLO("results_yolo/train_v13/weights/last.pt")  # pornește de la last.pt
    model.train(
        data="data/data.yaml",
        imgsz=1024,
        epochs=200,
        batch=8,
        device=0,
        project="results_yolo",
        name="train_v14",
        plots=True,
        exist_ok=True,
        workers=0
    )

if __name__ == "__main__":
    main()
