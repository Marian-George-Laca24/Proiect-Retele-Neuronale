from ultralytics import YOLO

def main():
    model = YOLO(
        "runs/detect/results_yolo/train_yolo26_v3_yolo26s/weights/best.pt"
    )

    model.train(
        data="data/yolo26/data.yaml",
        imgsz=1024,
        epochs=40,            
        batch=8,
        device=0,
        workers=0,

        project="results_yolo",
        name="train_yolo26_v5_finetune_yolo26s",
        exist_ok=True,

        lr0=3e-4,              
        lrf=1e-2,
        close_mosaic=0,     
        patience=15,

        plots=True,
        cache=False
    )

if __name__ == "__main__":
    main()
