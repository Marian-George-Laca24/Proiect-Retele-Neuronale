from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data/data.yaml",
        imgsz=1024,
        epochs=50,
        batch=8,
        device=0,
        workers=0,      
        cache=False, 
        project="results_yolo",
        name="train_v1",
        plots=True
    )

if __name__ == "__main__":
    main()

