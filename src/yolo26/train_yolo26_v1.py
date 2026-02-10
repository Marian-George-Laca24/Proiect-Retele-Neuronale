from ultralytics import YOLO

def main():
    # YOLO26 pretrained (baseline)
    model = YOLO("yolo26n.pt")

    model.train(
        data="data/yolo26/data.yaml",
        imgsz=1024,
        epochs=100,
        batch=-1,
        device=0,
        workers=0,
        project="results_yolo",
        name="train_yolo26_v1",
        plots=True,
        cache=False
    )

if __name__ == "__main__":
    main()
