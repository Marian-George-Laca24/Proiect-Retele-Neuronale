from ultralytics import YOLO

def main():
    model = YOLO("yolo26s.pt")  

    model.train(
        data=r"data/yolo26/data.yaml",
        imgsz=1024,
        epochs=120,
        batch=8,          
        device=0,
        workers=0,
        project="results_yolo",
        name="train_yolo26_v3_yolo26s",
        exist_ok=True,
        plots=True,
        cache="ram",

        lr0 = 5e-4,
        lrf = 5e-3,
        close_mosaic=20,
        patience=50,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,
        translate=0.05,
        scale=0.5,
        fliplr=0.5
    )

if __name__ == "__main__":
    main()
