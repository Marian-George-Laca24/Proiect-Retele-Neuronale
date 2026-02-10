from ultralytics import YOLO

def main():
    # pornește din best-ul deja obținut
    model = YOLO(r"runs/detect/results_yolo/train_yolo26_v1/weights/best.pt")

    model.train(
        data=r"data/yolo26/data.yaml",
        imgsz=1024,
        epochs=120,
        batch=-1,          # auto batch (în funcție de VRAM)
        device=0,
        workers=0,
        project="results_yolo",
        name="train_yolo26_v2_finetune_best",
        exist_ok=True,
        plots=True,
        cache=False,

        # fine-tuning mai "fin": LR mai mic
        lr0=5e-4,
        lrf=1e-2,

        # augmentări utile pt defecte fine + generalizare
        mixup=0.05,
        copy_paste=0.10,
        close_mosaic=10,   # oprește mosaic spre final (reduce artefacte)
        patience=40
    )

if __name__ == "__main__":
    main()
