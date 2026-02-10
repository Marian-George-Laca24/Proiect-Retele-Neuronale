import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 inference on a single image.")
    p.add_argument("--model", type=str, default="models/yolo/best.pt", help="Path to .pt model")
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    p.add_argument("--device", type=str, default="0", help="0 for GPU, 'cpu' for CPU")
    p.add_argument("--save_dir", type=str, default="docs/screenshots", help="Where to save output")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False
    )

    # Save annotated image (Ultralytics helper)
    annotated = results[0].plot()  # numpy array (BGR)
    out_path = save_dir / f"yolo_pred_{image_path.stem}.jpg"

    # write with OpenCV only if available; fallback to PIL
    try:
        import cv2
        cv2.imwrite(str(out_path), annotated)
    except Exception:
        from PIL import Image
        import numpy as np
        rgb = annotated[:, :, ::-1]  # BGR->RGB
        Image.fromarray(rgb.astype(np.uint8)).save(out_path)

    # Print detections summary
    boxes = results[0].boxes
    names = results[0].names

    print("=== YOLO PREDICTION SUMMARY ===")
    if boxes is None or len(boxes) == 0:
        print("OK - no detections above threshold.")
    else:
        for i, b in enumerate(boxes, start=1):
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            print(f"{i}. {names.get(cls_id, cls_id)} conf={conf:.3f} box=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})")

    print(f"\nSaved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
