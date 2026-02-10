import os
import json
import shutil
from tqdm import tqdm

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMAGES_COMBINED = os.path.join(PROCESSED_DIR, "images")
ANNOT_FILE = os.path.join(PROCESSED_DIR, "annotations_combined.json")
EXPORT_DIR = "data/classes"

# Mapări între clase și datasetul lor raw
RAW_CLASS_FOLDERS = {
    "scratch": "Glass Defect Detection-scratch.v3i.coco",
    "inclusion": "Inclusion.v1i.coco",
    "bubble": "Glass Dataset Bubbles.v2i.coco",
    "crack": "crack.v2i.coco",
    "ok": "ok"
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def export_classes():
    print("Loading annotations...")
    with open(ANNOT_FILE, "r") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"].lower() for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    # Creeăm folderele finale
    ensure_dir(EXPORT_DIR)
    for c in RAW_CLASS_FOLDERS.keys():
        ensure_dir(os.path.join(EXPORT_DIR, c))

    # Adnotări per clasă
    cls_image_ids = {c: set() for c in RAW_CLASS_FOLDERS.keys()}

    # Distribuie imaginile după categoria lor
    for ann in coco["annotations"]:
        cls = categories[ann["category_id"]]
        if cls in cls_image_ids:
            cls_image_ids[cls].add(ann["image_id"])

    print("\nExporting CRACK + BUBBLE from annotations...")

    for cls in ["crack", "bubble"]:
        ids = list(cls_image_ids[cls])
        print(f"\nClass: {cls} ({len(ids)} images found in COCO)")

        for img_id in tqdm(ids):
            img = images[img_id]
            src = os.path.join(IMAGES_COMBINED, img["file_name"])
            dst = os.path.join(EXPORT_DIR, cls, img["file_name"])
            if os.path.exists(src):
                shutil.copy(src, dst)

    print("\nExporting SCRATCH + INCLUSION from raw dataset folders (no annotations)...")

    for cls in ["scratch", "inclusion"]:
        raw_folder = os.path.join(RAW_DIR, RAW_CLASS_FOLDERS[cls], "train")
        if not os.path.exists(raw_folder):
            continue

        files = [f for f in os.listdir(raw_folder) if f.lower().endswith((".jpg",".png"))]
        print(f"\nClass: {cls} ({len(files)} images found in raw dataset)")

        for f in tqdm(files):
            shutil.copy(os.path.join(raw_folder, f), os.path.join(EXPORT_DIR, cls, f))

    print("\nExporting OK images...")

    ok_folder = os.path.join(RAW_DIR, RAW_CLASS_FOLDERS["ok"])
    ok_files = [f for f in os.listdir(ok_folder) if f.lower().endswith((".jpg",".png"))]

    print(f"Class: OK ({len(ok_files)} images)")

    for f in tqdm(ok_files):
        shutil.copy(os.path.join(ok_folder, f), os.path.join(EXPORT_DIR, "ok", f))

    print("\n===============================")
    print(" ALL CLASSES EXPORTED SUCCESSFULLY!")
    print(" Output folder: data/classes/")
    print("===============================")


if __name__ == "__main__":
    export_classes()
