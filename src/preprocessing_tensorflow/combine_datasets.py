import os
import json
import shutil
from glob import glob

# ===========================
#        CONFIG
# ===========================

RAW_DATA_DIR = "data/raw/"
OUTPUT_IMAGES_DIR = "data/processed/images/"
OUTPUT_COCO_JSON = "data/processed/annotations_combined.json"

# Clasele finale
FINAL_CATEGORIES = {
    "Scratch": 1,
    "Crack": 2,
    "Inclusion": 3,
    "Bubble": 4,
    "OK": 5
}

# Map pentru redenumirea claselor
CLASS_MAP = {
    "scratch": "Scratch",
    "scratches": "Scratch",

    "crack": "Crack",
    "cracks": "Crack",

    "bubble": "Bubble",
    "bubbles": "Bubble",

    "inclusion": "Inclusion",
    "inclusions": "Inclusion",
    "impurity": "Inclusion",
}

# ===========================
#        UTILITY FUNCS
# ===========================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def normalize_class(name):
    name = name.lower().strip()
    return CLASS_MAP.get(name, None)

# ===========================
#        MAIN FUNCTION
# ===========================

def combine_datasets():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    final_coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # adaugă categoriile
    for name, cid in FINAL_CATEGORIES.items():
        final_coco["categories"].append({"id": cid, "name": name})

    image_id = 1
    annotation_id = 1

    # detectează TOATE fișierele COCO (_annotations.coco, _annotations.coco.json, etc.)
    coco_files = glob(os.path.join(RAW_DATA_DIR, "**/_annotations.coco*"), recursive=True)

    print(f"Found {len(coco_files)} annotation files.\n")

    # ===========================================
    #       PARCURGEM TOATE DATASET-URILE
    # ===========================================
    for coco_path in coco_files:
        print(f"Processing: {coco_path}")

        coco = load_json(coco_path)

        # folderul cu imaginile sale
        base_folder = os.path.dirname(coco_path)

        # Map pentru id-urile imaginii
        local_to_global_img_id = {}

        # ===========================
        #      IMAGINI
        # ===========================
        for img in coco.get("images", []):
            old_name = img["file_name"]
            old_path = os.path.join(base_folder, old_name)

            if not os.path.exists(old_path):
                print(f"Warning: Missing image {old_path}")
                continue

            new_name = f"img_{image_id}.jpg"
            new_path = os.path.join(OUTPUT_IMAGES_DIR, new_name)

            shutil.copy(old_path, new_path)

            final_coco["images"].append({
                "id": image_id,
                "file_name": new_name,
                "width": img.get("width", 256),
                "height": img.get("height", 256)
            })

            local_to_global_img_id[img["id"]] = image_id
            image_id += 1

        # ===========================
        #      ADNOTĂRI
        # ===========================
        for ann in coco.get("annotations", []):
            old_cat_id = ann["category_id"]

            # găsim categoria din fișierul original
            category_name = None
            for cat in coco.get("categories", []):
                if cat["id"] == old_cat_id:
                    category_name = normalize_class(cat["name"])
                    break

            if category_name is None:
                # ignorăm clasele care nu sunt în map
                continue

            final_coco["annotations"].append({
                "id": annotation_id,
                "image_id": local_to_global_img_id.get(ann["image_id"]),
                "category_id": FINAL_CATEGORIES[category_name],
                "bbox": ann["bbox"],
                "iscrowd": ann.get("iscrowd", 0),
                "area": ann.get("area", 0)
            })

            annotation_id += 1

    # ===========================================
    #             CLASA OK (FĂRĂ DEFECTE)
    # ===========================================
    ok_folder = os.path.join(RAW_DATA_DIR, "ok")

    if os.path.exists(ok_folder):
        print("\nAdding OK images...")
        for fname in os.listdir(ok_folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                old_path = os.path.join(ok_folder, fname)
                new_name = f"img_{image_id}.jpg"
                new_path = os.path.join(OUTPUT_IMAGES_DIR, new_name)

                shutil.copy(old_path, new_path)

                # imagine fără adnotări → clasa OK
                final_coco["images"].append({
                    "id": image_id,
                    "file_name": new_name,
                    "width": 256,
                    "height": 256
                })

                image_id += 1

    # ===========================================
    #     SALVARE FIȘIER COCO FINAL
    # ===========================================

    os.makedirs(os.path.dirname(OUTPUT_COCO_JSON), exist_ok=True)

    with open(OUTPUT_COCO_JSON, "w") as f:
        json.dump(final_coco, f, indent=4)

    print("\n===============================")
    print("   Dataset combinat cu succes!")
    print("===============================")
    print(f"Total imagini: {len(final_coco['images'])}")
    print(f"Total adnotări: {len(final_coco['annotations'])}")
    print(f"Fișier COCO final: {OUTPUT_COCO_JSON}")
    print("===============================\n")

# ===========================
#       EXECUȚIE SCRIPT
# ===========================

if __name__ == "__main__":
    combine_datasets()
