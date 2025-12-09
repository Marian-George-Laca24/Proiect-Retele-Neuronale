import os
import shutil
import random
from tqdm import tqdm

# Setări
SOURCE_DIR = "data/classes"        # aici sunt folderele tale crack/scratch...
TARGET_DIR = "data/split"          # aici vom crea structura finală

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_files(file_list):
    random.shuffle(file_list)
    n = len(file_list)

    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    return (
        file_list[:train_end],
        file_list[train_end:val_end],
        file_list[val_end:]
    )

def process_class(class_name):
    print(f"\nProcessing class: {class_name}")

    source_path = os.path.join(SOURCE_DIR, class_name)
    files = [f for f in os.listdir(source_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(files) == 0:
        print(f"⚠ No images found for class {class_name}!")
        return

    # împărțire
    train_files, val_files, test_files = split_files(files)

    # creăm folderele
    for split in ["train", "val", "test"]:
        create_dir(os.path.join(TARGET_DIR, class_name, split))

    # copiere imagini
    def copy_files(file_list, split):
        dest_dir = os.path.join(TARGET_DIR, class_name, split)
        for f in tqdm(file_list, desc=f"{class_name} -> {split}"):
            shutil.copy(os.path.join(source_path, f), os.path.join(dest_dir, f))

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"✔ Done: {class_name} ({len(files)} images split)")

def main():
    print("=== SPLITTING DATASET INTO TRAIN / VAL / TEST ===")

    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    for cls in classes:
        process_class(cls)

    print("\n========================================")
    print(" SPLIT COMPLETE!")
    print(f" Output folder: {TARGET_DIR}/")
    print("========================================")

if __name__ == "__main__":
    main()
