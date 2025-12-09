import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = "data/classes_balanced"
TARGET_DIR = "data/split_balanced"

os.makedirs(TARGET_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    train_imgs, temp = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)

    for split, img_set in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        dst_dir = os.path.join(TARGET_DIR, split, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        for img in img_set:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(dst_dir, img)
            )

    print(f"{class_name}: Train {len(train_imgs)}, Val {len(val_imgs)}, Test {len(test_imgs)}")

print("\n✅ Dataset BALANCED SPLIT în train/val/test & gata de antrenare!")
