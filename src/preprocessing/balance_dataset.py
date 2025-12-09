import os
import random
import shutil

SOURCE_DIR = "data/classes"
TARGET_DIR = "data/classes_balanced"
MAX_PER_CLASS = 300  # cât vrei din fiecare clasă

os.makedirs(TARGET_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    src_class = os.path.join(SOURCE_DIR, class_name)
    dst_class = os.path.join(TARGET_DIR, class_name)

    os.makedirs(dst_class, exist_ok=True)

    images = os.listdir(src_class)

    if len(images) > MAX_PER_CLASS:
        images = random.sample(images, MAX_PER_CLASS)

    for img in images:
        shutil.copy(
            os.path.join(src_class, img),
            os.path.join(dst_class, img)
        )

    print(f"{class_name}: {len(images)} imagini păstrate")

print("\n✅ Dataset BALANCED creat cu succes!")
