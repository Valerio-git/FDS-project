import os
import shutil
import json
from pathlib import Path

DATASETS_ROOT = "Datasets"
MAPPING_FILE = "mapping.json"

# Upload the mapping
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)

def safe_move_with_rename(src, dst_dir, prefix):
    """
    Move the file from `src` to `dst_dir`, renaming it
    based on the prefix (type of item) and a progressive counter.
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    ext = Path(src).suffix
    counter = 1

    while True:
        new_name = f"{prefix}_{counter:06d}{ext}"
        dst = os.path.join(dst_dir, new_name)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            break
        counter += 1


def reorganize(dataset_path):
    print(f"\n📂 Processing dataset: {dataset_path}")

    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if not os.path.isdir(item_path):
            continue
        
        if item not in mapping:
            print(f"⚠️ '{item}' not in mapping -> SKIP")
            continue

        category = mapping[item]
        category_path = os.path.join(dataset_path, category)
        os.makedirs(category_path, exist_ok=True)

        # default / real_world

        for sub in ["default", "real_world"]:
            original_sub = os.path.join(item_path, sub)
            if os.path.isdir(original_sub):
                dest_sub = os.path.join(category_path, sub)

                for file in os.listdir(original_sub):
                    src_file = os.path.join(original_sub, file)
                    safe_move_with_rename(src_file, dest_sub, item)

        # Delete empty item folder
        shutil.rmtree(item_path)

    print(f"✔️ COMPLETED: {dataset_path}")

if __name__ == "__main__":
    reorganize(os.path.join(DATASETS_ROOT, "raw_dataset"))
    reorganize(os.path.join(DATASETS_ROOT, "white_dataset"))
