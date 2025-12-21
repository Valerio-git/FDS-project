import os
import json
import random
from typing import Dict, List, Tuple

def _collect_class_files(root_dir: str, white: bool) -> Dict[str, List[str]]:
    
    # Returns dict: class_name -> file list (absolute or relative path)
    
    classes = sorted(os.listdir(root_dir))
    per_class_files = {}

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)

        if white:
            subfolders = ["real_world"]
        else:
            subfolders = ["default", "real_world"]

        files = []
        for sub in subfolders:
            sub_dir = os.path.join(class_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            names = sorted(os.listdir(sub_dir))
            for n in names:
                files.append(os.path.join(sub_dir, n)) # for each file's name it stores their path into files

        per_class_files[class_name] = files # dict in which to each class corresponds a list of every path of each file belonging to subfolders of that class

    return per_class_files


def create_or_load_splits(
    root_dir: str,
    split_path: str,
    white: bool,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2), # percentage of data in each split
) -> Dict:
    """
    If split does not exist, creates it and saves it to `split_path`.
    If it exists, loads and returns it.

    JSON structure:
    {
      "meta": {...},
      "classes": [...],
      "splits": {
        "train": [{"path": "...", "label": 0}, ...],
        "val":   [{"path": "...", "label": 0}, ...],
        "test":  [{"path": "...", "label": 0}, ...]
      }
    }
    """
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f) # if the split path already exists it just load it 

    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    r_train, r_val, r_test = ratios
    if abs((r_train + r_val + r_test) - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    per_class = _collect_class_files(root_dir, white=white)
    classes = sorted(per_class.keys())

    rng = random.Random(seed)

    out = {
        "meta": {
            "root_dir": root_dir,
            "white": bool(white),
            "seed": seed,
            "ratios": {"train": r_train, "val": r_val, "test": r_test},
        },
        "classes": classes,
        "splits": {"train": [], "val": [], "test": []},
    }

    for label, class_name in enumerate(classes):
        files = list(per_class[class_name])
        rng.shuffle(files)

        n = len(files)
        a = int(r_train * n)
        b = int((r_train + r_val) * n)

        train_files = files[:a]
        val_files = files[a:b]
        test_files = files[b:]

        out["splits"]["train"].extend([{"path": p, "label": label} for p in train_files])
        out["splits"]["val"].extend([{"path": p, "label": label} for p in val_files])
        out["splits"]["test"].extend([{"path": p, "label": label} for p in test_files])

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out