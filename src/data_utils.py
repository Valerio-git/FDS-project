import os
from pathlib import Path

try:
    import kagglehub
except ImportError:
    kagglehub = None


# === Kaggle dataset ===
KAGGLE_DATASET = "tommasosbrenna/recyclable-and-household-waste-white-background"

# === Cartella madre ===
BASE_DIR = Path("Datasets")

# === Sotto-cartelle ===
WHITE_SUBDIR = "white_dataset"
RAW_SUBDIR = "raw_dataset"


def _ensure_kaggle_available():
    """If kagglehub is not installed, raises an error explaining how to install it."""
    if kagglehub is None:
        raise RuntimeError(
            "Dataset not found locally.\n"
            "Install kagglehub with:  pip install kagglehub\n"
            "and configure your Kaggle API key to download it automatically."
        )


def _download_dataset_if_needed() -> Path:
    
    if BASE_DIR.exists():
        return BASE_DIR

    _ensure_kaggle_available()

    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
    source_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Dataset downloaded to cache at:", source_path)

    
    os.makedirs(BASE_DIR, exist_ok=True)

    
    return BASE_DIR


def get_white_dataset_path() -> str:
    
    _download_dataset_if_needed()
    white_path = BASE_DIR / WHITE_SUBDIR

    if not white_path.exists():
        raise FileNotFoundError(
            f"The folder '{white_path}' does not exist.\n"
            "Make sure the Kaggle dataset contains the 'white_dataset/' subfolder."
        )

    return str(white_path)


def get_raw_dataset_path() -> str:
    
    _download_dataset_if_needed()
    raw_path = BASE_DIR / RAW_SUBDIR

    if not raw_path.exists():
        raise FileNotFoundError(
            f"The folder '{raw_path}' does not exist.\n"
            "Make sure the Kaggle dataset contains the 'raw_dataset/' subfolder."
        )

    return str(raw_path)



if __name__ == "__main__":
    print("White dataset path:", get_white_dataset_path())
    print("Raw dataset path:", get_raw_dataset_path())
