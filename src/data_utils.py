import os
from pathlib import Path

try:
    import kagglehub
except ImportError:
    kagglehub = None


KAGGLE_DATASET = "tommasosbrenna/recyclable-and-household-waste-white-background"


def get_dataset_path() -> str:
    """
    Restituisce il path locale del dataset.
    - Se esiste una cartella data/ locale, usa quella.
    - Altrimenti prova a scaricarlo da Kaggle con kagglehub.
    """

    # 1. Se l'utente ha già un data/ locale, usiamo quello
    local_data_dir = Path("white_dataset")
    if local_data_dir.exists():
        return str(local_data_dir)

    # 2. Se kagglehub non c'è, spiego cosa fare
    if kagglehub is None:
        raise RuntimeError(
            "Dataset not found. Install kagglehub (`pip install kagglehub`) "
            "and configure your Kaggle API key to download it automatically."
        )

    # 3. Usa kagglehub per scaricare (solo la prima volta farà il download vero)
    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle via kagglehub...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Dataset downloaded / found in cache at:", path)
    return path

print("Dataset path:", get_dataset_path())
