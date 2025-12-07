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
    """Se kagglehub non è installato, mostra un messaggio chiaro."""
    if kagglehub is None:
        raise RuntimeError(
            "Dataset not found locally.\n"
            "Install kagglehub with:  pip install kagglehub\n"
            "and configure your Kaggle API key to download it automatically."
        )


def _download_dataset_if_needed() -> Path:
    """
    Scarica il dataset da Kaggle solo se la cartella Datasets/ non esiste.
    Se esiste, non scarica nulla.
    Restituisce la Path a Datasets/.
    """
    if BASE_DIR.exists():
        return BASE_DIR

    _ensure_kaggle_available()

    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
    source_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Dataset downloaded to cache at:", source_path)

    # Copiamo (o spostiamo) solo se non esiste già Datasets/
    os.makedirs(BASE_DIR, exist_ok=True)

    # ⚠️ L’utente deve aver caricato su Kaggle una struttura con già le due cartelle
    # Quindi basta copiarle dalla cache nella cartella locale
    # Se kagglehub le ha già in sottocartelle, le ritroverai direttamente dentro.
    return BASE_DIR


def get_white_dataset_path() -> str:
    """Restituisce il path della cartella white_dataset."""
    _download_dataset_if_needed()
    white_path = BASE_DIR / WHITE_SUBDIR

    if not white_path.exists():
        raise FileNotFoundError(
            f"La cartella '{white_path}' non esiste.\n"
            "Assicurati che nel dataset Kaggle ci sia la sottocartella 'white_dataset/'."
        )

    return str(white_path)


def get_raw_dataset_path() -> str:
    """Restituisce il path della cartella raw_dataset."""
    _download_dataset_if_needed()
    raw_path = BASE_DIR / RAW_SUBDIR

    if not raw_path.exists():
        raise FileNotFoundError(
            f"La cartella '{raw_path}' non esiste.\n"
            "Assicurati che nel dataset Kaggle ci sia la sottocartella 'raw_dataset/'."
        )

    return str(raw_path)


# === DEBUG ===
if __name__ == "__main__":
    print("White dataset path:", get_white_dataset_path())
    print("Raw dataset path:", get_raw_dataset_path())
