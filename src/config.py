from pathlib import Path

# Root del progetto (dove ci sono README.md, data/, src/, ecc.)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Cartelle dati
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Percorsi specifici del dataset
DATASET_YAML_PATH = RAW_DATA_DIR / "data.yaml"
IMAGES_ROOT = RAW_DATA_DIR / "images"
LABELS_ROOT = RAW_DATA_DIR / "labels"


def _check_split(split: str) -> str:
    """Verifica che lo split sia valido."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Split non valido: {split!r} (usa 'train', 'val' o 'test')")
    return split


def get_images_dir(split: str):
    """Ritorna la cartella immagini per lo split dato."""
    split = _check_split(split)
    return IMAGES_ROOT / split


def get_labels_dir(split: str):
    """Ritorna la cartella labels per lo split dato."""
    split = _check_split(split)
    return LABELS_ROOT / split
