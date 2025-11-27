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

# Directory for model predictions (not tracked by git, under data/processed).
PREDICTIONS_DIR = DATA_DIR / "processed" / "predictions"


def get_sam3_yolo_predictions_dir(split: str) -> Path:
    """
    Return the directory where YOLO-style predictions produced by SAM 3
    will be stored for a given split (train, val, test).
    """
    split = _check_split(split)
    return PREDICTIONS_DIR / "sam3_yolo" / split

# Directory for segmentation outputs.
SEGMENTATIONS_DIR = DATA_DIR / "processed" / "segmentations"


def get_sam3_segmentation_dir(split: str) -> Path:
    """
    Return the directory where SAM 3 segmentation masks will be stored
    for a given split (train, val, test).
    """
    split = _check_split(split)
    return SEGMENTATIONS_DIR / "sam3" / split
