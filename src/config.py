from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

GRAPH_DIR = RAW_DIR / "wikispeedia_paths-and-graph"
PLAINTEXT_DIR = RAW_DIR / "plaintext_articles"

FIGURES_DIR = ROOT_DIR / "reports" / "figures"

