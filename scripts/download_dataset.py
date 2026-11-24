import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import GRAPH_DIR, PLAINTEXT_DIR, RAW_DIR

ARCHIVES = [
    (
        "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz",
        GRAPH_DIR,
    ),
    (
        "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_plaintext.tar.gz",
        PLAINTEXT_DIR,
    ),
]

try:
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, blocks=1, block_size=1, total_size=None):
            if total_size is not None:
                self.total = total_size
            self.update(blocks * block_size - self.n)
except ImportError:
    class DownloadProgressBar:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def update_to(self, blocks=1, block_size=1, total_size=None):
            pass


def download_and_extract(url: str, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[skip] Папка уже существует: {target_dir}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / url.split("/")[-1]

    print(f"[download] {url}")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=archive_path.name) as bar:
        urllib.request.urlretrieve(url, archive_path, reporthook=bar.update_to)

    print(f"[extract] Распаковка в {RAW_DIR} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=RAW_DIR)

    archive_path.unlink()
    print(f"[done] Готово: {target_dir}")


if __name__ == "__main__":
    for url, target_dir in ARCHIVES:
        download_and_extract(url, target_dir)
    print("\nВесь датасет загружен в:", RAW_DIR)
