"""
Step 2: Download the three MoleculeNet benchmark datasets we need.

- ESOL   -> water solubility (regression)
- BBBP   -> blood-brain barrier penetration (binary classification)
- Tox21  -> toxicity across 12 assays (multi-label binary classification)

These are the same files DeepChem's built-in MoleculeNet loaders pull from,
hosted as public CSVs on S3. Downloading them directly (instead of via the
deepchem package) keeps our dependency list small.

Run:
    .venv/Scripts/python.exe scripts/download_data.py
"""

import gzip
import shutil
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = {
    "esol.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "bbbp.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "tox21.csv.gz": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
}


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved -> {dest}  ({dest.stat().st_size:,} bytes)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DATASETS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"[skip] {filename} already present")
            continue
        print(f"[fetch] {filename}")
        download(url, dest)

    # tox21 ships gzipped; unpack it alongside the .gz so explore/featurize
    # steps can just read a plain .csv like the other two datasets.
    gz_path = RAW_DIR / "tox21.csv.gz"
    csv_path = RAW_DIR / "tox21.csv"
    if gz_path.exists() and not csv_path.exists():
        print("[unpack] tox21.csv.gz -> tox21.csv")
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("\nAll datasets ready in", RAW_DIR)


if __name__ == "__main__":
    main()
