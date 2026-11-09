"""Download the official dataset from Zenodo (record 4463683, CC-BY 4.0, 221 MB) into data/raw/."""

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = ("https://zenodo.org/api/records/4463683/files/"
       "Collision%20Avoidance%20Challenge%20-%20Dataset.zip/content")
WANTED = ["train_data.csv", "test_data.csv", "test_data_private.csv"]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in WANTED) and "--force" not in sys.argv:
        print("Data already present in", RAW)
        return
    print("Downloading the Kelvins dataset (221 MB) from Zenodo...")
    with urllib.request.urlopen(URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    prefix = "Collision Avoidance Challenge - Dataset/kelvins_competition_data/"
    for name in ("test_data.csv", "test_data_private.csv"):
        with outer.open(prefix + name) as src, open(RAW / name, "wb") as dst:
            shutil.copyfileobj(src, dst)
    with outer.open(prefix + "train_data.zip") as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read()))
        with inner.open("train_data.csv") as src, open(RAW / "train_data.csv", "wb") as dst:
            shutil.copyfileobj(src, dst)
    for f in WANTED:
        print(f"  {f}: {(RAW / f).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
