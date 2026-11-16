"""Download PlantVillage (Kaggle) and PlantDoc (GitHub) into data/raw/."""
import shutil
import subprocess
from pathlib import Path

import kagglehub

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # PlantVillage: kagglehub downloads to a cache, then we copy it into the repo
    plantvillage = RAW_DIR / "plantvillage"
    if not plantvillage.exists():
        cache_path = kagglehub.dataset_download("emmarex/plantdisease")
        shutil.copytree(cache_path, plantvillage)

    # PlantDoc: clone the GitHub repo
    plantdoc = RAW_DIR / "plantdoc"
    if not plantdoc.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/pratikkayal/PlantDoc-Dataset", str(plantdoc)],
                       check=True)

    print("Done. Data is in", RAW_DIR)


if __name__ == "__main__":
    main()