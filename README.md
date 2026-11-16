# Plant Leaf Disease Detection using a CNN

COMP3220 (Artificial Intelligence) group project, UWI Mona, 2026.
Not for reuse by other students in this course.

## What it does

Classifies a photo of a single pepper, potato or tomato leaf into one of
15 classes: a healthy leaf for each crop, or one of 12 diseases. A small
CNN is compared with a logistic regression baseline, and both are tested
on lab photos (PlantVillage) and real-world field photos (PlantDoc).

## Setup and run

Requires **Python 3.12** (TensorFlow does not yet support newer versions) and git.

1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/matthewaberdeen11/plant-leaf-disease-cnn.git
   cd plant-leaf-disease-cnn
   ```
2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the datasets (PlantVillage needs a free Kaggle account; place your
   API token in `~/.kaggle/kaggle.json`):
   ```bash
   python src/download_data.py
   ```
5. _TODO: launch the demo app_

Activate the environment (`source .venv/bin/activate`) in every new terminal.
In VS Code, select the `.venv` kernel for notebooks.

## Data

| Dataset | Source | Use |
|---|---|---|
| PlantVillage | Kaggle: emmarex/plantdisease | Train, validation, lab test |
| PlantDoc | GitHub: pratikkayal/PlantDoc-Dataset | Field test only (never trained on) |

Data preparation is in `notebooks/01_explore_data.ipynb`:

- **PlantVillage:** 20,638 images in 15 classes, all readable. 14 exact
  duplicate pairs removed by MD5 hash, leaving **20,624** unique images.
  The Kaggle archive also contains a nested duplicate copy of the dataset,
  which is excluded.
- **Classes are imbalanced:** from 152 images (Potato healthy) to 3,208
  (Tomato yellow leaf curl virus). Splits are stratified, and class weights
  and macro F1 are used to account for this.
- **Splits:** stratified 70 / 15 / 15 (seed 42): 14,436 train, 3,094 validation,
  3,094 test. Saved in `splits/`. Everyone uses these files; do not re-split.
- **Field test set:** PlantDoc folders mapped to 12 of the 15 classes, giving
  **1,089** images after removing duplicates, with **no overlap** with PlantVillage.
  Potato healthy and Tomato target spot have no PlantDoc equivalent, and Tomato
  spider mites has only 2 images, so these three classes are evaluated on lab
  data only.
- **macOS/Windows note:** PlantDoc contains a few filenames that differ only
  by case. On case-insensitive file systems git keeps one of each pair; this
  affects 1 image in a class we use.

## Using the data in code

All loading goes through `src/data.py`:

```python
import sys; sys.path.append("../src")   # from inside notebooks/
from data import (load_datasets, get_class_weights,
                  load_baseline_arrays, preprocess_image, CLASS_NAMES)

ds = load_datasets()                    # dict of tf.data datasets: train, val, test, field_test
weights = get_class_weights()           # pass to model.fit(class_weight=weights)
X, y = load_baseline_arrays("train")    # 32x32 flattened arrays for logistic regression
```

- Images are resized to 128×128 and scaled to 0–1 by `preprocess_image`.
  The demo app must use the same function.
- Label numbers follow the order in `splits/labels.json`.

## Project structure

    src/            download_data.py, data.py
    notebooks/      01_explore_data (cleaning, splits), 02_test_data (loader checks)
    splits/         train/val/test/field_test CSVs and labels.json
    figures/        charts for the report
    data/           downloaded datasets (not committed)

## Results



## Team

| Member | Role |
|---|---|
| Matthew Aberdeen | Dataset preparation and image preprocessing |
| Samuel Cummings | CNN implementation, training and optimization |
| Roshaun Marshall | Logistic regression baseline and model evaluation |
| Rahjay Jones | User interface and system integration |

## Sources

- PlantVillage dataset: Hughes & Salathé (2015), via Kaggle (emmarex/plantdisease).
- PlantDoc dataset: Singh et al. (2020), "PlantDoc: A Dataset for Visual Plant
  Disease Detection", via GitHub (pratikkayal/PlantDoc-Dataset).