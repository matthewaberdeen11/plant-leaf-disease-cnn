# Plant Leaf Disease Detection using a CNN

COMP3220 (Artificial Intelligence) group project, UWI Mona, 2026.
Not for reuse by other students in this course.

## What it does

Classifies a photo of a single pepper, potato or tomato leaf into one of
15 classes: a healthy leaf for each crop, or one of 12 diseases. A small
CNN is compared with a logistic regression baseline, and both are tested
on lab photos (PlantVillage) and real-world field photos (PlantDoc).

## Setup and run

Requires Python 3.12 and git.

1. Clone the repo:
   `git clone https://github.com/matthewaberdeen11/plant-leaf-disease-cnn.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the datasets: `python src/download_data.py`
   (PlantVillage needs a Kaggle account; place your API token in `~/.kaggle/kaggle.json`)
4. _TODO: train the models_
5. _TODO: launch the demo app_

## Data

- **PlantVillage** (Kaggle: emmarex/plantdisease): 20,638 lab images,
  15 classes. Used for training, validation and testing.
- **PlantDoc** (GitHub: pratikkayal/PlantDoc-Dataset): real-world field
  photos. Used for testing only.

Cleaning and splits are in `notebooks/01_explore_data.ipynb`:

- All images readable; 14 exact duplicate pairs removed (20,624 unique images).
- The Kaggle archive contains a nested duplicate copy of the dataset, which is excluded.
- Stratified 70/15/15 train/validation/test split (seed 42), saved in `splits/`.
  Everyone uses these files; do not re-split.

## Project structure

    src/            download and preprocessing code
    notebooks/      data exploration
    splits/         fixed train/val/test CSVs and class order (labels.json)
    figures/        charts for the report
    data/           downloaded datasets (not committed)

## Results

_TODO: accuracy and macro F1 for both models on the lab and field test sets._

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