"""
data.py - Load the fixed splits as model-ready data.

Teammates import from here; never re-split the data.
Written with help from Claude (Anthropic).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "splits"
PV_DIR = ROOT / "data" / "raw" / "plantvillage" / "PlantVillage"
PD_DIR = ROOT / "data" / "raw" / "plantdoc"

CLASS_NAMES = json.loads((SPLITS_DIR / "labels.json").read_text())
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMG_SIZE = 128


def read_split(name):
    """Return full image paths and integer labels for 'train', 'val', 'test' or 'field_test'."""
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    base = PD_DIR if name == "field_test" else PV_DIR
    paths = [str(base / p) for p in df["path"]]
    labels = df["label"].map(LABEL_TO_INDEX).to_numpy()
    return paths, labels


def preprocess_image(image, img_size=IMG_SIZE):
    """Resize and scale pixels to 0-1. Used in training AND by the demo app."""
    image = image[..., :3]                       # drop alpha channel if present
    image = tf.image.resize(image, (img_size, img_size))
    return tf.cast(image, tf.float32) / 255.0


def _load_image(path, label, img_size):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return preprocess_image(image, img_size), label


def make_dataset(name, img_size=IMG_SIZE, batch_size=32, shuffle=False):
    """Build a batched tf.data pipeline for one split."""
    paths, labels = read_split(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(lambda p, l: _load_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_datasets(img_size=IMG_SIZE, batch_size=32):
    """For the CNN: train (shuffled), val, test and field_test datasets."""
    return {
        "train": make_dataset("train", img_size, batch_size, shuffle=True),
        "val": make_dataset("val", img_size, batch_size),
        "test": make_dataset("test", img_size, batch_size),
        "field_test": make_dataset("field_test", img_size, batch_size),
    }


def get_class_weights():
    """For the CNN: weights that make rare classes count more in the loss."""
    _, labels = read_split("train")
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {i: len(labels) / (len(CLASS_NAMES) * c) for i, c in enumerate(counts)}


def load_baseline_arrays(name, img_size=32):
    """For logistic regression: small images flattened into one row each."""
    paths, labels = read_split(name)
    X = np.stack([
        np.asarray(Image.open(p).convert("RGB").resize((img_size, img_size)))
        for p in paths
    ])
    X = X.reshape(len(X), -1).astype(np.float32) / 255.0
    return X, labels