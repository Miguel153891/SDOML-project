"""Download the automobile dataset from Kaggle."""

from pathlib import Path

import kagglehub
import pandas as pd


DATASET_ID = "deeplumiere/automobile-market-analytics-dataset"
DATASET_FILE = "automobile_dataset.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / DATASET_FILE


def download_dataset():
    """Download the automobile dataset to the project's raw data directory.

    Returns
    -------
    pathlib.Path
        Path to the downloaded CSV file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = kagglehub.dataset_load(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        DATASET_ID,
        DATASET_FILE,
    )

    df.to_csv(RAW_DATA_FILE, index=False)

    print(f"Dataset saved to: {RAW_DATA_FILE}")

    return RAW_DATA_FILE


if __name__ == "__main__":
    download_dataset()
