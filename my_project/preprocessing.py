"""Dataset preprocessing utilities."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from my_project.fileManager import FileManager
from my_project.download_data import download_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "automobile_dataset"
PROCESSED_TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "automobile_dataset"
PROCESSED_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "automobile_test"


def preprocess_dataset(df):
    """Clean and encode the automobile dataset."""

    # Optimize numerical types
    for column in df.columns:

        if df[column].dtype == "int64":
            df[column] = pd.to_numeric(
                df[column],
                downcast="integer"
            )

        elif df[column].dtype == "float64":
            df[column] = pd.to_numeric(
                df[column],
                downcast="float"
            )

        elif df[column].dtype == "object":
            df[column] = df[column].astype("category")

    # Remove rows containing missing values
    df = df.dropna().copy()

    # Encode service history
    service_mapping = {
        "No Service": 0,
        "Partial Service": 1,
        "Full Service": 2
    }

    df["Service_History"] = df["Service_History"].map(
        service_mapping
    )

    # One-hot encode categorical columns
    categorical_columns = [
        "Make",
        "Model",
        "Fuel_Type",
        "Transmission",
        "Color",
        "Body_Type",
        "Drivetrain",
        "Location"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    return df


def prepare_dataset():
    """Download, preprocess and save the dataset."""

    raw_csv = PROJECT_ROOT / "data" / "raw" / "automobile_dataset.csv"

    if not raw_csv.exists():
        download_dataset()

    fm = FileManager()
    fm.set_format("csv")

    df = fm.read(str(RAW_FILE))

    df = preprocess_dataset(df)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=22
    )

    PROCESSED_TRAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fm.set_format("parquet")

    fm.write(
        train_df,
        str(PROCESSED_TRAIN_FILE)
    )

    fm.write(
        test_df,
        str(PROCESSED_TEST_FILE)
    )

    return train_df, test_df


if __name__ == "__main__":
    prepare_dataset()