from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "public_modeling_dataset.csv"
)


# Features and target to include
PUBLIC_COLUMNS = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer",
    "is_repeat_session",
    "converted",
]


def main():

    print("Loading feature-engineered dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)

    # Select only modelling features and target
    public_df = df[PUBLIC_COLUMNS].copy()

    # Validate the dataset
    assert public_df.isnull().sum().sum() == 0
    assert public_df["converted"].isin([0, 1]).all()

    assert public_df.shape[0] == df.shape[0]

    # Save shareable dataset
    public_df.to_csv(OUTPUT_FILE, index=False)

    print("\nPUBLIC DATASET CREATED")
    print("Output:", OUTPUT_FILE)
    print("Shape:", public_df.shape)

    print("\nColumns:")
    print(public_df.columns.tolist())

    print("\nTarget distribution:")
    print(public_df["converted"].value_counts())

    print("\nMissing values:")
    print(public_df.isnull().sum())

    print("\nDataset saved successfully.")


if __name__ == "__main__":
    main()