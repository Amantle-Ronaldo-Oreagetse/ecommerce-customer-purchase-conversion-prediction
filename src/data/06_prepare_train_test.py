import pandas as pd
import json
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "preprocessing"

TRAIN_PATH = DATA_PATH / "train"
TEST_PATH = DATA_PATH / "test"

REPORT_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_PATH.mkdir(parents=True, exist_ok=True)
TEST_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FEATURE-ENGINEERED DATASET
# ============================================================

print("=" * 70)
print("STAGE 06: PREPROCESSING AND TRAIN/TEST PREPARATION")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH / "feature_engineered_dataset.csv"
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="raise"
)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# VALIDATE DATA
# ============================================================

if df["website_session_id"].duplicated().any():
    raise ValueError("Duplicate session IDs detected.")

if df["converted"].isna().any():
    raise ValueError("Missing target values detected.")

if not df["converted"].isin([0, 1]).all():
    raise ValueError("Unexpected target values detected.")

if df.isna().any().any():
    raise ValueError("Missing values detected in dataset.")


# ============================================================
# DEFINE FEATURE GROUPS
# ============================================================

core_categorical_features = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

core_binary_features = [
    "is_repeat_session"
]

additional_time_features = [
    "session_hour",
    "session_day_of_week",
    "session_month"
]

core_features = (
    core_categorical_features
    + core_binary_features
)

extended_features = (
    core_features
    + additional_time_features
)

target_column = "converted"

identifier_columns = [
    "website_session_id",
    "user_id",
    "created_at"
]


# ============================================================
# CHRONOLOGICAL SORTING
# ============================================================

df = df.sort_values(
    "created_at"
).reset_index(drop=True)

print("\nChronological sorting completed.")

print(
    "Earliest session:",
    df["created_at"].min()
)

print(
    "Latest session:",
    df["created_at"].max()
)


# ============================================================
# CHRONOLOGICAL 80/20 SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

# Select the timestamp at the intended split position.
# Sessions with the same timestamp will remain together.

split_timestamp = df.loc[
    split_index,
    "created_at"
]

train_df = df[
    df["created_at"] < split_timestamp
].copy()

test_df = df[
    df["created_at"] >= split_timestamp
].copy()


# ============================================================
# SPLIT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT SUMMARY")
print("=" * 70)

print("Intended split index:", split_index)
print("Actual split timestamp:", split_timestamp)

print("\nTraining sessions:", len(train_df))
print("Testing sessions:", len(test_df))

print(
    "Training percentage:",
    round(len(train_df) / len(df) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(test_df) / len(df) * 100, 2),
    "%"
)

print("\nTraining date range:")

print("Start:", train_df["created_at"].min())
print("End:", train_df["created_at"].max())

print("\nTesting date range:")

print("Start:", test_df["created_at"].min())
print("End:", test_df["created_at"].max())

if train_df.empty or test_df.empty:
    raise ValueError("Train or test dataset is empty.")

if train_df["created_at"].max() >= test_df["created_at"].min():
    raise ValueError("Chronological separation failed.")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_summary = []

for dataset_name, dataset in [
    ("Full dataset", df),
    ("Training", train_df),
    ("Testing", test_df)
]:

    distribution = (
        dataset["converted"]
        .value_counts()
        .sort_index()
    )

    conversion_rate = (
        dataset["converted"].mean() * 100
    )

    target_summary.append({
        "dataset": dataset_name,
        "total_sessions": len(dataset),
        "non_converted": int(distribution.get(0, 0)),
        "converted": int(distribution.get(1, 0)),
        "conversion_rate_percent": conversion_rate
    })

    print(f"\n{dataset_name}")

    print(distribution)

    print(
        f"Conversion rate: {conversion_rate:.4f}%"
    )

target_summary_df = pd.DataFrame(target_summary)

target_summary_df.to_csv(
    REPORT_PATH / "target_distribution_by_split.csv",
    index=False
)


# ============================================================
# PREPARE FEATURE MATRICES
# ============================================================

# Keep identifiers separately for traceability.
# They will not be passed to the model.

train_metadata = train_df[
    identifier_columns
].copy()

test_metadata = test_df[
    identifier_columns
].copy()

# Core feature set

X_train_core = train_df[
    core_features
].copy()

X_test_core = test_df[
    core_features
].copy()

# Extended feature set

X_train_extended = train_df[
    extended_features
].copy()

X_test_extended = test_df[
    extended_features
].copy()

# Target vectors

y_train = train_df[target_column].copy()
y_test = test_df[target_column].copy()


# ============================================================
# SAVE TRAINING AND TEST DATA
# ============================================================

X_train_core.to_csv(
    TRAIN_PATH / "X_train_core.csv",
    index=False
)

X_test_core.to_csv(
    TEST_PATH / "X_test_core.csv",
    index=False
)

X_train_extended.to_csv(
    TRAIN_PATH / "X_train_extended.csv",
    index=False
)

X_test_extended.to_csv(
    TEST_PATH / "X_test_extended.csv",
    index=False
)

y_train.to_frame().to_csv(
    TRAIN_PATH / "y_train.csv",
    index=False
)

y_test.to_frame().to_csv(
    TEST_PATH / "y_test.csv",
    index=False
)

train_metadata.to_csv(
    TRAIN_PATH / "train_metadata.csv",
    index=False
)

test_metadata.to_csv(
    TEST_PATH / "test_metadata.csv",
    index=False
)


# ============================================================
# SAVE PREPROCESSING CONFIGURATION
# ============================================================

preprocessing_config = {
    "split_method": "Chronological",
    "intended_train_percentage": 80,
    "intended_test_percentage": 20,
    "split_timestamp": str(split_timestamp),

    "core_categorical_features": (
        core_categorical_features
    ),

    "core_binary_features": (
        core_binary_features
    ),

    "additional_time_features": (
        additional_time_features
    ),

    "target": target_column,

    "excluded_identifier_columns": (
        identifier_columns
    ),

    "categorical_encoding": "OneHotEncoder",

    "numerical_scaling": "StandardScaler",

    "missing_value_strategy": (
        "Missing categorical values represented "
        "as an explicit category."
    ),

    "preprocessing_fit_policy": (
        "Fit transformations on training data only."
    )
}

with open(
    REPORT_PATH / "preprocessing_configuration.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        preprocessing_config,
        file,
        indent=4
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAIN/TEST VALIDATION")
print("=" * 70)

print("\nCore feature shapes:")

print("X_train_core:", X_train_core.shape)
print("X_test_core:", X_test_core.shape)

print("\nExtended feature shapes:")

print("X_train_extended:", X_train_extended.shape)
print("X_test_extended:", X_test_extended.shape)

print("\nTarget shapes:")

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nTraining and testing overlap check:")

train_ids = set(train_metadata["website_session_id"])
test_ids = set(test_metadata["website_session_id"])

print("Overlapping session IDs:", len(train_ids & test_ids))

print("\nTraining and testing class counts:")

print("Training:", y_train.value_counts().to_dict())
print("Testing:", y_test.value_counts().to_dict())


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 06 COMPLETE")
print("=" * 70)

print("Training data saved to:", TRAIN_PATH)
print("Testing data saved to:", TEST_PATH)
print("Preprocessing reports saved to:", REPORT_PATH)