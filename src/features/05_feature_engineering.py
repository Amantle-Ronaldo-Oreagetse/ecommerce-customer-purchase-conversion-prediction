import pandas as pd
import json
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_engineering"

REPORT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STAGE 05: FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH / "initial_modeling_dataset.csv"
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce"
)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "website_session_id",
    "created_at",
    "user_id",
    "is_repeat_session",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "device_type",
    "http_referer",
    "converted"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["created_at"].isna().any():
    raise ValueError(
        "Invalid session timestamps detected."
    )

if not df["converted"].isin([0, 1]).all():
    raise ValueError(
        "Target contains values other than 0 and 1."
    )

if not df["is_repeat_session"].isin([0, 1]).all():
    raise ValueError(
        "is_repeat_session contains unexpected values."
    )


# ============================================================
# HANDLE MISSING CATEGORICAL VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)

categorical_features = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

print("\nMissing values before handling:")

print(
    df[categorical_features]
    .isna()
    .sum()
)

for column in categorical_features:

    df[column] = (
        df[column]
        .fillna("Missing")
        .astype(str)
        .str.strip()
    )

print("\nMissing values after handling:")

print(
    df[categorical_features]
    .isna()
    .sum()
)


# ============================================================
# CREATE TIME-BASED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TIME-BASED FEATURE ENGINEERING")
print("=" * 70)

df["session_hour"] = df["created_at"].dt.hour

# Monday = 0, Sunday = 6
df["session_day_of_week"] = (
    df["created_at"].dt.dayofweek
)

df["session_month"] = (
    df["created_at"].dt.month
)

print("\nSession hour distribution:")

print(
    df["session_hour"]
    .value_counts()
    .sort_index()
)

print("\nSession day-of-week distribution:")

print(
    df["session_day_of_week"]
    .value_counts()
    .sort_index()
)

print("\nSession month distribution:")

print(
    df["session_month"]
    .value_counts()
    .sort_index()
)


# ============================================================
# DEFINE FEATURE GROUPS
# ============================================================

core_features = [
    "device_type",
    "is_repeat_session",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

additional_time_features = [
    "session_hour",
    "session_day_of_week",
    "session_month"
]

identifier_columns = [
    "website_session_id",
    "user_id",
    "created_at"
]

target_column = "converted"

all_predictive_features = (
    core_features + additional_time_features
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

print("\nCore predictive features:")

for feature in core_features:
    print(
        f"{feature}: "
        f"{df[feature].nunique()} unique values"
    )

print("\nAdditional time features:")

for feature in additional_time_features:
    print(
        f"{feature}: "
        f"{df[feature].nunique()} unique values"
    )


# ============================================================
# CHECK CATEGORICAL VALUES
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE VALUES")
print("=" * 70)

for feature in categorical_features:

    print(f"\n{feature}:")

    print(
        df[feature]
        .value_counts()
        .to_string()
    )


# ============================================================
# CREATE FEATURE SUMMARY
# ============================================================

feature_summary = []

for feature in all_predictive_features:

    feature_summary.append({
        "feature": feature,
        "data_type": str(df[feature].dtype),
        "unique_values": int(df[feature].nunique()),
        "missing_values": int(df[feature].isna().sum()),
        "feature_group": (
            "Core"
            if feature in core_features
            else "Time-based"
        )
    })

feature_summary_df = pd.DataFrame(feature_summary)

feature_summary_df.to_csv(
    REPORT_PATH / "feature_summary.csv",
    index=False
)


# ============================================================
# CREATE FEATURE DICTIONARY
# ============================================================

feature_dictionary = {
    "target": {
        "name": "converted",
        "description": (
            "Indicates whether a website session resulted "
            "in a purchase."
        ),
        "values": {
            "0": "No purchase",
            "1": "Purchase"
        }
    },

    "core_features": {
        "device_type": (
            "Device used during the website session."
        ),
        "is_repeat_session": (
            "Indicates whether the session belongs to "
            "a returning visitor."
        ),
        "utm_source": (
            "Marketing source associated with the session."
        ),
        "utm_campaign": (
            "Marketing campaign associated with the session."
        ),
        "utm_content": (
            "Marketing advertisement content associated "
            "with the session."
        ),
        "http_referer": (
            "Referring website associated with the session."
        )
    },

    "additional_time_features": {
        "session_hour": (
            "Hour of the day when the session began."
        ),
        "session_day_of_week": (
            "Day of the week when the session began. "
            "Monday = 0 and Sunday = 6."
        ),
        "session_month": (
            "Calendar month when the session began."
        )
    },

    "excluded_from_predictors": {
        "website_session_id": (
            "Unique session identifier."
        ),
        "user_id": (
            "Unique user identifier."
        ),
        "created_at": (
            "Original session timestamp, retained for "
            "chronological splitting and analysis."
        )
    },

    "missing_value_strategy": (
        "Missing categorical values are represented "
        "by an explicit Missing category."
    )
}

with open(
    REPORT_PATH / "feature_dictionary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        feature_dictionary,
        file,
        indent=4
    )


# ============================================================
# CREATE FEATURE-ENGINEERED DATASET
# ============================================================

output_columns = (
    identifier_columns
    + all_predictive_features
    + [target_column]
)

feature_engineered_data = df[
    output_columns
].copy()

output_file = (
    DATA_PATH / "feature_engineered_dataset.csv"
)

feature_engineered_data.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET VALIDATION")
print("=" * 70)

print(
    "Dataset shape:",
    feature_engineered_data.shape
)

print(
    "\nTotal missing values:",
    feature_engineered_data.isna().sum().sum()
)

print(
    "Duplicate session IDs:",
    feature_engineered_data[
        "website_session_id"
    ].duplicated().sum()
)

print(
    "Target distribution:"
)

print(
    feature_engineered_data[
        "converted"
    ].value_counts().sort_index()
)

print("\nSaved dataset:", output_file)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 05 COMPLETE")
print("=" * 70)

print("Feature-engineered dataset saved.")
print("Feature reports saved to:", REPORT_PATH)