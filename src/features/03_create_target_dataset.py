import pandas as pd
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STAGE 03: TARGET CONSTRUCTION")
print("=" * 70)

sessions = pd.read_csv(DATA_PATH / "website_sessions.csv")
orders = pd.read_csv(DATA_PATH / "orders_mav.csv")

print("\nOriginal sessions:", len(sessions))
print("Original orders:", len(orders))


# ============================================================
# CONSTRUCT CONVERSION TARGET
# ============================================================

# Extract unique session IDs that resulted in an order.

converted_sessions = orders["website_session_id"].drop_duplicates()

# Create the binary target.

sessions["converted"] = (
    sessions["website_session_id"]
    .isin(converted_sessions)
    .astype(int)
)


# ============================================================
# TARGET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_distribution = (
    sessions["converted"]
    .value_counts()
    .sort_index()
)

print(target_distribution)

print("\nTarget percentages:")

print(
    sessions["converted"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)


# ============================================================
# CONVERSION RATE
# ============================================================

total_sessions = len(sessions)
converted_count = sessions["converted"].sum()
non_converted_count = total_sessions - converted_count

conversion_rate = converted_count / total_sessions * 100

print("\n" + "=" * 70)
print("CONVERSION SUMMARY")
print("=" * 70)

print("Total sessions:", total_sessions)
print("Converted sessions:", converted_count)
print("Non-converted sessions:", non_converted_count)

print(f"Conversion rate: {conversion_rate:.4f}%")


# ============================================================
# CHECK TARGET AGAINST ORIGINAL ORDER DATA
# ============================================================

print("\n" + "=" * 70)
print("ORDER-TO-TARGET VALIDATION")
print("=" * 70)

print(
    "Unique order sessions:",
    orders["website_session_id"].nunique()
)

print(
    "Sessions labelled as converted:",
    sessions.loc[sessions["converted"] == 1,
                 "website_session_id"].nunique()
)

print(
    "Converted sessions without an order:",
    len(
        set(
            sessions.loc[
                sessions["converted"] == 1,
                "website_session_id"
            ]
        )
        -
        set(orders["website_session_id"])
    )
)

print(
    "Order sessions incorrectly labelled as non-converted:",
    len(
        set(orders["website_session_id"])
        -
        set(
            sessions.loc[
                sessions["converted"] == 1,
                "website_session_id"
            ]
        )
    )
)


# ============================================================
# INITIAL MODELLING DATASET
# ============================================================

feature_columns = [
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

modeling_data = sessions[feature_columns].copy()


# ============================================================
# MISSING VALUE INVESTIGATION
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES IN MODELLING DATASET")
print("=" * 70)

missing_summary = pd.DataFrame({
    "missing_count": modeling_data.isnull().sum(),
    "missing_percentage": (
        modeling_data.isnull().mean() * 100
    )
})

print(missing_summary)


# ============================================================
# MARKETING MISSINGNESS PATTERN
# ============================================================

print("\n" + "=" * 70)
print("MARKETING FIELD MISSINGNESS PATTERN")
print("=" * 70)

marketing_columns = [
    "utm_source",
    "utm_campaign",
    "utm_content"
]

missing_patterns = (
    modeling_data[marketing_columns]
    .isnull()
    .value_counts()
    .reset_index(name="session_count")
)

print(missing_patterns.to_string(index=False))


# ============================================================
# DATE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SESSION DATE VALIDATION")
print("=" * 70)

modeling_data["created_at"] = pd.to_datetime(
    modeling_data["created_at"],
    errors="coerce"
)

print(
    "Invalid session dates:",
    modeling_data["created_at"].isna().sum()
)

print("Earliest session:", modeling_data["created_at"].min())
print("Latest session:", modeling_data["created_at"].max())


# ============================================================
# SAVE DATASET
# ============================================================

output_file = OUTPUT_PATH / "initial_modeling_dataset.csv"

modeling_data.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("STAGE 03 COMPLETE")
print("=" * 70)

print("Dataset shape:", modeling_data.shape)
print("Saved to:", output_file)