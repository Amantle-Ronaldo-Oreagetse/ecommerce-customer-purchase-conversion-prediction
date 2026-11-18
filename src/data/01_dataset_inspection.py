# ============================================================
# ANALYSTLAB AFRICA - WEEK 2
# PROJECT: E-COMMERCE CUSTOMER PURCHASE CONVERSION PREDICTION
# STAGE 02: DATASET INSPECTION
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw"

datasets = {
    "website_sessions": DATA_PATH / "website_sessions.csv",
    "website_pageviews": DATA_PATH / "website_pageviews.csv",
    "orders": DATA_PATH / "orders_mav.csv"
}


# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

data = {}

for name, path in datasets.items():

    print(f"\nLoading: {name}")

    df = pd.read_csv(path)

    data[name] = df

    print("Successfully loaded.")


# ------------------------------------------------------------
# 3. GENERAL DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET DIMENSIONS")
print("=" * 70)

for name, df in data.items():

    print(f"\n{name.upper()}")

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 4. COLUMN NAMES AND DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN INFORMATION")
print("=" * 70)

for name, df in data.items():

    print(f"\n{name.upper()}")

    print(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        }).to_string(index=False)
    )


# ------------------------------------------------------------
# 5. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

for name, df in data.items():

    print(f"\n{name.upper()}")

    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })

    missing = missing[
        missing["Missing Count"] > 0
    ]

    if missing.empty:
        print("No missing values detected.")

    else:
        print(missing.to_string())


# ------------------------------------------------------------
# 6. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 70)

for name, df in data.items():

    print(f"\n{name.upper()}")

    print(
        "Exact duplicate rows:",
        df.duplicated().sum()
    )


# ------------------------------------------------------------
# 7. UNIQUE IDENTIFIER INVESTIGATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("POTENTIAL IDENTIFIER ANALYSIS")
print("=" * 70)

identifier_columns = [
    "website_session_id",
    "website_pageview_id",
    "order_id",
    "user_id"
]

for name, df in data.items():

    print(f"\n{name.upper()}")

    for column in identifier_columns:

        if column in df.columns:

            print(
                f"{column}: "
                f"{df[column].nunique()} unique values"
            )


# ------------------------------------------------------------
# 8. DATE RANGE INVESTIGATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATE RANGE INVESTIGATION")
print("=" * 70)

date_columns = [
    "created_at",
    "website_session_created_at",
    "website_pageview_created_at",
    "order_created_at"
]

for name, df in data.items():

    print(f"\n{name.upper()}")

    for column in date_columns:

        if column in df.columns:

            dates = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            print(f"\nColumn: {column}")
            print("Earliest:", dates.min())
            print("Latest:", dates.max())


# ------------------------------------------------------------
# 9. SAMPLE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE RECORDS")
print("=" * 70)

for name, df in data.items():

    print(f"\n{name.upper()}")

    print(df.head(5).to_string(index=False))


# ------------------------------------------------------------
# 10. INSPECTION COMPLETED
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETED")
print("=" * 70)