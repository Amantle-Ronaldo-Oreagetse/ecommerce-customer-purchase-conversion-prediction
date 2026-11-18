import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda"
FIGURE_PATH = PROJECT_ROOT / "figures" / "eda"

REPORT_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STAGE 04: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH / "initial_modeling_dataset.csv")

df["created_at"] = pd.to_datetime(df["created_at"])

# Temporary copy for categorical analysis.
# This does not permanently modify the original dataset.

eda_df = df.copy()

categorical_columns = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

for column in categorical_columns:
    eda_df[column] = eda_df[column].fillna("Missing")


# ============================================================
# GENERAL OVERVIEW
# ============================================================

print("\nGENERAL DATASET OVERVIEW")

print("Total sessions:", len(df))
print("Converted sessions:", df["converted"].sum())
print("Overall conversion rate:",
      round(df["converted"].mean() * 100, 4), "%")

print("\nUnique values in categorical columns:")

for column in categorical_columns:
    print(f"{column}: {eda_df[column].nunique()}")


# ============================================================
# FUNCTION: CONVERSION BY CATEGORY
# ============================================================

def conversion_by_category(data, column):

    summary = (
        data.groupby(column, dropna=False)
        .agg(
            sessions=("converted", "size"),
            conversions=("converted", "sum"),
            conversion_rate=("converted", "mean")
        )
        .reset_index()
    )

    summary["conversion_rate"] *= 100

    summary = summary.sort_values(
        "sessions",
        ascending=False
    )

    return summary


# ============================================================
# DEVICE TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION BY DEVICE TYPE")
print("=" * 70)

device_summary = conversion_by_category(
    eda_df, "device_type"
)

print(device_summary.to_string(index=False))

device_summary.to_csv(
    REPORT_PATH / "conversion_by_device.csv",
    index=False
)


# ============================================================
# REPEAT SESSION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION BY SESSION TYPE")
print("=" * 70)

repeat_summary = (
    df.groupby("is_repeat_session")
    .agg(
        sessions=("converted", "size"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean")
    )
    .reset_index()
)

repeat_summary["conversion_rate"] *= 100

repeat_summary["session_type"] = (
    repeat_summary["is_repeat_session"]
    .map({
        0: "New session",
        1: "Repeat session"
    })
)

print(repeat_summary.to_string(index=False))

repeat_summary.to_csv(
    REPORT_PATH / "conversion_by_session_type.csv",
    index=False
)


# ============================================================
# MARKETING ANALYSIS
# ============================================================

marketing_columns = [
    "utm_source",
    "utm_campaign",
    "utm_content"
]

for column in marketing_columns:

    print("\n" + "=" * 70)
    print(f"CONVERSION BY {column.upper()}")
    print("=" * 70)

    summary = conversion_by_category(
        eda_df, column
    )

    print(summary.head(20).to_string(index=False))

    summary.to_csv(
        REPORT_PATH / f"conversion_by_{column}.csv",
        index=False
    )


# ============================================================
# REFERRER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION BY REFERRER")
print("=" * 70)

referrer_summary = conversion_by_category(
    eda_df, "http_referer"
)

print(referrer_summary.head(20).to_string(index=False))

referrer_summary.to_csv(
    REPORT_PATH / "conversion_by_referrer.csv",
    index=False
)


# ============================================================
# MISSING MARKETING INFORMATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION BY MARKETING INFORMATION AVAILABILITY")
print("=" * 70)

eda_df["marketing_information_status"] = (
    eda_df["utm_source"]
    .eq("Missing")
    .map({
        True: "Marketing information missing",
        False: "Marketing information available"
    })
)

missing_marketing_summary = (
    eda_df.groupby("marketing_information_status")
    .agg(
        sessions=("converted", "size"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean")
    )
    .reset_index()
)

missing_marketing_summary["conversion_rate"] *= 100

print(missing_marketing_summary.to_string(index=False))

missing_marketing_summary.to_csv(
    REPORT_PATH / "conversion_by_marketing_availability.csv",
    index=False
)


# ============================================================
# MONTHLY CONVERSION TRENDS
# ============================================================

print("\n" + "=" * 70)
print("MONTHLY CONVERSION TRENDS")
print("=" * 70)

df["month"] = df["created_at"].dt.to_period("M").astype(str)

monthly_summary = (
    df.groupby("month")
    .agg(
        sessions=("converted", "size"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean")
    )
    .reset_index()
)

monthly_summary["conversion_rate"] *= 100

print(monthly_summary.head(15).to_string(index=False))

monthly_summary.to_csv(
    REPORT_PATH / "monthly_conversion_trends.csv",
    index=False
)


# ============================================================
# FIGURE 1: CONVERSION BY DEVICE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    device_summary["device_type"],
    device_summary["conversion_rate"]
)

plt.title("Conversion Rate by Device Type")
plt.xlabel("Device Type")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "conversion_by_device.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2: CONVERSION BY SESSION TYPE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    repeat_summary["session_type"],
    repeat_summary["conversion_rate"]
)

plt.title("Conversion Rate: New vs Repeat Sessions")
plt.xlabel("Session Type")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "conversion_by_session_type.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3: MONTHLY CONVERSION RATE
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_summary["month"],
    monthly_summary["conversion_rate"],
    marker="o"
)

plt.title("Monthly E-commerce Conversion Rate")
plt.xlabel("Month")
plt.ylabel("Conversion Rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "monthly_conversion_rate.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 4: MONTHLY SESSION VOLUME
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_summary["month"],
    monthly_summary["sessions"],
    marker="o"
)

plt.title("Monthly Website Session Volume")
plt.xlabel("Month")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "monthly_session_volume.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE GENERAL SUMMARY
# ============================================================

general_summary = pd.DataFrame({
    "metric": [
        "Total sessions",
        "Converted sessions",
        "Non-converted sessions",
        "Overall conversion rate",
        "Sessions with missing marketing information",
        "Sessions with missing referrer"
    ],
    "value": [
        len(df),
        int(df["converted"].sum()),
        int((df["converted"] == 0).sum()),
        df["converted"].mean() * 100,
        int(df["utm_source"].isna().sum()),
        int(df["http_referer"].isna().sum())
    ]
})

general_summary.to_csv(
    REPORT_PATH / "general_eda_summary.csv",
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 04 COMPLETE")
print("=" * 70)

print("EDA reports saved to:", REPORT_PATH)
print("EDA figures saved to:", FIGURE_PATH)

print("\nReports generated:")
for file in sorted(REPORT_PATH.glob("*.csv")):
    print("-", file.name)

print("\nFigures generated:")
for file in sorted(FIGURE_PATH.glob("*.png")):
    print("-", file.name)