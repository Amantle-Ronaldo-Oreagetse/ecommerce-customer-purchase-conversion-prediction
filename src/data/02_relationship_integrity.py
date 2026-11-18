# ============================================================
# ANALYSTLAB AFRICA - WEEK 2
# PROJECT: E-COMMERCE CUSTOMER PURCHASE CONVERSION PREDICTION
# STAGE 02.1: RELATIONSHIP AND DATA INTEGRITY INVESTIGATION
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# 1. LOAD DATASETS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw"

sessions = pd.read_csv(DATA_PATH / "website_sessions.csv")
pageviews = pd.read_csv(DATA_PATH / "website_pageviews.csv")
orders = pd.read_csv(DATA_PATH / "orders_mav.csv")


# ============================================================
# 2. CONVERT DATE COLUMNS
# ============================================================

sessions["created_at"] = pd.to_datetime(
    sessions["created_at"],
    errors="coerce"
)

pageviews["created_at"] = pd.to_datetime(
    pageviews["created_at"],
    errors="coerce"
)

orders["created_at"] = pd.to_datetime(
    orders["created_at"],
    errors="coerce"
)


# ============================================================
# 3. IDENTIFIER UNIQUENESS
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFIER UNIQUENESS")
print("=" * 70)

print(
    "\nSession ID duplicates:",
    sessions["website_session_id"].duplicated().sum()
)

print(
    "Pageview ID duplicates:",
    pageviews["website_pageview_id"].duplicated().sum()
)

print(
    "Order ID duplicates:",
    orders["order_id"].duplicated().sum()
)

print(
    "Order session ID duplicates:",
    orders["website_session_id"].duplicated().sum()
)


# ============================================================
# 4. SESSION COVERAGE
# ============================================================

session_ids = set(sessions["website_session_id"])

pageview_session_ids = set(
    pageviews["website_session_id"]
)

order_session_ids = set(
    orders["website_session_id"]
)

print("\n" + "=" * 70)
print("SESSION COVERAGE")
print("=" * 70)

print(
    "\nSessions represented in pageviews:",
    len(pageview_session_ids)
)

print(
    "Sessions represented in orders:",
    len(order_session_ids)
)

print(
    "Pageview sessions missing from sessions:",
    len(pageview_session_ids - session_ids)
)

print(
    "Order sessions missing from sessions:",
    len(order_session_ids - session_ids)
)

print(
    "Sessions without pageviews:",
    len(session_ids - pageview_session_ids)
)


# ============================================================
# 5. ORDER AND SESSION TIMING
# ============================================================

print("\n" + "=" * 70)
print("ORDER TIMING VALIDATION")
print("=" * 70)

session_dates = sessions[
    ["website_session_id", "created_at"]
].rename(
    columns={"created_at": "session_created_at"}
)

order_check = orders.merge(
    session_dates,
    on="website_session_id",
    how="left",
    indicator=True
)

print("\nOrder-to-session merge status:")
print(order_check["_merge"].value_counts())

valid_orders = order_check[
    order_check["_merge"] == "both"
].copy()

orders_before_session = (
    valid_orders["created_at"]
    < valid_orders["session_created_at"]
)

orders_same_time = (
    valid_orders["created_at"]
    == valid_orders["session_created_at"]
)

print(
    "\nOrders timestamped before their sessions:",
    orders_before_session.sum()
)

print(
    "Orders timestamped at session creation:",
    orders_same_time.sum()
)

print(
    "Orders timestamped after session creation:",
    (
        valid_orders["created_at"]
        > valid_orders["session_created_at"]
    ).sum()
)


# ============================================================
# 6. PAGEVIEW TIMING VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PAGEVIEW TIMING VALIDATION")
print("=" * 70)

pageview_check = pageviews.merge(
    session_dates,
    on="website_session_id",
    how="left",
    indicator=True
)

valid_pageviews = pageview_check[
    pageview_check["_merge"] == "both"
]

print(
    "\nPageviews timestamped before session creation:",
    (
        valid_pageviews["created_at"]
        < valid_pageviews["session_created_at"]
    ).sum()
)

print(
    "Pageviews timestamped at session creation:",
    (
        valid_pageviews["created_at"]
        == valid_pageviews["session_created_at"]
    ).sum()
)

print(
    "Pageviews timestamped after session creation:",
    (
        valid_pageviews["created_at"]
        > valid_pageviews["session_created_at"]
    ).sum()
)


# ============================================================
# 7. PAGEVIEWS PER SESSION
# ============================================================

print("\n" + "=" * 70)
print("PAGEVIEWS PER SESSION")
print("=" * 70)

pageviews_per_session = (
    pageviews.groupby("website_session_id")
    .size()
)

print(
    "\nDescriptive statistics:"
)

print(
    pageviews_per_session.describe()
)

print(
    "\nSessions with one pageview:",
    (pageviews_per_session == 1).sum()
)

print(
    "Sessions with more than one pageview:",
    (pageviews_per_session > 1).sum()
)

print(
    "Maximum pageviews in one session:",
    pageviews_per_session.max()
)


# ============================================================
# 8. SESSION AND ORDER CONVERSION OVERVIEW
# ============================================================

print("\n" + "=" * 70)
print("PRELIMINARY CONVERSION OVERVIEW")
print("=" * 70)

total_sessions = sessions["website_session_id"].nunique()

total_order_sessions = (
    orders["website_session_id"].nunique()
)

preliminary_conversion_rate = (
    total_order_sessions / total_sessions
) * 100

print("\nTotal unique sessions:", total_sessions)

print(
    "Unique sessions with orders:",
    total_order_sessions
)

print(
    "Preliminary conversion rate:",
    round(preliminary_conversion_rate, 4),
    "%"
)


# ============================================================
# 9. PAGEVIEWS DATASET SIZE INVESTIGATION
# ============================================================

print("\n" + "=" * 70)
print("PAGEVIEWS DATASET SIZE INVESTIGATION")
print("=" * 70)

print(
    "\nPageviews records:",
    len(pageviews)
)

print(
    "Excel worksheet maximum:",
    1_048_576
)

if len(pageviews) == 1_048_575:

    print(
        "\nWARNING: Dataset contains exactly "
        "1,048,575 records."
    )

    print(
        "Investigate possible Excel export truncation."
    )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RELATIONSHIP INVESTIGATION COMPLETED")
print("=" * 70)