from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "stage_09"

REPORT_DIR = PROJECT_ROOT / "reports"
EVALUATION_DIR = REPORT_DIR / "model_evaluation"
IMPORTANCE_DIR = REPORT_DIR / "feature_importance"
ERROR_DIR = REPORT_DIR / "error_analysis"

for folder in [IMPORTANCE_DIR, ERROR_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50

# ============================================================
# 2. LOAD TRAINING DATA AND METADATA
# ============================================================

print("\n========== LOADING VALIDATION DATA ==========")

X = pd.read_csv(PROCESSED_DIR / "train" / "X_train_core.csv")
y = pd.read_csv(PROCESSED_DIR / "train" / "y_train.csv").squeeze()

metadata = pd.read_csv(
    PROCESSED_DIR / "train" / "train_metadata.csv"
)

# Sort chronologically to reproduce the Stage 09 split.
metadata["created_at"] = pd.to_datetime(
    metadata["created_at"],
    errors="coerce"
)

order = metadata.sort_values(
    "created_at",
    kind="mergesort"
).index

X = X.loc[order].reset_index(drop=True)
y = y.loc[order].reset_index(drop=True)
metadata = metadata.loc[order].reset_index(drop=True)

# Reproduce the Stage 09 chronological 80/20 split.
split_index = int(len(X) * 0.80)

X_validation = X.iloc[split_index:].copy()
y_validation = y.iloc[split_index:].copy()
validation_metadata = metadata.iloc[split_index:].copy()

print("Validation records:", len(X_validation))
print("Validation conversion rate:", round(y_validation.mean(), 4))

# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

print("\n========== LOADING TRAINED MODELS ==========")

logistic_model = joblib.load(
    MODEL_DIR / "logistic_regression.joblib"
)

gradient_model = joblib.load(
    MODEL_DIR / "gradient_boosting.joblib"
)

models = {
    "Logistic Regression": logistic_model,
    "Gradient Boosting": gradient_model
}

# ============================================================
# 4. GENERATE VALIDATION PREDICTIONS
# ============================================================

print("\n========== GENERATING VALIDATION PREDICTIONS ==========")

prediction_data = validation_metadata.copy()

prediction_data["actual"] = y_validation.to_numpy()

for name, model in models.items():

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)

    prefix = (
        "logistic_regression"
        if name == "Logistic Regression"
        else "gradient_boosting"
    )

    prediction_data[f"{prefix}_probability"] = probabilities
    prediction_data[f"{prefix}_prediction"] = predictions

    print(f"\n{name}")
    print("Predicted positive sessions:", predictions.sum())
    print("Actual positive sessions:", y_validation.sum())

# Save combined predictions.
prediction_data.to_csv(
    ERROR_DIR / "validation_predictions_with_metadata.csv",
    index=False
)

# ============================================================
# 5. LOGISTIC REGRESSION FEATURE IMPORTANCE
# ============================================================

print("\n========== LOGISTIC REGRESSION FEATURE IMPORTANCE ==========")

preprocessor = logistic_model.named_steps["preprocessor"]
classifier = logistic_model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

coefficients = classifier.coef_[0]

lr_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_coefficient": np.abs(coefficients)
})

lr_importance["direction"] = np.where(
    lr_importance["coefficient"] > 0,
    "Positive association",
    "Negative association"
)

lr_importance = lr_importance.sort_values(
    "absolute_coefficient",
    ascending=False
)

lr_importance.to_csv(
    IMPORTANCE_DIR / "logistic_regression_feature_importance.csv",
    index=False
)

print("\nTop 15 Logistic Regression features:")
print(lr_importance.head(15).to_string(index=False))

# Save strongest positive and negative associations separately.
lr_importance.sort_values(
    "coefficient",
    ascending=False
).head(15).to_csv(
    IMPORTANCE_DIR / "logistic_regression_top_positive_features.csv",
    index=False
)

lr_importance.sort_values(
    "coefficient",
    ascending=True
).head(15).to_csv(
    IMPORTANCE_DIR / "logistic_regression_top_negative_features.csv",
    index=False
)

# ============================================================
# 6. GRADIENT BOOSTING FEATURE IMPORTANCE
# ============================================================

print("\n========== GRADIENT BOOSTING FEATURE IMPORTANCE ==========")

gb_preprocessor = gradient_model.named_steps["preprocessor"]
gb_classifier = gradient_model.named_steps["classifier"]

gb_feature_names = gb_preprocessor.get_feature_names_out()

gb_importance = pd.DataFrame({
    "feature": gb_feature_names,
    "importance": gb_classifier.feature_importances_
})

gb_importance = gb_importance.sort_values(
    "importance",
    ascending=False
)

gb_importance.to_csv(
    IMPORTANCE_DIR / "gradient_boosting_feature_importance.csv",
    index=False
)

print("\nTop 15 Gradient Boosting features:")
print(gb_importance.head(15).to_string(index=False))

# ============================================================
# 7. AGGREGATE IMPORTANCE BY ORIGINAL FEATURE
# ============================================================

print("\n========== AGGREGATED FEATURE IMPORTANCE ==========")

def original_feature_name(encoded_name):

    name = encoded_name.split("__", 1)[-1]

    original_features = [
        "device_type",
        "utm_source",
        "utm_campaign",
        "utm_content",
        "http_referer",
        "is_repeat_session"
    ]

    for feature in original_features:
        if name == feature or name.startswith(feature + "_"):
            return feature

    return name


lr_importance["original_feature"] = (
    lr_importance["feature"].apply(original_feature_name)
)

gb_importance["original_feature"] = (
    gb_importance["feature"].apply(original_feature_name)
)

lr_grouped = (
    lr_importance
    .groupby("original_feature")["absolute_coefficient"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

gb_grouped = (
    gb_importance
    .groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

lr_grouped.to_csv(
    IMPORTANCE_DIR / "logistic_regression_grouped_importance.csv",
    index=False
)

gb_grouped.to_csv(
    IMPORTANCE_DIR / "gradient_boosting_grouped_importance.csv",
    index=False
)

print("\nLogistic Regression grouped importance:")
print(lr_grouped.to_string(index=False))

print("\nGradient Boosting grouped importance:")
print(gb_grouped.to_string(index=False))

# ============================================================
# 8. ERROR ANALYSIS
# ============================================================

print("\n========== ERROR ANALYSIS ==========")

for model_name, prefix in [
    ("Logistic Regression", "logistic_regression"),
    ("Gradient Boosting", "gradient_boosting")
]:

    prediction_column = f"{prefix}_prediction"
    probability_column = f"{prefix}_probability"

    errors = prediction_data.copy()

    errors["predicted"] = errors[prediction_column]
    errors["predicted_probability"] = errors[probability_column]

    errors["error_type"] = np.select(
        [
            (errors["actual"] == 1) & (errors["predicted"] == 1),
            (errors["actual"] == 0) & (errors["predicted"] == 0),
            (errors["actual"] == 0) & (errors["predicted"] == 1),
            (errors["actual"] == 1) & (errors["predicted"] == 0)
        ],
        [
            "True Positive",
            "True Negative",
            "False Positive",
            "False Negative"
        ],
        default="Unknown"
    )

    errors.to_csv(
        ERROR_DIR / f"{prefix}_error_records.csv",
        index=False
    )

    error_summary = (
        errors["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    error_summary.to_csv(
        ERROR_DIR / f"{prefix}_error_summary.csv",
        index=False
    )

    print(f"\n{model_name}")
    print(error_summary.to_string(index=False))

# ============================================================
# 9. PERFORMANCE BY CUSTOMER/SESSION SEGMENT
# ============================================================

print("\n========== SEGMENT PERFORMANCE ==========")

segment_columns = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "is_repeat_session"
]

segment_results = []

for model_name, prefix in [
    ("Logistic Regression", "logistic_regression"),
    ("Gradient Boosting", "gradient_boosting")
]:

    prediction_column = f"{prefix}_prediction"
    probability_column = f"{prefix}_probability"

    for segment in segment_columns:

        if segment not in prediction_data.columns:
            continue

        for segment_value, group in prediction_data.groupby(
            segment,
            dropna=False
        ):

            actual = group["actual"]
            predicted = group[prediction_column]

            segment_results.append({
                "model": model_name,
                "segment": segment,
                "segment_value": str(segment_value),
                "sessions": len(group),
                "conversions": int(actual.sum()),
                "actual_conversion_rate": actual.mean(),
                "predicted_positive_rate": predicted.mean(),
                "accuracy": accuracy_score(actual, predicted),
                "precision": precision_score(
                    actual, predicted, zero_division=0
                ),
                "recall": recall_score(
                    actual, predicted, zero_division=0
                ),
                "f1": f1_score(
                    actual, predicted, zero_division=0
                ),
                "mean_predicted_probability": group[
                    probability_column
                ].mean()
            })

segment_results = pd.DataFrame(segment_results)

segment_results.to_csv(
    ERROR_DIR / "performance_by_segment.csv",
    index=False
)

print("\nSegment analysis saved.")

# ============================================================
# 10. PROBABILITY ANALYSIS
# ============================================================

print("\n========== PREDICTED PROBABILITY ANALYSIS ==========")

probability_results = []

for model_name, prefix in [
    ("Logistic Regression", "logistic_regression"),
    ("Gradient Boosting", "gradient_boosting")
]:

    probability_column = f"{prefix}_probability"

    probability_data = prediction_data[
        ["actual", probability_column]
    ].copy()

    probability_data["probability_group"] = pd.qcut(
        probability_data[probability_column],
        q=10,
        duplicates="drop"
    )

    grouped = (
        probability_data
        .groupby("probability_group", observed=True)
        .agg(
            sessions=("actual", "size"),
            actual_conversions=("actual", "sum"),
            actual_conversion_rate=("actual", "mean"),
            average_predicted_probability=(
                probability_column,
                "mean"
            )
        )
        .reset_index()
    )

    grouped["model"] = model_name

    probability_results.append(grouped)

probability_results = pd.concat(
    probability_results,
    ignore_index=True
)

probability_results.to_csv(
    ERROR_DIR / "probability_group_analysis.csv",
    index=False
)

print(probability_results.to_string(index=False))

# ============================================================
# 11. VALIDATION THRESHOLD ANALYSIS
# ============================================================

print("\n========== VALIDATION THRESHOLD ANALYSIS ==========")

threshold_results = []

thresholds = np.arange(0.10, 0.91, 0.05)

for model_name, prefix in [
    ("Logistic Regression", "logistic_regression"),
    ("Gradient Boosting", "gradient_boosting")
]:

    probabilities = prediction_data[
        f"{prefix}_probability"
    ].to_numpy()

    actual = prediction_data["actual"].to_numpy()

    for threshold in thresholds:

        predicted = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        ).ravel()

        threshold_results.append({
            "model": model_name,
            "threshold": round(threshold, 2),
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "f1": f1_score(
                actual, predicted, zero_division=0
            ),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        })

threshold_results = pd.DataFrame(threshold_results)

threshold_results.to_csv(
    ERROR_DIR / "validation_threshold_analysis.csv",
    index=False
)

print(threshold_results.to_string(index=False))

# ============================================================
# 12. OVERALL VALIDATION PERFORMANCE
# ============================================================

print("\n========== VALIDATION PERFORMANCE SUMMARY ==========")

performance_results = []

actual = prediction_data["actual"]

for model_name, prefix in [
    ("Logistic Regression", "logistic_regression"),
    ("Gradient Boosting", "gradient_boosting")
]:

    predicted = prediction_data[
        f"{prefix}_prediction"
    ]

    probabilities = prediction_data[
        f"{prefix}_probability"
    ]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    performance_results.append({
        "model": model_name,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probabilities),
        "average_precision": average_precision_score(
            actual, probabilities
        ),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    })

performance_results = pd.DataFrame(performance_results)

performance_results.to_csv(
    ERROR_DIR / "validation_performance_summary.csv",
    index=False
)

print(performance_results.to_string(index=False))

# ============================================================
# 13. SAVE CONFIGURATION AND COMPLETION SUMMARY
# ============================================================

configuration = {
    "stage": 10,
    "purpose": "Model interpretation and error analysis",
    "validation_records": int(len(X_validation)),
    "validation_conversion_rate": float(y_validation.mean()),
    "validation_split": "Chronological 80/20 split of Stage 09 training data",
    "models_analyzed": [
        "Logistic Regression",
        "Gradient Boosting"
    ],
    "default_threshold": DEFAULT_THRESHOLD,
    "threshold_analysis_range": [
        float(thresholds.min()),
        float(thresholds.max())
    ],
    "final_test_set_used": False,
    "feature_importance_methods": {
        "logistic_regression": "Absolute and signed model coefficients",
        "gradient_boosting": "Tree-based feature importance"
    }
}

with open(
    REPORT_DIR / "stage_10_configuration.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(configuration, file, indent=4)

print("\n========== STAGE 10 COMPLETED ==========")

print("Feature importance reports:", IMPORTANCE_DIR)
print("Error analysis reports:", ERROR_DIR)

print("\nThe final test set was not used.")