from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models" / "final"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final_validation"

for directory in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


CATEGORICAL_FEATURES = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

NUMERIC_FEATURES = [
    "is_repeat_session"
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET = "converted"

DEFAULT_THRESHOLD = 0.50

RANDOM_STATE = 42


# ============================================================
# 2. LOAD TRAINING AND TEST DATA
# ============================================================

print("\n" + "=" * 65)
print("STAGE 11: FINAL VALIDATION AND DEPLOYMENT PREPARATION")
print("=" * 65)

print("\nLoading datasets...")

X_train = pd.read_csv(
    TRAIN_DIR / "X_train_core.csv"
)

X_test = pd.read_csv(
    TEST_DIR / "X_test_core.csv"
)

y_train = pd.read_csv(
    TRAIN_DIR / "y_train.csv"
).squeeze("columns")

y_test = pd.read_csv(
    TEST_DIR / "y_test.csv"
).squeeze("columns")

test_metadata = pd.read_csv(
    TEST_DIR / "test_metadata.csv"
)

print("\nDataset shapes:")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
print("Test metadata:", test_metadata.shape)


# ============================================================
# 3. DATA VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("DATA VALIDATION")
print("=" * 65)

assert list(X_train.columns) == FEATURES, (
    "Training feature columns do not match expected features."
)

assert list(X_test.columns) == FEATURES, (
    "Testing feature columns do not match expected features."
)

assert len(X_train) == len(y_train), (
    "Training feature and target lengths do not match."
)

assert len(X_test) == len(y_test), (
    "Testing feature and target lengths do not match."
)

assert len(X_test) == len(test_metadata), (
    "Test features and metadata lengths do not match."
)

assert set(y_train.unique()).issubset({0, 1}), (
    "Unexpected values found in training target."
)

assert set(y_test.unique()).issubset({0, 1}), (
    "Unexpected values found in testing target."
)

assert X_train.isnull().sum().sum() == 0, (
    "Unexpected missing values in training features."
)

assert X_test.isnull().sum().sum() == 0, (
    "Unexpected missing values in testing features."
)

print("Feature schema validation: PASSED")
print("Target validation: PASSED")
print("Missing-value validation: PASSED")
print("Metadata alignment validation: PASSED")

print("\nTraining conversion rate:", round(y_train.mean(), 4))
print("Testing conversion rate:", round(y_test.mean(), 4))


# ============================================================
# 4. REBUILD THE SELECTED LOGISTIC REGRESSION PIPELINE
# ============================================================

print("\n" + "=" * 65)
print("BUILDING FINAL LOGISTIC REGRESSION MODEL")
print("=" * 65)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=False
            )
        )
    ]
)

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        )
    ]
)

final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# 5. TRAIN FINAL MODEL ON THE FULL TRAINING SET
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINAL MODEL")
print("=" * 65)

print("\nFitting preprocessing and Logistic Regression...")

final_model.fit(X_train, y_train)

print("Final model training completed.")

print("\nTraining records used:", len(X_train))


# ============================================================
# 6. SAVE FINAL MODEL
# ============================================================

model_path = MODEL_DIR / "logistic_regression_final.joblib"

joblib.dump(final_model, model_path)

print("\nFinal model saved:")
print(model_path)


# ============================================================
# 7. FINAL TEST SET PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEST SET PREDICTIONS")
print("=" * 65)

y_probability = final_model.predict_proba(X_test)[:, 1]

y_predictions = (
    y_probability >= DEFAULT_THRESHOLD
).astype(int)

print("\nPredictions generated.")

print("Actual positive sessions:", int(y_test.sum()))
print("Predicted positive sessions:", int(y_predictions.sum()))


# ============================================================
# 8. CALCULATE FINAL EVALUATION METRICS
# ============================================================

print("\n" + "=" * 65)
print("FINAL MODEL PERFORMANCE")
print("=" * 65)

metrics = {
    "accuracy": accuracy_score(y_test, y_predictions),
    "precision": precision_score(
        y_test, y_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, y_predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, y_predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, y_probability),
    "average_precision": average_precision_score(
        y_test, y_probability
    ),
    "brier_score": brier_score_loss(
        y_test, y_probability
    )
}

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.6f}")


# ============================================================
# 9. CONFUSION MATRIX AND CLASSIFICATION REPORT
# ============================================================

cm = confusion_matrix(y_test, y_predictions)

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX")
print(cm)

print("\nConfusion matrix interpretation:")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

report = classification_report(
    y_test,
    y_predictions,
    target_names=["No Purchase", "Purchase"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)


# ============================================================
# 10. SAVE FINAL EVALUATION REPORTS
# ============================================================

metrics_path = REPORT_DIR / "final_evaluation_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

confusion_path = REPORT_DIR / "final_confusion_matrix.csv"

pd.DataFrame(
    cm,
    index=["Actual_No_Purchase", "Actual_Purchase"],
    columns=["Predicted_No_Purchase", "Predicted_Purchase"]
).to_csv(confusion_path)

report_path = REPORT_DIR / "final_classification_report.txt"

with open(report_path, "w") as file:
    file.write("FINAL LOGISTIC REGRESSION EVALUATION\n\n")
    file.write(report)
    file.write("\n\nCONFUSION MATRIX\n")
    file.write(str(cm))
    file.write("\n\nADDITIONAL METRICS\n")

    for metric_name, metric_value in metrics.items():
        file.write(
            f"{metric_name}: {metric_value:.6f}\n"
        )


# ============================================================
# 11. SAVE TEST PREDICTIONS
# ============================================================

prediction_results = test_metadata.reset_index(drop=True).copy()

prediction_results["actual_converted"] = (
    y_test.reset_index(drop=True)
)

prediction_results["purchase_probability"] = y_probability

prediction_results["predicted_converted"] = y_predictions

prediction_results["classification_threshold"] = (
    DEFAULT_THRESHOLD
)

prediction_path = REPORT_DIR / "final_test_predictions.csv"

prediction_results.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 12. PREDICTION PROBABILITY GROUPS
# ============================================================

probability_results = pd.DataFrame({
    "actual_converted": y_test.reset_index(drop=True),
    "purchase_probability": y_probability
})

probability_results["probability_group"] = pd.cut(
    probability_results["purchase_probability"],
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    include_lowest=True
)

probability_groups = (
    probability_results
    .groupby("probability_group", observed=False)
    .agg(
        session_count=("actual_converted", "size"),
        actual_conversions=("actual_converted", "sum"),
        observed_conversion_rate=("actual_converted", "mean"),
        average_predicted_probability=("purchase_probability", "mean")
    )
    .reset_index()
)

probability_groups.to_csv(
    REPORT_DIR / "final_probability_groups.csv",
    index=False
)


# ============================================================
# 13. GENERATE FINAL FIGURES
# ============================================================

# Confusion matrix

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Purchase", "Purchase"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("Final Logistic Regression Confusion Matrix")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_confusion_matrix.png",
    dpi=300
)

plt.close()


# ROC curve

fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_probability,
    ax=ax
)

ax.set_title("Final Logistic Regression ROC Curve")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_roc_curve.png",
    dpi=300
)

plt.close()


# Precision-recall curve

fig, ax = plt.subplots(figsize=(7, 6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_probability,
    ax=ax
)

ax.set_title("Final Logistic Regression Precision-Recall Curve")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_precision_recall_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. CREATE DEPLOYMENT CONFIGURATION
# ============================================================

deployment_configuration = {
    "project": "E-commerce Customer Purchase Conversion Prediction",
    "model_name": "Final Logistic Regression",
    "model_file": "logistic_regression_final.joblib",
    "target": TARGET,
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "classification_threshold": DEFAULT_THRESHOLD,
    "preprocessing": {
        "categorical_missing_values": "Most frequent imputation",
        "categorical_encoding": (
            "OneHotEncoder with handle_unknown='ignore' "
            "and min_frequency=10"
        ),
        "numeric_missing_values": "Median imputation",
        "numeric_scaling": "StandardScaler"
    },
    "classifier": {
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    }
}

configuration_path = (
    REPORT_DIR / "final_deployment_configuration.json"
)

with open(configuration_path, "w") as file:
    json.dump(deployment_configuration, file, indent=4)


# ============================================================
# 15. CREATE MODEL MANIFEST
# ============================================================

train_metadata = pd.read_csv(
    TRAIN_DIR / "train_metadata.csv"
)

train_dates = pd.to_datetime(
    train_metadata["created_at"]
)

test_dates = pd.to_datetime(
    test_metadata["created_at"]
)

manifest = {
    "model_file": "logistic_regression_final.joblib",
    "model_type": "Logistic Regression",
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "training_start": str(train_dates.min()),
    "training_end": str(train_dates.max()),
    "testing_start": str(test_dates.min()),
    "testing_end": str(test_dates.max()),
    "training_conversion_rate": float(y_train.mean()),
    "testing_conversion_rate": float(y_test.mean()),
    "threshold": DEFAULT_THRESHOLD,
    "test_metrics": {
        key: float(value)
        for key, value in metrics.items()
    },
    "features": FEATURES,
    "target": TARGET
}

manifest_path = MODEL_DIR / "model_manifest.json"

with open(manifest_path, "w") as file:
    json.dump(manifest, file, indent=4)


# ============================================================
# 16. RELOAD MODEL AND VERIFY PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("FINAL MODEL REPRODUCIBILITY CHECK")
print("=" * 65)

reloaded_model = joblib.load(model_path)

reloaded_probabilities = (
    reloaded_model.predict_proba(X_test)[:, 1]
)

maximum_difference = np.max(
    np.abs(y_probability - reloaded_probabilities)
)

reproducibility_passed = np.allclose(
    y_probability,
    reloaded_probabilities,
    atol=1e-12
)

print("Maximum probability difference:", maximum_difference)
print("Reproducibility passed:", reproducibility_passed)

assert reproducibility_passed, (
    "Reloaded model predictions do not match original predictions."
)


# ============================================================
# 17. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STAGE 11 COMPLETED")
print("=" * 65)

print("\nFINAL MODEL:")
print(model_path)

print("\nFINAL REPORTS:")
print(REPORT_DIR)

print("\nFINAL FIGURES:")
print(FIGURE_DIR)

print("\nSaved files:")
print("- Final trained Logistic Regression model")
print("- Model manifest")
print("- Deployment configuration")
print("- Final evaluation metrics")
print("- Confusion matrix")
print("- Classification report")
print("- Final test predictions")
print("- Prediction probability groups")
print("- Confusion matrix figure")
print("- ROC curve figure")
print("- Precision-recall curve figure")

print("\nFinal test evaluation completed.")
print("The final test set was not used for model fitting or tuning.")