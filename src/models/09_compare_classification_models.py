import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models" / "stage_09"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD CORE TRAINING DATA
# ============================================================

print("=" * 70)
print("LOADING CORE TRAINING DATA")
print("=" * 70)

X = pd.read_csv(
    DATA_DIR / "train" / "X_train_core.csv"
)

y = pd.read_csv(
    DATA_DIR / "train" / "y_train.csv"
)["converted"]

metadata = pd.read_csv(
    DATA_DIR / "train" / "train_metadata.csv"
)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Metadata shape:", metadata.shape)


# ============================================================
# 3. CREATE CHRONOLOGICAL TRAINING/VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING CHRONOLOGICAL VALIDATION SPLIT")
print("=" * 70)

dates = pd.to_datetime(metadata["created_at"])

if not (dates.is_monotonic_increasing):
    raise ValueError(
        "Training metadata is not chronologically sorted."
    )

split_index = int(len(X) * 0.80)

split_timestamp = dates.iloc[split_index]

train_mask = dates < split_timestamp

validation_mask = dates >= split_timestamp

X_train = X.loc[train_mask].reset_index(drop=True)

X_validation = X.loc[validation_mask].reset_index(drop=True)

y_train = y.loc[train_mask].reset_index(drop=True)

y_validation = y.loc[validation_mask].reset_index(drop=True)

train_dates = dates.loc[train_mask]

validation_dates = dates.loc[validation_mask]


print("\nTraining sessions:", len(X_train))

print("Validation sessions:", len(X_validation))

print("\nTraining date range:")

print("Start:", train_dates.min())

print("End:", train_dates.max())

print("\nValidation date range:")

print("Start:", validation_dates.min())

print("End:", validation_dates.max())

print("\nTraining conversion rate:")

print(f"{y_train.mean():.4%}")

print("\nValidation conversion rate:")

print(f"{y_validation.mean():.4%}")


if train_dates.max() >= validation_dates.min():

    raise ValueError(
        "Chronological separation failed."
    )

print("\nChronological validation split passed.")


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

NUMERICAL_FEATURES = [
    "is_repeat_session"
]


# ============================================================
# 5. CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor():

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10 ,
                    sparse_output=False
                )
            )
        ]
    )

    numerical_transformer = Pipeline(
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
                categorical_transformer,
                CATEGORICAL_FEATURES
            ),

            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ============================================================
# 6. DEFINE CLASSIFICATION MODELS
# ============================================================

models = {

    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=75,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        random_state=42
    ),

    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

prediction_files = []

for model_name, classifier in models.items():

    print("\n" + "=" * 70)

    print(f"TRAINING MODEL: {model_name.upper()}")

    print("=" * 70)

    # Build pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),

            ("classifier", classifier)
        ]
    )

    # Handle class imbalance
    if model_name in [
        "gradient_boosting",
        "hist_gradient_boosting"
    ]:

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        pipeline.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights
        )

    else:

        pipeline.fit(
            X_train,
            y_train
        )

    print("Training completed.")

    # Validation predictions
    y_pred = pipeline.predict(X_validation)

    y_probability = pipeline.predict_proba(
        X_validation
    )[:, 1]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Evaluation metrics
    metrics = {

        "model": model_name,

        "accuracy": accuracy_score(
            y_validation,
            y_pred
        ),

        "precision": precision_score(
            y_validation,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_validation,
            y_pred,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_validation,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_validation,
            y_probability
        ),

        "average_precision": average_precision_score(
            y_validation,
            y_probability
        ),

        "true_negatives": tn,

        "false_positives": fp,

        "false_negatives": fn,

        "true_positives": tp

    }

    results.append(metrics)

    # Save validation predictions
    predictions = pd.DataFrame({

        "actual": y_validation,

        "predicted": y_pred,

        "conversion_probability": y_probability

    })

    prediction_path = (
        REPORT_DIR /
        f"{model_name}_validation_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False
    )

    prediction_files.append(prediction_path)

    # Save trained model
    model_path = (
        MODEL_DIR /
        f"{model_name}.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print("\nVALIDATION PERFORMANCE")

    for metric, value in metrics.items():

        if metric != "model":

            print(f"{metric}: {value:.4f}")

    print("\nSaved model:", model_path)


# ============================================================
# 8. COMPARE ALL MODELS
# ============================================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="average_precision",
    ascending=False
)

comparison_path = (
    REPORT_DIR /
    "stage_09_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 9. SAVE EXPERIMENT CONFIGURATION
# ============================================================

configuration = {

    "stage": 9,

    "objective": "Compare classification algorithms",

    "features": CATEGORICAL_FEATURES + NUMERICAL_FEATURES,

    "validation_strategy": (
        "Chronological 80/20 split of original training data"
    ),

    "final_test_set_used": False,

    "class_imbalance_handling": {

        "logistic_regression": "class_weight=balanced",

        "random_forest": "class_weight=balanced_subsample",

        "gradient_boosting": "balanced sample weights",

        "hist_gradient_boosting": "balanced sample weights"

    },

    "models": list(models.keys()),

    "primary_comparison_metric": "average_precision",

    "secondary_comparison_metric": "roc_auc"

}

import json

configuration_path = (
    REPORT_DIR /
    "stage_09_configuration.json"
)

with open(configuration_path, "w") as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


# ============================================================
# 10. PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 90)

print("STAGE 09 — MODEL COMPARISON RESULTS")

print("=" * 90)

print(
    comparison.to_string(index=False)
)

print("\nSaved comparison:", comparison_path)

print("Saved configuration:", configuration_path)

print("\nSTAGE 09 COMPLETED SUCCESSFULLY.")