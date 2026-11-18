import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING AND TESTING DATA
# ============================================================

print("=" * 70)
print("LOADING TRAINING AND TESTING DATA")
print("=" * 70)

X_train_core = pd.read_csv(
    DATA_DIR / "train" / "X_train_core.csv"
)

X_test_core = pd.read_csv(
    DATA_DIR / "test" / "X_test_core.csv"
)

X_train_extended = pd.read_csv(
    DATA_DIR / "train" / "X_train_extended.csv"
)

X_test_extended = pd.read_csv(
    DATA_DIR / "test" / "X_test_extended.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "train" / "y_train.csv"
)["converted"]

y_test = pd.read_csv(
    DATA_DIR / "test" / "y_test.csv"
)["converted"]


print("Core training shape:", X_train_core.shape)
print("Core testing shape:", X_test_core.shape)

print("Extended training shape:", X_train_extended.shape)
print("Extended testing shape:", X_test_extended.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# ============================================================
# 3. DEFINE FEATURE GROUPS
# ============================================================

CORE_CATEGORICAL_FEATURES = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

CORE_BINARY_FEATURES = [
    "is_repeat_session"
]

TIME_FEATURES = [
    "session_hour",
    "session_day_of_week",
    "session_month"
]


# ============================================================
# 4. CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(include_time_features=False):

    categorical_features = CORE_CATEGORICAL_FEATURES.copy()

    numerical_features = CORE_BINARY_FEATURES.copy()

    if include_time_features:
        numerical_features.extend(TIME_FEATURES)

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
                    min_frequency=10
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
                categorical_features
            ),
            (
                "numerical",
                numerical_transformer,
                numerical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# 5. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

def create_model(include_time_features=False):

    preprocessor = create_preprocessor(
        include_time_features=include_time_features
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    return model


# ============================================================
# 6. TRAIN AND EVALUATE MODEL
# ============================================================

def train_and_evaluate(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    include_time_features=False
):

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model = create_model(
        include_time_features=include_time_features
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    # Predictions
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "model": model_name,

        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_probability
        ),

        "average_precision": average_precision_score(
            y_test,
            y_probability
        )
    }

    print("\nMODEL PERFORMANCE")

    for metric, value in metrics.items():

        if metric != "model":

            print(f"{metric}: {value:.4f}")

    # Save trained model
    model_path = MODEL_DIR / f"{model_name}.joblib"

    joblib.dump(model, model_path)

    print("\nSaved model:", model_path)

    # Save predictions
    predictions = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": y_pred,
        "conversion_probability": y_probability
    })

    predictions_path = (
        REPORT_DIR / f"{model_name}_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False
    )

    print("Saved predictions:", predictions_path)

    return metrics, model


# ============================================================
# 7. TRAIN CORE MODEL
# ============================================================

core_metrics, core_model = train_and_evaluate(
    model_name="logistic_regression_core",
    X_train=X_train_core,
    X_test=X_test_core,
    y_train=y_train,
    y_test=y_test,
    include_time_features=False
)


# ============================================================
# 8. TRAIN EXTENDED MODEL
# ============================================================

extended_metrics, extended_model = train_and_evaluate(
    model_name="logistic_regression_extended",
    X_train=X_train_extended,
    X_test=X_test_extended,
    y_train=y_train,
    y_test=y_test,
    include_time_features=True
)


# ============================================================
# 9. COMPARE MODELS
# ============================================================

comparison = pd.DataFrame([
    core_metrics,
    extended_metrics
])

comparison_path = (
    REPORT_DIR / "logistic_regression_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(index=False)
)

print("\nSaved comparison:", comparison_path)


# ============================================================
# 10. SAVE MODEL CONFIGURATION
# ============================================================

configuration = {
    "algorithm": "Logistic Regression",

    "class_weight": "balanced",

    "max_iter": 1000,

    "random_state": 42,

    "categorical_encoding": "OneHotEncoder",

    "unknown_categories": "ignore",

    "rare_category_threshold": 10,

    "numerical_scaling": "StandardScaler",

    "validation_strategy": "Chronological 80/20 split",

    "models": [
        "logistic_regression_core",
        "logistic_regression_extended"
    ]
}

configuration_path = (
    REPORT_DIR / "logistic_regression_configuration.json"
)

with open(configuration_path, "w") as file:

    json.dump(
        configuration,
        file,
        indent=4
    )

print("\nSaved configuration:", configuration_path)

print("\nSTAGE 07 COMPLETED SUCCESSFULLY.")