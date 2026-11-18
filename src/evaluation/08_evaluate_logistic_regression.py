import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

FIGURE_DIR = PROJECT_ROOT / "figures" / "model_evaluation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD ACTUAL TARGETS AND MODEL PREDICTIONS
# ============================================================

print("=" * 70)
print("LOADING MODEL PREDICTIONS")
print("=" * 70)

y_test = pd.read_csv(
    DATA_DIR / "test" / "y_test.csv"
)["converted"]

core_predictions = pd.read_csv(
    REPORT_DIR / "logistic_regression_core_predictions.csv"
)

extended_predictions = pd.read_csv(
    REPORT_DIR / "logistic_regression_extended_predictions.csv"
)


models = {
    "Core Logistic Regression": core_predictions,
    "Extended Logistic Regression": extended_predictions
}


# ============================================================
# 3. EVALUATE EACH MODEL
# ============================================================

evaluation_results = []

threshold_results = []

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


for model_name, predictions in models.items():

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)

    y_pred = predictions["predicted"]

    y_probability = predictions["conversion_probability"]

    # --------------------------------------------------------
    # A. CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\nCONFUSION MATRIX")

    print(cm)

    print("\nCONFUSION MATRIX INTERPRETATION")

    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    # Save confusion matrix figure
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=",",
        cmap="Blues",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"]
    )

    plt.title(f"{model_name} - Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    figure_path = (
        FIGURE_DIR /
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(figure_path, dpi=300)

    plt.close()

    # --------------------------------------------------------
    # B. CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nCLASSIFICATION REPORT")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Purchase", "Purchase"],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # C. SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "model": model_name,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        )
    }

    evaluation_results.append(metrics)

    # --------------------------------------------------------
    # D. THRESHOLD ANALYSIS
    # --------------------------------------------------------

    for threshold in thresholds:

        threshold_predictions = (
            y_probability >= threshold
        ).astype(int)

        threshold_results.append({

            "model": model_name,

            "threshold": threshold,

            "accuracy": accuracy_score(
                y_test,
                threshold_predictions
            ),

            "precision": precision_score(
                y_test,
                threshold_predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                threshold_predictions,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_test,
                threshold_predictions,
                zero_division=0
            ),

            "predicted_conversions": int(
                threshold_predictions.sum()
            )
        })


# ============================================================
# 4. SAVE EVALUATION RESULTS
# ============================================================

evaluation_df = pd.DataFrame(evaluation_results)

evaluation_path = (
    REPORT_DIR / "logistic_regression_detailed_evaluation.csv"
)

evaluation_df.to_csv(
    evaluation_path,
    index=False
)


# ============================================================
# 5. SAVE THRESHOLD ANALYSIS
# ============================================================

threshold_df = pd.DataFrame(threshold_results)

threshold_path = (
    REPORT_DIR / "logistic_regression_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)


# ============================================================
# 6. PLOT THRESHOLD PERFORMANCE
# ============================================================

for model_name in threshold_df["model"].unique():

    model_data = threshold_df[
        threshold_df["model"] == model_name
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        model_data["threshold"],
        model_data["precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        model_data["threshold"],
        model_data["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        model_data["threshold"],
        model_data["f1_score"],
        marker="o",
        label="F1-score"
    )

    plt.xlabel("Classification Threshold")

    plt.ylabel("Score")

    plt.title(f"{model_name} - Threshold Analysis")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    figure_path = (
        FIGURE_DIR /
        f"{model_name.lower().replace(' ', '_')}_threshold_analysis.png"
    )

    plt.savefig(figure_path, dpi=300)

    plt.close()


# ============================================================
# 7. PRINT FINAL SUMMARIES
# ============================================================

print("\n" + "=" * 70)
print("DETAILED MODEL EVALUATION")
print("=" * 70)

print(
    evaluation_df.to_string(index=False)
)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    threshold_df.to_string(index=False)
)

print("\nSaved evaluation:", evaluation_path)

print("Saved threshold analysis:", threshold_path)

print("Saved figures:", FIGURE_DIR)

print("\nSTAGE 08 COMPLETED SUCCESSFULLY.")