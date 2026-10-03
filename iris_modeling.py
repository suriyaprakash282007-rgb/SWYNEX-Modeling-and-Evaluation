"""Train and evaluate a Logistic Regression model on the local Iris CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "iris_dataset.csv"
MODEL_PATH = BASE_DIR / "iris_logistic_regression.joblib"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET = "species"


def main() -> None:
    """Load data, inspect it, train the baseline, evaluate, and save outputs."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    expected_columns = FEATURES + [TARGET]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Expected CSV columns: {expected_columns}; got {list(df.columns)}")

    print("\n=== Dataset overview ===")
    print(f"Shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nColumn types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df[FEATURES].describe().round(2))
    print("\nMissing values by column:")
    print(df.isna().sum())
    duplicate_count = int(df.duplicated().sum())
    print(f"\nExact duplicate rows: {duplicate_count}")
    print("\nClass counts:")
    print(df[TARGET].value_counts().sort_index())

    # Keep this check explicit so data problems are visible before training.
    if df.isna().any().any():
        raise ValueError("The dataset contains missing values; resolve them before training.")
    if df.duplicated().any():
        print("Note: duplicate rows exist; they are retained to preserve the supplied dataset.")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"\nTraining rows: {len(X_train)} | Test rows: {len(X_test)}")

    # Scaling is fitted only on training data through the pipeline, preventing leakage.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    labels = sorted(y.unique())

    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision (macro): {precision_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro):    {recall_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(f"F1 (macro):        {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Iris Classification — Confusion Matrix")
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()
    print(f"Confusion matrix image saved to: {CONFUSION_MATRIX_PATH.name}")

    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURES)
    sample_prediction = model.predict(sample)[0]
    print("\n=== Sample prediction ===")
    print(f"Measurements (cm): {sample.iloc[0].to_dict()}")
    print(f"Predicted species: {sample_prediction}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nTrained pipeline saved to: {MODEL_PATH.name}")
    print("The saved pipeline includes both scaling and the classifier.")


if __name__ == "__main__":
    main()
