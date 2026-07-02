from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
DOCS_DIR = Path("docs")

CLEANED_RESUMES_FILE = PROCESSED_DIR / "cleaned_resumes.csv"

MODEL_FILE = MODELS_DIR / "logistic_regression_baseline.pkl"
RESULTS_FILE = DOCS_DIR / "baseline_results.md"


def score_to_label(score):
    """
    Convert matched_score into a 3-class label.

    0 = low match
    1 = medium match
    2 = high match
    """
    try:
        score = float(score)
    except Exception:
        return None

    if score < 0.50:
        return 0
    elif score < 0.75:
        return 1
    else:
        return 2


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned resume data...")
    resumes = pd.read_csv(CLEANED_RESUMES_FILE)

    print(f"Original resumes shape: {resumes.shape}")

    print("\nCreating 3-class labels from matched_score...")
    resumes["label"] = resumes["matched_score"].apply(score_to_label)

    # Remove rows without label or text
    df = resumes.dropna(subset=["label", "clean_resume_text"]).copy()
    df["label"] = df["label"].astype(int)
    df["clean_resume_text"] = df["clean_resume_text"].fillna("")

    print(f"Dataset after cleaning labels: {df.shape}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    X = df["clean_resume_text"]
    y = df["label"]

    print("\nSplitting data into train and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    print("\nTraining TF-IDF + Logistic Regression model...")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["low_match", "medium_match", "high_match"],
        digits=4,
    )
    matrix = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(matrix)

    print(f"\nSaving model to: {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)

    print(f"Saving results report to: {RESULTS_FILE}")

    with open(RESULTS_FILE, "w") as f:
        f.write("# Baseline Evaluation Results\n\n")

        f.write("## Model\n\n")
        f.write("TF-IDF + Logistic Regression baseline model.\n\n")

        f.write("## Label Definition\n\n")
        f.write("- 0 = low match: matched_score < 0.50\n")
        f.write("- 1 = medium match: 0.50 <= matched_score < 0.75\n")
        f.write("- 2 = high match: matched_score >= 0.75\n\n")

        f.write("## Dataset\n\n")
        f.write(f"- Total rows after cleaning: {len(df)}\n")
        f.write(f"- Train size: {len(X_train)}\n")
        f.write(f"- Test size: {len(X_test)}\n\n")

        f.write("## Accuracy\n\n")
        f.write(f"{accuracy:.4f}\n\n")

        f.write("## Precision, Recall, and F1-score\n\n")
        f.write("```text\n")
        f.write(report)
        f.write("```\n\n")

        f.write("## Confusion Matrix\n\n")
        f.write("```text\n")
        f.write(str(matrix))
        f.write("\n```\n\n")

        f.write("## Notes\n\n")
        f.write(
            "This is a baseline model. The labels are generated from the "
            "matched_score column in the resume dataset, so the results should "
            "be interpreted as a starting point for comparison rather than a "
            "final production-quality evaluation.\n"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()