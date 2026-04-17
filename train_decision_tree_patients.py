#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a Decision Tree on a Patients dataset with:
- Holdout evaluation
- K-fold Cross-Validation
- Confusion Matrix
- Feature Importances

Outputs:
- Prints metrics to the console
- Saves plots to ./outputs/
- Saves a JSON report to ./outputs/report.json
- Saves the trained model to ./outputs/model.pkl

If "patients_dataset_v2.xlsx" isn't found, a synthetic dataset is generated.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from openpyxl.workbook import Workbook
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
OUTPUT_DIR = Path("./outputs").resolve()
DATA_PATH_DEFAULT = Path("patients_dataset_v2.xlsx")


def ensure_data(path: Path) -> pd.DataFrame:
    """Load dataset if exists; otherwise generate a synthetic one compatible with the project."""
    if path.exists():
        df = pd.read_excel(path)
        # ensure expected columns exist
        expected_cols = {
            "Age","Gender","BloodPressure","Cholesterol","Glucose","BMI","HeartRate",
            "SmokingStatus","FamilyHistory","LabTests","SymptomsScore","Diagnosis"
        }
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Input file is missing columns: {missing}")
        return df

    # Generate synthetic dataset
    np.random.seed(RANDOM_STATE)
    n_samples = 200
    df = pd.DataFrame({
        "Age": np.random.randint(20, 80, size=n_samples),
        "Gender": np.random.choice(["Male", "Female"], size=n_samples),
        "BloodPressure": np.random.randint(90, 180, size=n_samples),
        "Cholesterol": np.random.randint(150, 300, size=n_samples),
        "Glucose": np.random.randint(70, 200, size=n_samples),
        "BMI": np.round(np.random.uniform(18, 40, size=n_samples), 1),
        "HeartRate": np.random.randint(50, 120, size=n_samples),
        "SmokingStatus": np.random.choice(["Yes", "No"], size=n_samples),
        "FamilyHistory": np.random.choice(["Yes", "No"], size=n_samples),
        "LabTests": np.random.randint(50, 200, size=n_samples),
        "SymptomsScore": np.random.randint(0, 11, size=n_samples),
    })
    # Risk-based target
    risk_points = (
        (df["Age"] > 55).astype(int) +
        (df["BloodPressure"] > 140).astype(int) +
        (df["Cholesterol"] > 240).astype(int) +
        (df["Glucose"] >= 126).astype(int) +
        (df["BMI"] >= 30).astype(int) +
        (df["SmokingStatus"] == "Yes").astype(int) +
        (df["FamilyHistory"] == "Yes").astype(int) +
        (df["SymptomsScore"] >= 7).astype(int) +
        (df["LabTests"] > 120).astype(int)
    )
    df["Diagnosis"] = np.where(risk_points >= 3, "Disease", "No Disease")

    # Add a bit of noise
    flip_n = max(1, int(0.08 * n_samples))
    flip_idx = np.random.choice(df.index, size=flip_n, replace=False)
    df.loc[flip_idx, "Diagnosis"] = df.loc[flip_idx, "Diagnosis"].map(
        {"Disease": "No Disease", "No Disease": "Disease"}
    )

    # Save for reproducibility
    df.to_excel(path, index=False)
    return df


def build_pipeline(categorical: List[str], numeric: List[str]) -> Pipeline:
    """Create a preprocessing + DecisionTree pipeline."""
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    clf = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def evaluate_holdout(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["No Disease", "Disease"])
    report = classification_report(y_test, y_pred, zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fitted_pipe": pipe,
        "splits": (X_train, X_test, y_train, y_test),
    }


def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, k: int = 5) -> dict:
    scores = cross_val_score(pipe, X, y, cv=k)
    return {"fold_scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}


def extract_feature_importances(fitted_pipe: Pipeline, categorical: List[str], numeric: List[str]) -> pd.DataFrame:
    """Map tree feature importances back to original (one-hot expanded) names."""
    pre = fitted_pipe.named_steps["pre"]
    clf = fitted_pipe.named_steps["clf"]

    # Names after OneHot + passthrough
    ohe: OneHotEncoder = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical))
    feat_names = cat_names + numeric

    importances = clf.feature_importances_
    fi = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importances(fi: pd.DataFrame, top_n: int, out_path: Path) -> None:
    top = fi.head(top_n)
    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(top))
    plt.barh(y_pos, top["importance"].values)
    plt.yticks(y_pos, top["feature"].values)
    plt.title(f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_PATH_DEFAULT
    print(f"[i] Using data file: {data_path.resolve()}")

    df = ensure_data(data_path)

    # Split features / target
    target_col = "Diagnosis"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical = ["Gender", "SmokingStatus", "FamilyHistory"]
    numeric = [c for c in X.columns if c not in categorical]

    # Build pipeline
    pipe = build_pipeline(categorical, numeric)

    # Holdout evaluation
    hold = evaluate_holdout(pipe, X, y)
    print("\n=== Holdout Evaluation ===")
    print(f"Accuracy:  {hold['accuracy']:.4f}")
    print(f"Precision: {hold['precision']:.4f}")
    print(f"Recall:    {hold['recall']:.4f}")
    print(f"F1-score:  {hold['f1']:.4f}")
    print("\nClassification Report:\n", hold["classification_report"])

    # Confusion matrix plot
    cm = np.array(hold["confusion_matrix"])
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, labels=["No Disease", "Disease"], out_path=cm_path)
    print(f"[+] Saved confusion matrix plot to: {cm_path}")

    # CV evaluation
    cv = evaluate_cv(build_pipeline(categorical, numeric), X, y, k=5)
    print("\n=== Cross-Validation (5-fold) ===")
    print("Fold scores:", cv["fold_scores"])
    print(f"Mean: {cv['mean']:.4f} | Std: {cv['std']:.4f}")

    # Feature importances (from holdout-fitted model)
    fi = extract_feature_importances(hold["fitted_pipe"], categorical, numeric)
    fi_path_csv = OUTPUT_DIR / "feature_importances.csv"
    fi.to_csv(fi_path_csv, index=False)
    print(f"[+] Saved feature importances CSV to: {fi_path_csv}")

    # Plot top-N importances
    fi_plot_path = OUTPUT_DIR / "feature_importances_top10.png"
    plot_feature_importances(fi, top_n=min(10, len(fi)), out_path=fi_plot_path)
    print(f"[+] Saved feature importances plot to: {fi_plot_path}")

    # Save trained model
    model_path = OUTPUT_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(hold["fitted_pipe"], f)
    print(f"[+] Saved trained model to: {model_path}")

    # Save a JSON report
    report = {
        "holdout": {
            "accuracy": hold["accuracy"],
            "precision": hold["precision"],
            "recall": hold["recall"],
            "f1": hold["f1"],
            "confusion_matrix": hold["confusion_matrix"],
        },
        "cv": cv,
        "class_distribution": df["Diagnosis"].value_counts().to_dict(),
        "columns": df.columns.tolist(),
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved JSON report to: {report_path}")

    # Optional: save a visual of the trained tree using a temporary fit on the full dataset
    # (for visualization only; not for evaluation purposes)
    full_pipe = build_pipeline(categorical, numeric)
    full_pipe.fit(X, y)
    # Extract the trained tree estimator
    tree = full_pipe.named_steps["clf"]
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=list(fi["feature"]), class_names=["No Disease", "Disease"], filled=True)
    tree_path = OUTPUT_DIR / "decision_tree_full_fit.png"
    plt.title("Decision Tree (Full Fit)")
    plt.tight_layout()
    plt.savefig(tree_path)
    plt.close()
    print(f"[+] Saved full-fit decision tree plot to: {tree_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
