"""Train churn classifiers with preprocessing fit only to the training split."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
DROP = ("customerID",)


def prepare(df):
    if TARGET not in df.columns:
        raise ValueError("Dataset must contain a Churn column")
    frame = df.copy()
    frame["Churn"] = frame[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    if frame[TARGET].isna().any():
        raise ValueError("Churn must consist of Yes/No values")
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    features = frame.drop(columns=[TARGET, *[c for c in DROP if c in frame.columns]])
    # Avoid silently treating blanks as distinct categories.
    features = features.replace(r"^\s*$", pd.NA, regex=True)
    return features, frame[TARGET].astype(int)


def make_pipeline(X, classifier):
    nums = list(X.select_dtypes(include="number").columns)
    cats = [c for c in X.columns if c not in nums]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), nums),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cats)
    ])
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def scores(model, X, y):
    pred = model.predict(X)
    result = {k: round(float(metric(y, pred, zero_division=0)), 4) for k, metric in [
        ("precision", precision_score), ("recall", recall_score), ("f1", f1_score)
    ]}
    result["accuracy"] = round(float(accuracy_score(y, pred)), 4)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        result["roc_auc"] = round(float(roc_auc_score(y, prob)), 4)
        result["average_precision"] = round(float(average_precision_score(y, prob)), 4)
    return result


def train(data, output):
    X, y = prepare(pd.read_csv(data))
    if y.nunique() != 2:
        raise ValueError("Training data must include both churn classes")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(X_train, estimator)
        cv_results[name] = round(float(cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="average_precision").mean()), 4)
    winner = max(cv_results, key=cv_results.get)
    results = {"train_rows": len(X_train), "test_rows": len(X_test),
               "churn_rate": round(float(y.mean()), 4),
               "cross_validation_average_precision": cv_results,
               "selected_model": winner, "held_out_test": {}}
    fitted = {}
    for name, estimator in candidates.items():
        model = make_pipeline(X_train, estimator)
        model.fit(X_train, y_train)
        fitted[name] = model
        results["held_out_test"][name] = scores(model, X_test, y_test)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[winner], output / "churn_model.joblib")
    joblib.dump({"features": X.columns.tolist(), "types": {
        col: ("numeric" if pd.api.types.is_numeric_dtype(X[col]) else "categorical")
        for col in X.columns}, "categories": {
            col: sorted(X[col].dropna().astype(str).unique().tolist())
            for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
        }}, output / "schema.joblib")
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train(args.data, args.output)
