"""Model-agnostic SHAP explanations for the evaluated churn pipeline.

Run: python src/explain.py --artifacts artifacts
Permutation-style SHAP uses a background subset and explains held-out examples.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap


def explain(folder: Path):
    model = joblib.load(folder / "churn_model.joblib")
    examples = joblib.load(folder / "explanation_samples.joblib")
    reference, sample = examples["reference"], examples["explain"]
    columns = list(reference.columns)
    def predict(frame):
        frame = pd.DataFrame(frame, columns=columns)
        return model.predict_proba(frame)[:, 1]
    masker = shap.maskers.Independent(reference, max_samples=min(80, len(reference)))
    explainer = shap.Explainer(predict, masker, algorithm="permutation")
    values = explainer(sample, max_evals=max(2 * len(columns) + 1, 101))
    summary = pd.DataFrame({
        "feature": columns,
        "mean_absolute_shap": np.abs(values.values).mean(axis=0)
    }).sort_values("mean_absolute_shap", ascending=False)
    summary.to_csv(folder / "shap_importance.csv", index=False)
    print(summary.head(15).to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    explain(parser.parse_args().artifacts)
