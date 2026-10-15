"""SHAP permutation explanations for the fitted churn model.

Explain model probabilities in encoded numeric feature space to avoid SHAP
tabular maskers trying to subtract raw categorical string values.

Run: python src/explain.py --artifacts artifacts
"""
import argparse
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def numeric_matrix(preprocessor, rows):
    transformed = preprocessor.transform(rows)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float)


def explain(folder: Path):
    model = joblib.load(folder / "churn_model.joblib")
    examples = joblib.load(folder / "explanation_samples.joblib")
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    reference = numeric_matrix(preprocessor, examples["reference"])
    sample = numeric_matrix(preprocessor, examples["explain"])
    names = preprocessor.get_feature_names_out().tolist()
    if reference.shape[1] != len(names) or sample.shape[1] != len(names):
        raise ValueError("Encoded feature columns do not match feature names")

    # The classifier consumes exactly the preprocessed values explained below.
    def predict_probability(encoded_rows):
        return classifier.predict_proba(encoded_rows)[:, 1]

    masker = shap.maskers.Independent(reference, max_samples=min(80, len(reference)))
    explainer = shap.Explainer(predict_probability, masker, algorithm="permutation")
    values = explainer(sample, max_evals=max(2 * len(names) + 1, 101))
    summary = pd.DataFrame({
        "feature": names,
        "mean_absolute_shap": np.abs(values.values).mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)
    folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(folder / "shap_importance.csv", index=False)

    top = summary.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["mean_absolute_shap"])
    plt.title("Global SHAP importance (encoded features)")
    plt.xlabel("Mean absolute impact on predicted churn probability")
    plt.tight_layout()
    plt.savefig(folder / "shap_importance.png", dpi=160)
    plt.close()
    print(summary.head(15).to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    explain(parser.parse_args().artifacts)
