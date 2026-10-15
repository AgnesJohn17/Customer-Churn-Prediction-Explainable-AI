# Customer Churn Prediction & Explainable AI

**[🚀 Try the Live Streamlit App](https://customer-churn-prediction-explainable-ai-eycmnbvby7uikvbt6gy7p.streamlit.app/)** · [View validated GitHub Actions](https://github.com/AgnesJohn17/Customer-Churn-Prediction-Explainable-AI/actions) · [Model comparison](reports/model_evaluation.csv)

Portfolio project predicting telecom customer churn, explaining risk drivers, and demonstrating a practical machine-learning workflow.

**Status:** Public Streamlit deployment shared and shown running by the project owner. Automated unit tests, end-to-end Streamlit prediction smoke tests, training and global SHAP analysis passed on GitHub Actions. The live app retrains at startup; its exact metrics may vary slightly with installed dependency versions.

## Dataset

Download the IBM Telco Customer Churn sample dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) and save it as `data/telco_churn.csv`. The dataset is not bundled in this repository.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data data/telco_churn.csv --output artifacts
python -m unittest discover -s tests -v
streamlit run app.py
```

## Approach
- Prepare data in a leakage-aware, stratified train/test split.
- Handle missing numeric values and one-hot encode categorical predictors inside sklearn pipelines.
- Compare a majority-class baseline, Logistic Regression, Random Forest and XGBoost with accuracy, precision, recall, F1, ROC-AUC and average precision.
- Choose a model using cross-validation on training data, then report the held-out test results.
- Make individual predictions with a Streamlit app using the trained preprocessing/model pipeline.
- Explain the selected model with global SHAP permutation feature importance.

## Notes
Churn is observational: feature relationships do not establish causes. The historical sample is not representative of every telecom customer population. Avoid using customer identifiers as predictors.

## Roadmap
- [x] Project structure and training/evaluation scripts
- [x] Run against the full public dataset and verify metrics
- [x] Generate SHAP global feature importance in GitHub Actions
- [x] Publish Streamlit app (owner-shared URL and screenshot)
- [x] Document validated results and link live demo

## Reproducible training and explainability

A [GitHub Actions workflow](.github/workflows/train-evaluate.yml) downloads the public IBM dataset, verifies its 7,043-row count, trains/evaluates models, computes permutation SHAP explanations and uploads `metrics.json` / `shap_importance.csv` as a workflow artifact. Open **Actions → Train and evaluate churn model** to inspect the run. This is distinct from the unit-test workflow.

Run locally:
```bash
python src/train.py --data data/telco_churn.csv --output artifacts
python src/explain.py --artifacts artifacts
```

**Interpretation:** Model selection uses training-only cross-validation scored by average precision; test metrics use a separate held-out set. SHAP results describe model behavior, not causal explanations. Customer records and fitted artifacts are git-ignored. Real-world model accuracy and explanations must be reviewed from completed runs before quoting any measured performance.

## Verified IBM Telco evaluation

In the successful [training and SHAP workflow](https://github.com/AgnesJohn17/Customer-Churn-Prediction-Explainable-AI/actions/runs/38657594168), 7,043 customer records were split into 5,634 training and 1,409 test examples. Model selection used five-fold training-only average-precision cross-validation. **XGBoost** ranked highest (CV average precision **0.6694**). On the held-out test: **accuracy 80.48%, ROC-AUC 0.8470, precision 66.67%, recall 52.94%, average precision 0.6654**. At the default decision threshold, recall is limited; compare business tradeoffs before using a model in practice.

The SHAP step completed successfully and identified **month-to-month contracts, tenure, fibre-optic internet and monthly charges** among the highest mean-absolute-impact encoded features. These are predictive associations, not proven causes of churn.

## Streamlit app

Run `streamlit run app.py` locally. On its first launch it downloads the 7,043-row IBM public dataset, trains the model if necessary, and caches it. First launch may take longer; internet access is needed. **Public demo:** [Open the deployed application](https://customer-churn-prediction-explainable-ai-eycmnbvby7uikvbt6gy7p.streamlit.app/). The owner supplied this URL and a screenshot of the running app, and the repository's automated Streamlit launch-and-prediction test passed. The exact public endpoint was not independently exercised during this README update. To deploy independently, connect the repo at [Streamlit Community Cloud](https://share.streamlit.io/) and set the main file to `app.py`. No passwords or customer records should be committed to GitHub.

## Internship portfolio summary

Built an end-to-end telecom customer churn prediction application in **Python, pandas, scikit-learn, XGBoost, SHAP and Streamlit**, with a reproducible pipeline, five-fold cross-validation, held-out evaluation, and CI-based automated tests. On the 7,043-row IBM Telco dataset, the selected XGBoost model achieved **0.847 ROC-AUC and 80.48% accuracy** on the reference held-out evaluation. Published an interactive web app with individual churn-risk estimates, model comparisons and global SHAP feature-importance visualizations.
