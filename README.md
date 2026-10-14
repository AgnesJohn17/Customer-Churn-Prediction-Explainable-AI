# Customer Churn Prediction & Explainable AI

Portfolio project predicting telecom customer churn, explaining risk drivers, and demonstrating a practical machine-learning workflow.

**Status:** Implementation started. No performance numbers are claimed until training and evaluation have been run on the dataset.

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
- Compare a majority-class baseline, Logistic Regression and Random Forest with accuracy, precision, recall, F1, ROC-AUC and average precision.
- Choose a model using cross-validation on training data, then report the held-out test results.
- Make individual predictions with a Streamlit app using the trained preprocessing/model pipeline.
- Add SHAP explainability once the models are validated.

## Notes
Churn is observational: feature relationships do not establish causes. The historical sample is not representative of every telecom customer population. Avoid using customer identifiers as predictors.

## Roadmap
- [x] Project structure and training/evaluation scripts
- [ ] Run against the full public dataset and verify metrics
- [ ] Add SHAP explanations and interpretation tests
- [ ] Deploy the Streamlit app
- [ ] Document validated results and screenshots
