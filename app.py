"""Streamlit demo for the evaluated, locally trained churn model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Risk Explorer", page_icon="📊")
st.title("Customer Churn Risk Explorer")
st.caption("Demonstration model — predictions are not causal evidence or business decisions.")
location = Path("artifacts")
if not (location / "churn_model.joblib").exists():
    st.warning("Train the model first: python src/train.py --data data/telco_churn.csv --output artifacts")
    st.stop()

model = joblib.load(location / "churn_model.joblib")
schema = joblib.load(location / "schema.joblib")
values = {}
with st.form("customer"):
    for name in schema["features"]:
        if schema["types"][name] == "numeric":
            values[name] = st.number_input(name, value=0.0)
        else:
            opts = schema["categories"][name]
            values[name] = st.selectbox(name, opts)
    submitted = st.form_submit_button("Predict churn risk")
if submitted:
    probability = float(model.predict_proba(pd.DataFrame([values]))[0, 1])
    st.metric("Predicted churn probability", f"{probability:.1%}")
    st.progress(probability)
    st.caption("This is a model-estimated probability; thresholds require business validation.")
