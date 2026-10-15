"""Headless end-to-end Streamlit smoke test using a freshly trained IBM model.

Run after training:
    python scripts/smoke_test_app.py

This tests the app itself without exposing user/customer-level data.
"""
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("app.py", default_timeout=120).run()
if len(app.exception):
    raise AssertionError("Streamlit failed during startup: " + str(app.exception))
if not app.button:
    raise AssertionError("Prediction form submit button was not rendered")

# Submit default, dataset-informed feature values and check prediction output.
app.button[0].click().run()
if len(app.exception):
    raise AssertionError("Streamlit prediction failed: " + str(app.exception))
metric_labels = [m.label for m in app.metric]
if "Estimated churn probability" not in metric_labels:
    raise AssertionError("No churn probability appeared after clicking Predict")
print("PASS: Streamlit launches, loads model and predicts successfully")
