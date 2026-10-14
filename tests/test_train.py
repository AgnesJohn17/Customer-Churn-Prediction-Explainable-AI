import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import prepare, make_pipeline
from sklearn.linear_model import LogisticRegression

class TestPipeline(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({
            "customerID": ["a", "b", "c", "d", "e", "f"],
            "tenure": [1, 4, 12, 5, 20, 2],
            "TotalCharges": [" ", "120.2", "400", "95", "700", "20"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "One year"],
            "Churn": ["Yes", "No", "No", "Yes", "No", "Yes"],
        })
    def test_preparation(self):
        X, y = prepare(self.sample())
        self.assertNotIn("customerID", X.columns)
        self.assertEqual(int(X["TotalCharges"].isna().sum()), 1)
        self.assertEqual(y.sum(), 3)
    def test_model_fit(self):
        X, y = prepare(self.sample())
        model = make_pipeline(X, LogisticRegression(max_iter=300))
        model.fit(X, y)
        self.assertEqual(len(model.predict_proba(X)), 6)
    def test_invalid_labels(self):
        df = self.sample()
        df.loc[0, "Churn"] = "Unknown"
        with self.assertRaises(ValueError):
            prepare(df)

if __name__ == "__main__":
    unittest.main()
