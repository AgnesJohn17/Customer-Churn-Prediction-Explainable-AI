import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import prepare

class AdditionalTests(unittest.TestCase):
    def test_no_identifier_as_feature(self):
        df = pd.DataFrame({"customerID": ["a", "b"], "TotalCharges": [" ", "123.45"],
                           "Contract": ["Month-to-month", "Two year"], "Churn": ["Yes", "No"]})
        X, y = prepare(df)
        self.assertNotIn("customerID", X.columns)
        self.assertEqual(X["TotalCharges"].isna().sum(), 1)

if __name__ == "__main__":
    unittest.main()
