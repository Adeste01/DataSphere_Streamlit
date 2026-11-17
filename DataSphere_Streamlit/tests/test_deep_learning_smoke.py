"""Small CPU-only smoke tests for NusaInsight's PyTorch tabular model."""
import unittest

import numpy as np
import pandas as pd

from app_modules.deep_learning import _prepare_arrays, _train_mlp


class DeepLearningSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(7)

    def test_classification_training(self):
        features = self.rng.normal(size=(90, 4))
        frame = pd.DataFrame(features, columns=["f1", "f2", "f3", "f4"])
        labels = np.array(["class_a"] * 30 + ["class_b"] * 30 + ["class_c"] * 30)
        frame["target"] = labels
        frame["class_name"] = labels
        prepared = _prepare_arrays(frame, ["f1", "f2", "f3", "f4"], "Classification", 0.2, 42)
        x_train, x_test, y_train, y_test, *_ = prepared
        _, metrics, history, predictions = _train_mlp(
            x_train, x_test, y_train, y_test, "Classification", 2, 42
        )
        self.assertEqual(len(predictions), len(y_test))
        self.assertEqual(len(history), 2)
        self.assertIn("Accuracy", metrics)
        self.assertIn("F1-score", metrics)

    def test_regression_training(self):
        frame = pd.DataFrame(self.rng.normal(size=(100, 4)), columns=["x1", "x2", "x3", "x4"])
        frame["target"] = 2 * frame["x1"] - frame["x2"] + self.rng.normal(0, 0.1, 100)
        prepared = _prepare_arrays(frame, ["x1", "x2", "x3", "x4"], "Regression", 0.2, 42)
        x_train, x_test, y_train, y_test, *_ = prepared
        _, metrics, history, predictions = _train_mlp(
            x_train, x_test, y_train, y_test, "Regression", 2, 42
        )
        self.assertEqual(len(predictions), len(y_test))
        self.assertEqual(len(history), 2)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)


if __name__ == "__main__":
    unittest.main()
