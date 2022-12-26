from ml_steps.model_training.model import compute_model_metrics, inference
import numpy as np


class FakeModel:

    def predict(self, X):
        return X.sum(axis=1) > 2


def test_compute_model_metrics():
    y = np.arange(5) % 4
    y_preds = np.arange(5) % 4
    results = compute_model_metrics(y, y_preds)
    precision, recall, fbeta = results
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference():
    X = np.eye(5)
    model = FakeModel()
    X[3:, :] = 1
    predictions = inference(model, X)
    assert predictions.shape[0] == X.shape[0]
