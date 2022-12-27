from mlops_fastapi.model_training.model import (
    compute_model_metrics,
    inference,
    ClassifierWrapper,
)
import numpy as np
from sklearn import preprocessing
from mock_classes import MockModel, MockEncoder
import pytest


@pytest.fixture
def data():
    n = 5
    return [{"x": x, "z": x % 2} for x in np.arange(n)]


@pytest.fixture
def target():
    return np.array(["no", "no", "no", "yes", "yes"])


@pytest.fixture
def classifier_wrapper():
    wrapper = ClassifierWrapper(
        MockModel(), MockEncoder(), preprocessing._label.LabelBinarizer()
    )
    wrapper.target_encoder.fit(["yes", "no"])
    return wrapper


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
    model = MockModel()
    X[3:, :] = 1
    predictions = inference(model, X)
    assert predictions.shape[0] == X.shape[0]


def test_classifier_wrapper_predict(classifier_wrapper, data, target):

    for i in range(len(data)):
        prediction = classifier_wrapper.predict_single(data[i])
        assert prediction == target[i]


def test_classifier_wrapper_predict_proba(classifier_wrapper, data, target):
    proba_predictions = classifier_wrapper.predict_single_proba(data[0])
    assert set(proba_predictions.keys()) == {"yes", "no"}
    for prob in proba_predictions.values():
        assert 0 <= prob <= 1
